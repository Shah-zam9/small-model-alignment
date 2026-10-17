"""Deterministic greedy generation on fixed evaluation sets with EOS/truncation tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.align.prompt import format_prompt


def generate_on_eval_set(
    model_name_or_path: str,
    eval_set_path: str | Path,
    output_path: str | Path,
    generator_name: str,
    dataset_name: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    use_chat_template: bool = False,
    device: str | None = None,
    limit: int | None = None,
    sanity_print_count: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run greedy generation on an evaluation JSON file without post-processing outputs.

    Records per sample whether generation stopped on EOS or hit max_new_tokens, and
    reports the percentage of samples that hit max_new_tokens.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)
    if limit is not None:
        eval_items = eval_items[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype).to(device)
    model.eval()

    gen_settings = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "use_chat_template": use_chat_template,
        "prompt_template": "chat_template" if use_chat_template else "### Instruction:\\n{instruction}\\n\\n### Response:\\n",
    }

    results: list[dict[str, Any]] = []
    stopped_on_eos_count = 0
    hit_max_new_tokens_count = 0
    start_time = time.time()

    for start in tqdm(range(0, len(eval_items), batch_size), desc=f"Generating ({generator_name})"):
        batch_items = eval_items[start : start + batch_size]
        prompts: list[str] = []
        for item in batch_items:
            inst = item["instruction"]
            if use_chat_template:
                prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": inst}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                prompts.append(format_prompt(inst))

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        input_width = enc["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)

        for row_idx, item in enumerate(batch_items):
            new_tokens = out_ids[row_idx, input_width:]
            eos_positions = (
                (new_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                if eos_token_id is not None
                else torch.tensor([], dtype=torch.long)
            )
            if len(eos_positions) > 0:
                first_eos = int(eos_positions[0].item())
                stopped_on_eos = True
                hit_limit = False
                content_tokens = new_tokens[:first_eos]
                num_gen_tokens = first_eos + 1
                stopped_on_eos_count += 1
            else:
                stopped_on_eos = False
                hit_limit = len(new_tokens) >= max_new_tokens
                content_tokens = new_tokens
                num_gen_tokens = int(len(new_tokens))
                if hit_limit:
                    hit_max_new_tokens_count += 1

            # Keep raw decoded output without stripping or post-processing
            raw_text = tokenizer.decode(content_tokens, skip_special_tokens=True)

            record: dict[str, Any] = {
                "id": item.get("id", f"item-{start + row_idx:04d}"),
                "instruction": item["instruction"],
                "output": raw_text,
                "generator": generator_name,
                "dataset": dataset_name or item.get("dataset", "unknown"),
                "stopped_on_eos": stopped_on_eos,
                "hit_max_new_tokens": hit_limit,
                "num_generated_tokens": num_gen_tokens,
                "generation_settings": gen_settings,
            }
            if "category" in item:
                record["category"] = item["category"]
            if "expected_behavior" in item:
                record["expected_behavior"] = item["expected_behavior"]
            results.append(record)

            if len(results) <= sanity_print_count:
                print(
                    f"\n--- Sanity Sample {len(results)} ({generator_name}) "
                    f"[tokens={num_gen_tokens}, stopped_on_eos={stopped_on_eos}, hit_limit={hit_limit}] ---"
                )
                print(f"ID: {record['id']}")
                print(f"Instruction: {record['instruction'][:200]}")
                print(f"Output: {record['output'][:400]}")

    elapsed = time.time() - start_time
    total_n = len(results)
    pct_hit_limit = round(100.0 * hit_max_new_tokens_count / max(1, total_n), 2)
    pct_stopped_eos = round(100.0 * stopped_on_eos_count / max(1, total_n), 2)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "generator": generator_name,
        "dataset": dataset_name or "unknown",
        "output_path": str(out_file),
        "total_samples": total_n,
        "stopped_on_eos_count": stopped_on_eos_count,
        "pct_stopped_on_eos": pct_stopped_eos,
        "hit_max_new_tokens_count": hit_max_new_tokens_count,
        "pct_hit_max_new_tokens": pct_hit_limit,
        "elapsed_sec": round(elapsed, 2),
        "generation_settings": gen_settings,
    }
    summary_file = out_file.with_name(out_file.stem + "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"GEN SUMMARY ({generator_name}):", json.dumps(summary, indent=2))
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model outputs on a fixed evaluation subset.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--eval_set_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--generator_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    generate_on_eval_set(
        model_name_or_path=args.model_name_or_path,
        eval_set_path=args.eval_set_path,
        output_path=args.output_path,
        generator_name=args.generator_name,
        dataset_name=args.dataset_name,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_chat_template=args.use_chat_template,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
