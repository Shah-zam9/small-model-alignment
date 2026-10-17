"""Dataset loading, single-turn filtering, token-length filtering, and manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.align.prompt import format_prompt

SEED = 42
HUMAN_TAG = "\n\nHuman: "
ASSISTANT_TAG = "\n\nAssistant: "


def build_alpaca_eval_subset(
    output_path: str | Path = "eval_sets/alpaca_eval_300_seed42.json",
    num_samples: int = 300,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Download tatsu-lab/alpaca_eval and sample a fixed 300-prompt subset with seed 42."""
    raw_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    indices = list(range(len(raw_data)))
    rng = random.Random(seed)
    sampled_indices = rng.sample(indices, num_samples)

    subset: list[dict[str, Any]] = []
    for out_idx, raw_idx in enumerate(sampled_indices):
        row = raw_data[raw_idx]
        subset.append(
            {
                "id": f"alpaca-eval-{seed}-{out_idx:04d}",
                "raw_index": raw_idx,
                "instruction": row["instruction"].strip(),
                "dataset": "alpaca_eval",
                "source_dataset": row.get("dataset", ""),
            }
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    return subset


def extract_ultrachat_first_turn(row: dict[str, Any]) -> tuple[str, str] | None:
    """Extract first user instruction and first assistant response from an UltraChat row."""
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    first_msg, second_msg = messages[0], messages[1]
    if first_msg.get("role") != "user" or second_msg.get("role") != "assistant":
        return None
    instruction = (first_msg.get("content") or "").strip()
    response = (second_msg.get("content") or "").strip()
    if not instruction or not response:
        return None
    return instruction, response


def parse_hh_single_turn(transcript: str) -> tuple[str, str] | None:
    """Parse a single-turn Anthropic HH-RLHF transcript into (instruction, response)."""
    if not isinstance(transcript, str):
        return None
    if transcript.count(HUMAN_TAG) != 1 or transcript.count(ASSISTANT_TAG) != 1:
        return None
    h_idx = transcript.find(HUMAN_TAG)
    a_idx = transcript.find(ASSISTANT_TAG)
    if h_idx != 0 or a_idx <= h_idx:
        return None
    instruction = transcript[h_idx + len(HUMAN_TAG) : a_idx].strip()
    response = transcript[a_idx + len(ASSISTANT_TAG) :].strip()
    if not instruction or not response:
        return None
    return instruction, response


def extract_hh_pair(row: dict[str, Any]) -> tuple[str, str, str] | None:
    """Extract single-turn (instruction, chosen_response, rejected_response) from HH-RLHF."""
    chosen_parsed = parse_hh_single_turn(row.get("chosen", ""))
    rejected_parsed = parse_hh_single_turn(row.get("rejected", ""))
    if chosen_parsed is None or rejected_parsed is None:
        return None
    inst_c, resp_c = chosen_parsed
    inst_r, resp_r = rejected_parsed
    if inst_c != inst_r or resp_c == resp_r:
        return None
    return inst_c, resp_c, resp_r


def extract_ultrafeedback_pair(row: dict[str, Any]) -> tuple[str, str, str, float, float] | None:
    """Extract (instruction, chosen_response, rejected_response, score_c, score_r) from UltraFeedback."""
    prompt = (row.get("prompt") or "").strip()
    chosen_msgs = row.get("chosen") or []
    rejected_msgs = row.get("rejected") or []
    if not prompt or len(chosen_msgs) < 2 or len(rejected_msgs) < 2:
        return None
    # Single-turn check: 1 user message + 1 assistant message
    if len(chosen_msgs) != 2 or len(rejected_msgs) != 2:
        return None
    chosen_resp = (chosen_msgs[-1].get("content") or "").strip()
    rejected_resp = (rejected_msgs[-1].get("content") or "").strip()
    if not chosen_resp or not rejected_resp or chosen_resp == rejected_resp:
        return None
    score_c = float(row.get("score_chosen", 0.0))
    score_r = float(row.get("score_rejected", 0.0))
    return prompt, chosen_resp, rejected_resp, score_c, score_r


def _token_lengths(tokenizer, texts: list[str], add_special_tokens: bool = False, batch_size: int = 4096) -> list[int]:
    """Compute token lengths for a list of strings using fast batched tokenization when available."""
    if not texts:
        return []
    if getattr(tokenizer, "is_fast", False):
        lengths: list[int] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(chunk, add_special_tokens=add_special_tokens, truncation=False, padding=False)
            lengths.extend(len(ids) for ids in enc["input_ids"])
        return lengths
    if add_special_tokens:
        return [len(tokenizer.encode(t)) for t in texts]
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def prepare_sft_split(
    ds,
    tokenizer,
    target_count: int,
    max_length: int = 512,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter UltraChat split, prefer examples fitting in max_length with EOS, and sample target_count."""
    total_raw = len(ds)
    valid_indices: list[int] = []
    prompts_list: list[str] = []
    responses_list: list[str] = []

    for idx, row in enumerate(ds):
        parsed = extract_ultrachat_first_turn(row)
        if parsed is None:
            continue
        instruction, response = parsed
        valid_indices.append(idx)
        prompts_list.append(format_prompt(instruction))
        responses_list.append(response)

    valid_non_empty = len(valid_indices)
    eos_len = 1 if getattr(tokenizer, "eos_token_id", None) is not None else 0

    p_lens = _token_lengths(tokenizer, prompts_list, add_special_tokens=True)
    r_lens = _token_lengths(tokenizer, responses_list, add_special_tokens=False)

    fits_512_indices: list[int] = []
    truncated_512_indices: list[int] = []

    for idx, p_len, r_len in zip(valid_indices, p_lens, r_lens):
        if p_len + r_len + eos_len <= max_length:
            fits_512_indices.append(idx)
        else:
            truncated_512_indices.append(idx)

    pct_would_truncate_all = (
        100.0 * len(truncated_512_indices) / max(1, valid_non_empty)
    )

    rng = random.Random(seed)
    pool = list(fits_512_indices)
    rng.shuffle(pool)
    selected_indices = pool[:target_count]
    selected_trunc_count = 0

    if len(selected_indices) < target_count:
        extra_pool = list(truncated_512_indices)
        rng.shuffle(extra_pool)
        needed = target_count - len(selected_indices)
        selected_indices.extend(extra_pool[:needed])
        selected_trunc_count = needed

    subset_ds = ds.select(selected_indices)
    records: list[dict[str, Any]] = []
    for idx, row in zip(selected_indices, subset_ds):
        instruction, response = extract_ultrachat_first_turn(row)
        records.append(
            {
                "raw_index": int(idx),
                "prompt_id": row.get("prompt_id", str(idx)),
                "instruction": instruction,
                "prompt": format_prompt(instruction),
                "response": response,
            }
        )

    stats = {
        "total_raw": total_raw,
        "after_non_empty_single_turn": valid_non_empty,
        "fits_in_512_with_eos": len(fits_512_indices),
        "exceeds_512_in_raw_pool": len(truncated_512_indices),
        "raw_pool_pct_truncated_at_512": round(pct_would_truncate_all, 2),
        "selected_count": len(records),
        "selected_pct_truncated_at_512": round(
            100.0 * selected_trunc_count / max(1, len(records)), 2
        ),
    }
    return records, stats


def prepare_hh_dpo_split(
    ds,
    tokenizer,
    target_count: int,
    max_length: int = 512,
    max_prompt_length: int = 256,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter HH-RLHF split to single-turn, drop prompt > 256 tokens, and sample target_count."""
    total_raw = len(ds)
    valid_indices: list[int] = []
    prompts_list: list[str] = []
    chosen_list: list[str] = []
    rejected_list: list[str] = []

    for idx, row in enumerate(ds):
        parsed = extract_hh_pair(row)
        if parsed is None:
            continue
        instruction, chosen, rejected = parsed
        valid_indices.append(idx)
        prompts_list.append(format_prompt(instruction))
        chosen_list.append(chosen)
        rejected_list.append(rejected)

    after_single_turn = len(valid_indices)
    eos_len = 1 if getattr(tokenizer, "eos_token_id", None) is not None else 0

    p_lens = _token_lengths(tokenizer, prompts_list, add_special_tokens=True)
    c_resp_lens = _token_lengths(tokenizer, chosen_list, add_special_tokens=False)
    r_resp_lens = _token_lengths(tokenizer, rejected_list, add_special_tokens=False)

    dropped_prompt_gt_256 = 0
    kept_candidates: list[tuple[int, int, int]] = []

    for idx, p_len, c_rlen, r_rlen in zip(valid_indices, p_lens, c_resp_lens, r_resp_lens):
        if p_len > max_prompt_length:
            dropped_prompt_gt_256 += 1
            continue
        kept_candidates.append((idx, p_len + c_rlen + eos_len, p_len + r_rlen + eos_len))

    rng = random.Random(seed)
    pool = list(kept_candidates)
    rng.shuffle(pool)
    selected = pool[:target_count]

    trunc_chosen = sum(1 for _, c_len, _ in selected if c_len > max_length)
    trunc_rejected = sum(1 for _, _, r_len in selected if r_len > max_length)
    trunc_either = sum(1 for _, c_len, r_len in selected if c_len > max_length or r_len > max_length)

    selected_indices = [idx for idx, _, _ in selected]
    subset_ds = ds.select(selected_indices)
    records: list[dict[str, Any]] = []
    for idx, row in zip(selected_indices, subset_ds):
        instruction, chosen, rejected = extract_hh_pair(row)
        records.append(
            {
                "raw_index": int(idx),
                "instruction": instruction,
                "prompt": format_prompt(instruction),
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    stats = {
        "total_raw": total_raw,
        "after_single_turn_and_non_empty": after_single_turn,
        "dropped_prompt_gt_256": dropped_prompt_gt_256,
        "after_prompt_le_256": len(kept_candidates),
        "selected_count": len(records),
        "selected_pct_chosen_truncated_at_512": round(100.0 * trunc_chosen / max(1, len(records)), 2),
        "selected_pct_rejected_truncated_at_512": round(100.0 * trunc_rejected / max(1, len(records)), 2),
        "selected_pct_either_truncated_at_512": round(100.0 * trunc_either / max(1, len(records)), 2),
    }
    return records, stats


def prepare_uf_dpo_splits_from_train_prefs(
    train_prefs_ds,
    tokenizer,
    train_count: int = 15000,
    val_count: int = 1000,
    max_length: int = 512,
    max_prompt_length: int = 256,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter UltraFeedback train_prefs: single-turn, score_chosen > score_rejected, prompt <= 256,
    and BOTH chosen and rejected fit in <= 512 tokens with EOS; sample disjoint train + val splits.
    """
    total_raw = len(train_prefs_ds)
    after_single_turn = 0
    dropped_equal_or_worse_score = 0

    score_kept_indices: list[int] = []
    prompts_list: list[str] = []
    chosen_list: list[str] = []
    rejected_list: list[str] = []

    for idx, row in enumerate(train_prefs_ds):
        parsed = extract_ultrafeedback_pair(row)
        if parsed is None:
            continue
        after_single_turn += 1
        instruction, chosen, rejected, score_c, score_r = parsed
        if score_c <= score_r:
            dropped_equal_or_worse_score += 1
            continue
        score_kept_indices.append(idx)
        prompts_list.append(format_prompt(instruction))
        chosen_list.append(chosen)
        rejected_list.append(rejected)

    eos_len = 1 if getattr(tokenizer, "eos_token_id", None) is not None else 0
    p_lens = _token_lengths(tokenizer, prompts_list, add_special_tokens=True)
    c_resp_lens = _token_lengths(tokenizer, chosen_list, add_special_tokens=False)
    r_resp_lens = _token_lengths(tokenizer, rejected_list, add_special_tokens=False)

    dropped_prompt_gt_256 = 0
    after_prompt_le_256 = 0
    dropped_exceeds_512 = 0
    kept_indices: list[int] = []

    for idx, p_len, c_rlen, r_rlen in zip(score_kept_indices, p_lens, c_resp_lens, r_resp_lens):
        if p_len > max_prompt_length:
            dropped_prompt_gt_256 += 1
            continue
        after_prompt_le_256 += 1
        c_len = p_len + c_rlen + eos_len
        r_len = p_len + r_rlen + eos_len
        if c_len > max_length or r_len > max_length:
            dropped_exceeds_512 += 1
            continue
        kept_indices.append(idx)

    total_needed = train_count + val_count
    if len(kept_indices) < total_needed:
        raise ValueError(f"Filtered UltraFeedback pool ({len(kept_indices)}) < {total_needed}")

    rng = random.Random(seed)
    pool = list(kept_indices)
    rng.shuffle(pool)
    train_indices = pool[:train_count]
    val_indices = pool[train_count : train_count + val_count]

    def _materialize(indices: list[int]) -> list[dict[str, Any]]:
        subset = train_prefs_ds.select(indices)
        out: list[dict[str, Any]] = []
        for idx, row in zip(indices, subset):
            instruction, chosen, rejected, score_c, score_r = extract_ultrafeedback_pair(row)
            out.append(
                {
                    "raw_index": int(idx),
                    "prompt_id": row.get("prompt_id", str(idx)),
                    "instruction": instruction,
                    "prompt": format_prompt(instruction),
                    "chosen": chosen,
                    "rejected": rejected,
                    "score_chosen": score_c,
                    "score_rejected": score_r,
                }
            )
        return out

    train_records = _materialize(train_indices)
    val_records = _materialize(val_indices)

    stats = {
        "source_split": "train_prefs (disjoint train + val)",
        "total_raw": total_raw,
        "after_single_turn_and_non_empty": after_single_turn,
        "dropped_equal_or_worse_score": dropped_equal_or_worse_score,
        "after_score_filter": after_single_turn - dropped_equal_or_worse_score,
        "dropped_prompt_gt_256": dropped_prompt_gt_256,
        "after_prompt_le_256": after_prompt_le_256,
        "dropped_either_exceeds_512_with_eos": dropped_exceeds_512,
        "both_fit_in_512_with_eos_pool": len(kept_indices),
        "selected_train_count": len(train_records),
        "selected_val_count": len(val_records),
        "selected_pct_chosen_truncated_at_512": 0.0,
        "selected_pct_rejected_truncated_at_512": 0.0,
        "selected_pct_either_truncated_at_512": 0.0,
    }
    return train_records, val_records, stats


def compute_persona_lora_stats(
    json_path: str | Path = "data/persona_lora/ramsay_style_preferences.json",
    tokenizer=None,
    max_length: int = 512,
    max_prompt_length: int = 256,
) -> dict[str, Any]:
    """Compute token length and truncation statistics on the provided Ramsay persona dataset."""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    total_raw = len(rows)
    non_empty = 0
    prompt_gt_256 = 0
    trunc_chosen = 0
    trunc_rejected = 0
    trunc_either = 0

    eos_len = 1 if (tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None) else 0

    for row in rows:
        p = (row.get("prompt") or "").strip()
        c = (row.get("chosen") or "").strip()
        r = (row.get("rejected") or "").strip()
        if not p or not c or not r or c == r:
            continue
        non_empty += 1
        if tokenizer is not None:
            p_len = len(tokenizer.encode(format_prompt(p)))
            if p_len > max_prompt_length:
                prompt_gt_256 += 1
            c_len = p_len + len(tokenizer.encode(c, add_special_tokens=False)) + eos_len
            r_len = p_len + len(tokenizer.encode(r, add_special_tokens=False)) + eos_len
            if c_len > max_length:
                trunc_chosen += 1
            if r_len > max_length:
                trunc_rejected += 1
            if c_len > max_length or r_len > max_length:
                trunc_either += 1

    return {
        "total_raw": total_raw,
        "after_non_empty": non_empty,
        "prompt_gt_256": prompt_gt_256,
        "pct_chosen_truncated_at_512": round(100.0 * trunc_chosen / max(1, non_empty), 2),
        "pct_rejected_truncated_at_512": round(100.0 * trunc_rejected / max(1, non_empty), 2),
        "pct_either_truncated_at_512": round(100.0 * trunc_either / max(1, non_empty), 2),
    }


def load_sft_from_manifest(manifest_path: str | Path = "data/manifests/sft_ultrachat_manifest.json"):
    """Load exact SFT train and val records using saved manifest indices."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(manifest["train_indices"])
    val_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").select(manifest["val_indices"])

    def _to_records(subset, indices):
        out = []
        for raw_idx, row in zip(indices, subset):
            inst, resp = extract_ultrachat_first_turn(row)
            out.append(
                {
                    "raw_index": int(raw_idx),
                    "prompt_id": row.get("prompt_id", str(raw_idx)),
                    "instruction": inst,
                    "prompt": format_prompt(inst),
                    "response": resp,
                }
            )
        return out

    return _to_records(train_ds, manifest["train_indices"]), _to_records(val_ds, manifest["val_indices"])


def load_hh_dpo_from_manifest(manifest_path: str | Path = "data/manifests/dpo_hh_manifest.json"):
    """Load exact HH-RLHF DPO train and val records using saved manifest indices."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_ds = load_dataset("Anthropic/hh-rlhf", split="train").select(manifest["train_indices"])
    val_ds = load_dataset("Anthropic/hh-rlhf", split="test").select(manifest["val_indices"])

    def _to_records(subset, indices):
        out = []
        for raw_idx, row in zip(indices, subset):
            inst, chosen, rejected = extract_hh_pair(row)
            out.append(
                {
                    "raw_index": int(raw_idx),
                    "instruction": inst,
                    "prompt": format_prompt(inst),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        return out

    return _to_records(train_ds, manifest["train_indices"]), _to_records(val_ds, manifest["val_indices"])


def load_uf_dpo_from_manifest(manifest_path: str | Path = "data/manifests/dpo_uf_manifest.json"):
    """Load exact UltraFeedback DPO train and val records (both sampled disjointly from train_prefs)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    train_ds = base_ds.select(manifest["train_indices"])
    val_ds = base_ds.select(manifest["val_indices"])

    def _to_records(subset, indices):
        out = []
        for raw_idx, row in zip(indices, subset):
            inst, chosen, rejected, score_c, score_r = extract_ultrafeedback_pair(row)
            out.append(
                {
                    "raw_index": int(raw_idx),
                    "prompt_id": row.get("prompt_id", str(raw_idx)),
                    "instruction": inst,
                    "prompt": format_prompt(inst),
                    "chosen": chosen,
                    "rejected": rejected,
                    "score_chosen": score_c,
                    "score_rejected": score_r,
                }
            )
        return out

    return _to_records(train_ds, manifest["train_indices"]), _to_records(val_ds, manifest["val_indices"])
