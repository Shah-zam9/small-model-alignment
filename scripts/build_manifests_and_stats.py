"""Build fixed evaluation subsets, train/val manifests, and print dataset filtering statistics."""

from __future__ import annotations

import json
from pathlib import Path
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from src.align.data import (
    SEED,
    build_alpaca_eval_subset,
    compute_persona_lora_stats,
    prepare_hh_dpo_split,
    prepare_sft_split,
    prepare_uf_dpo_splits_from_train_prefs,
)


def _retry_load(repo_id: str, split: str):
    for attempt in range(5):
        try:
            return load_dataset(repo_id, split=split)
        except Exception as err:
            print(f"[retry {attempt + 1}/5] {repo_id} ({split}): {err}")
            time.sleep(3)
    raise RuntimeError(f"Failed to load {repo_id} ({split})")


def main() -> None:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen/Qwen2.5-0.5B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    print("\n1. Building AlpacaEval 300-prompt fixed evaluation set (seed=42)...")
    ae_subset = build_alpaca_eval_subset("eval_sets/alpaca_eval_300_seed42.json", num_samples=300, seed=SEED)
    print(f"  Saved {len(ae_subset)} prompts to eval_sets/alpaca_eval_300_seed42.json")

    print("\n2. Processing SFT dataset: HuggingFaceH4/ultrachat_200k...")
    uc_train_ds = _retry_load("HuggingFaceH4/ultrachat_200k", "train_sft")
    uc_val_ds = _retry_load("HuggingFaceH4/ultrachat_200k", "test_sft")
    sft_train, sft_train_stats = prepare_sft_split(uc_train_ds, tokenizer, target_count=30000, max_length=512, seed=SEED)
    sft_val, sft_val_stats = prepare_sft_split(uc_val_ds, tokenizer, target_count=1000, max_length=512, seed=SEED)

    with open(manifest_dir / "sft_ultrachat_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "HuggingFaceH4/ultrachat_200k",
                "seed": SEED,
                "max_length": 512,
                "train_stats": sft_train_stats,
                "val_stats": sft_val_stats,
                "train_indices": [r["raw_index"] for r in sft_train],
                "val_indices": [r["raw_index"] for r in sft_val],
            },
            f,
            indent=2,
        )

    print("\n3. Processing HH-RLHF DPO dataset: Anthropic/hh-rlhf...")
    hh_train_ds = _retry_load("Anthropic/hh-rlhf", "train")
    hh_val_ds = _retry_load("Anthropic/hh-rlhf", "test")
    hh_train, hh_train_stats = prepare_hh_dpo_split(
        hh_train_ds, tokenizer, target_count=15000, max_length=512, max_prompt_length=256, seed=SEED
    )
    hh_val, hh_val_stats = prepare_hh_dpo_split(
        hh_val_ds, tokenizer, target_count=1000, max_length=512, max_prompt_length=256, seed=SEED
    )

    with open(manifest_dir / "dpo_hh_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "Anthropic/hh-rlhf",
                "seed": SEED,
                "max_length": 512,
                "max_prompt_length": 256,
                "train_stats": hh_train_stats,
                "val_stats": hh_val_stats,
                "train_indices": [r["raw_index"] for r in hh_train],
                "val_indices": [r["raw_index"] for r in hh_val],
            },
            f,
            indent=2,
        )

    print("\n4. Processing UltraFeedback DPO dataset: HuggingFaceH4/ultrafeedback_binarized...")
    uf_train_ds = _retry_load("HuggingFaceH4/ultrafeedback_binarized", "train_prefs")
    uf_train, uf_val, uf_stats = prepare_uf_dpo_splits_from_train_prefs(
        uf_train_ds,
        tokenizer,
        train_count=15000,
        val_count=1000,
        max_length=512,
        max_prompt_length=256,
        seed=SEED,
    )

    with open(manifest_dir / "dpo_uf_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "HuggingFaceH4/ultrafeedback_binarized",
                "source_split": "train_prefs",
                "seed": SEED,
                "max_length": 512,
                "max_prompt_length": 256,
                "stats": uf_stats,
                "train_indices": [r["raw_index"] for r in uf_train],
                "val_indices": [r["raw_index"] for r in uf_val],
            },
            f,
            indent=2,
        )

    print("\n5. Computing Persona LoRA dataset stats...")
    smol_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    persona_stats = compute_persona_lora_stats(
        "data/persona_lora/ramsay_style_preferences.json", tokenizer=smol_tok, max_length=512, max_prompt_length=256
    )

    all_stats = {
        "sft_ultrachat": {"train": sft_train_stats, "val": sft_val_stats},
        "dpo_hh_rlhf": {"train": hh_train_stats, "val": hh_val_stats},
        "dpo_ultrafeedback": uf_stats,
        "persona_lora": persona_stats,
    }
    with open(manifest_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print("\n=== SUMMARY OF DATASET STATS ===")
    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
