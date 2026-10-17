"""Task 5: TRL DPOTrainer + PEFT/LoRA Persona Alignment on SmolLM2-360M-Instruct."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random
import time
from typing import Any

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_conversational_row(row: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    """Convert raw {prompt, chosen, rejected} into TRL conversational DPO role-message lists."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"].strip()}],
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
    }


def load_persona_splits(
    data_path: str | Path,
    val_size: int = 20,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load ramsay_style_preferences.json, format as conversational records, and split train/val."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    formatted = [format_conversational_row(r) for r in raw_rows]
    rng = random.Random(seed)
    indices = list(range(len(formatted)))
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_rows = [formatted[i] for i in train_indices]
    val_rows = [formatted[i] for i in val_indices]
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def train_lora_dpo(config: dict[str, Any]) -> dict[str, Any]:
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_name = config.get("model_name_or_path", "HuggingFaceTB/SmolLM2-360M-Instruct")
    data_path = config.get("data_path", "data/persona_lora/ramsay_style_preferences.json")
    output_dir = Path(config.get("output_dir", "/kaggle/working/outputs/lora_persona"))
    output_dir.mkdir(parents=True, exist_ok=True)

    val_size = int(config.get("val_size", 20))
    beta = float(config.get("beta", 0.1))
    lr = float(config.get("learning_rate", 5.0e-5))
    epochs = float(config.get("epochs", 3))
    per_device_bs = int(config.get("per_device_batch_size", 2))
    grad_accum = int(config.get("gradient_accumulation_steps", 4))
    max_length = int(config.get("max_length", 512))
    max_prompt_length = int(config.get("max_prompt_length", 256))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    weight_decay = float(config.get("weight_decay", 0.01))
    eval_steps = int(config.get("eval_steps", 15))
    logging_steps = int(config.get("logging_steps", 5))

    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    lora_dropout = float(config.get("lora_dropout", 0.05))
    target_modules = list(
        config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    )

    train_ds, val_ds = load_persona_splits(data_path=data_path, val_size=val_size, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": 0} if has_cuda else None,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_cfg_sig = inspect.signature(DPOConfig.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in dpo_cfg_sig else "evaluation_strategy"
    dpo_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "trl_checkpoints"),
        "beta": beta,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_bs,
        "per_device_eval_batch_size": per_device_bs * 2,
        "gradient_accumulation_steps": grad_accum,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,
        "fp16": has_cuda,
        "logging_steps": logging_steps,
        eval_key: "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_rewards/accuracies",
        "greater_is_better": True,
        "seed": seed,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    filtered_dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in dpo_cfg_sig}
    training_args = DPOConfig(**filtered_dpo_kwargs)

    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    tok_key = "processing_class" if "processing_class" in trainer_sig else "tokenizer"
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        tok_key: tokenizer,
        "peft_config": peft_config,
    }
    trainer = DPOTrainer(**trainer_kwargs)

    # Count trainable vs total parameters
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_pct = round(100.0 * trainable_params / max(1, total_params), 4)
    print(f"LoRA trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct}%)")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats(0)

    start_time = time.time()
    eval0 = trainer.evaluate()
    print("Step-0 eval:", eval0)

    train_result = trainer.train()
    final_eval = trainer.evaluate()
    elapsed = time.time() - start_time

    # Save LoRA adapter
    adapter_dir = output_dir / "best_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    # Also save merged model for direct AutoModelForCausalLM loading in generation
    merged_dir = output_dir / "best_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        for entry in trainer.state.log_history:
            mf.write(json.dumps(entry) + "\n")

    peak_mem_gb = (
        round(torch.cuda.max_memory_allocated(0) / (1024**3), 3) if has_cuda else 0.0
    )

    summary = {
        "model_name": model_name,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
        "beta": beta,
        "learning_rate": lr,
        "epochs": epochs,
        "effective_batch_size": per_device_bs * grad_accum,
        "step_0_eval": eval0,
        "final_eval": final_eval,
        "train_runtime_sec": round(elapsed, 2),
        "peak_gpu0_mem_gb": peak_mem_gb,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    print("LORA DPO SUMMARY:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA Persona DPO model with TRL DPOTrainer.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    train_lora_dpo(config)


if __name__ == "__main__":
    main()
