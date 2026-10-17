"""Task 5: TRL DPOTrainer + PEFT/LoRA Persona Alignment on SmolLM2-360M-Instruct."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import random
import time
from typing import Any

# Restrict single-process TRL DPOTrainer to 1 GPU on multi-GPU nodes (e.g. Kaggle 2xT4)
if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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


def _eval_rank_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    acc = float(metrics.get("eval_rewards/accuracies", -1.0))
    margin = float(metrics.get("eval_rewards/margins", -1e9))
    loss = float(metrics.get("eval_loss", 1e9))
    return (acc, margin, -loss)


class BestLoRACheckpointCallback(TrainerCallback):
    """Track all evaluation passes during training and immediately save the best LoRA adapter."""

    def __init__(self, best_adapter_dir: Path, tokenizer: Any) -> None:
        self.best_adapter_dir = best_adapter_dir
        self.tokenizer = tokenizer
        self.best_key: tuple[float, float, float] = (-1.0, -1e9, -1e9)
        self.best_eval: dict[str, Any] | None = None
        self.eval_history: list[dict[str, Any]] = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or state.global_step <= 0:
            return
        record = dict(metrics)
        record["step"] = int(state.global_step)
        self.eval_history.append(record)
        key = _eval_rank_key(record)
        if key > self.best_key:
            self.best_key = key
            self.best_eval = record
            if model is not None:
                self.best_adapter_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(self.best_adapter_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(self.best_adapter_dir)


def train_lora_dpo(config: dict[str, Any]) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel
    import transformers
    import trl
    from trl import DPOConfig, DPOTrainer
    import peft

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
    max_steps = int(config.get("max_steps", -1))
    train_limit = config.get("train_limit", None)
    val_limit = config.get("val_limit", None)
    force_cpu = bool(config.get("force_cpu", False))
    save_merged = bool(config.get("save_merged", True))

    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    lora_dropout = float(config.get("lora_dropout", 0.05))
    target_modules = list(
        config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    )

    print(
        f"Versions -> torch={torch.__version__}, transformers={transformers.__version__}, "
        f"trl={trl.__version__}, peft={peft.__version__}"
    )

    train_ds, val_ds = load_persona_splits(data_path=data_path, val_size=val_size, seed=seed)
    if train_limit is not None:
        train_ds = train_ds.select(range(min(len(train_ds), int(train_limit))))
    if val_limit is not None:
        val_ds = val_ds.select(range(min(len(val_ds), int(val_limit))))
    print(f"Dataset split -> train_pairs={len(train_ds)}, val_pairs={len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print one fully chat-templated training example so the exact format TRL receives is visible in the log
    sample0 = train_ds[0]
    templated_prompt = tokenizer.apply_chat_template(
        sample0["prompt"], tokenize=False, add_generation_prompt=True
    )
    templated_chosen = tokenizer.apply_chat_template(
        sample0["prompt"] + sample0["chosen"], tokenize=False
    )
    templated_rejected = tokenizer.apply_chat_template(
        sample0["prompt"] + sample0["rejected"], tokenize=False
    )
    print("=== SAMPLE 0 CONVERSATIONAL RECORD (TRL INPUT) ===")
    print(json.dumps(sample0, indent=2))
    print("=== SAMPLE 0 CHAT-TEMPLATED PROMPT ===")
    print(templated_prompt)
    print("=== SAMPLE 0 CHAT-TEMPLATED CHOSEN (PROMPT + CHOSEN) ===")
    print(templated_chosen)
    print("=== SAMPLE 0 CHAT-TEMPLATED REJECTED (PROMPT + REJECTED) ===")
    print(templated_rejected)
    print("============================================================")

    has_cuda = torch.cuda.is_available() and not force_cpu
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
        "max_steps": max_steps,
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
        "use_cpu": not has_cuda,
        "logging_steps": logging_steps,
        eval_key: "steps",
        "eval_steps": eval_steps,
        "save_strategy": "no",
        "load_best_model_at_end": False,
        "seed": seed,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    filtered_dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in dpo_cfg_sig}
    training_args = DPOConfig(**filtered_dpo_kwargs)

    best_adapter_dir = output_dir / "best_adapter"
    final_adapter_dir = output_dir / "final_adapter"
    best_cb = BestLoRACheckpointCallback(best_adapter_dir=best_adapter_dir, tokenizer=tokenizer)

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
        "callbacks": [best_cb],
    }
    trainer = DPOTrainer(**trainer_kwargs)

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_pct = round(100.0 * trainable_params / max(1, total_params), 4)
    print(f"LoRA trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct}%)")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats(0)

    start_time = time.time()
    eval0 = dict(trainer.evaluate())
    eval0["step"] = 0
    print("STEP-0 EVAL:", json.dumps(eval0))

    trainer.train()

    final_eval = dict(trainer.evaluate())
    final_eval["step"] = int(trainer.state.global_step)
    print("FINAL EVAL:", json.dumps(final_eval))

    # Save final adapter
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter_dir)

    # Compare final_eval against best intermediate eval
    if best_cb.best_eval is None or _eval_rank_key(final_eval) >= best_cb.best_key:
        best_eval = dict(final_eval)
        best_adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(best_adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(best_adapter_dir)
    else:
        best_eval = dict(best_cb.best_eval)

    print("BEST EVAL:", json.dumps(best_eval))
    elapsed = time.time() - start_time

    merged_dir = output_dir / "best_merged"
    if save_merged:
        merged_dir.mkdir(parents=True, exist_ok=True)
        base_for_merge = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": 0} if has_cuda else None,
        )
        peft_for_merge = PeftModel.from_pretrained(base_for_merge, best_adapter_dir)
        merged_model = peft_for_merge.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write(json.dumps({"tag": "step_0_eval", **eval0}) + "\n")
        for entry in trainer.state.log_history:
            mf.write(json.dumps(entry) + "\n")
        mf.write(json.dumps({"tag": "best_eval", **best_eval}) + "\n")
        mf.write(json.dumps({"tag": "final_eval", **final_eval}) + "\n")

    peak_mem_gb = (
        round(torch.cuda.max_memory_allocated(0) / (1024**3), 3) if has_cuda else 0.0
    )

    summary = {
        "model_name": model_name,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "total_steps_run": int(trainer.state.global_step),
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
        "best_eval": best_eval,
        "final_eval": final_eval,
        "train_runtime_sec": round(elapsed, 2),
        "peak_gpu0_mem_gb": peak_mem_gb,
        "best_adapter_dir": str(best_adapter_dir),
        "final_adapter_dir": str(final_adapter_dir),
        "merged_dir": str(merged_dir) if save_merged else None,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    print("LORA DPO SUMMARY:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA Persona DPO model with TRL DPOTrainer.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.cpu:
        config["force_cpu"] = True
    train_lora_dpo(config)


if __name__ == "__main__":
    main()
