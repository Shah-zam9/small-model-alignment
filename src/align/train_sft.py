"""From-scratch PyTorch Supervised Fine-Tuning (SFT) training loop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.align.batching import build_sft_batch
from src.align.data import load_sft_from_manifest
from src.align.losses import compute_sft_loss_from_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup followed by cosine decay to 0."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_sft(
    model: torch.nn.Module,
    val_loader: DataLoader,
    use_amp: bool,
) -> tuple[float, float]:
    """Compute mean token-level SFT loss across the validation dataloader and return (val_loss, elapsed_sec)."""
    t0 = time.time()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = compute_sft_loss_from_batch(model, batch)
        total_loss += float(loss.item())
        num_batches += 1
    model.train()
    return total_loss / max(1, num_batches), time.time() - t0


def get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def save_checkpoint(model, tokenizer, save_dir: Path) -> float:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    return get_dir_size_mb(save_dir)


def train_sft(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_name = config.get("model_name", "Qwen/Qwen2.5-0.5B")
    manifest_path = config.get("manifest_path", "data/manifests/sft_ultrachat_manifest.json")
    output_dir = Path(config.get("output_dir", "outputs/sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    max_length = int(config.get("max_length", 512))
    per_device_batch_size = int(config.get("per_device_batch_size", 2))
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 16))
    lr = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.1))
    warmup_ratio = float(config.get("warmup_ratio", 0.03))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    epochs = int(config.get("epochs", 1))
    eval_every_steps = int(config.get("eval_every_steps", 100))
    max_steps = config.get("max_steps", None)
    train_limit = config.get("train_limit", None)
    val_limit = config.get("val_limit", None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(config.get("fp16", True))

    print(f"Loading tokenizer and model: {model_name} on {device} (fp32 weights, fp16 autocast={use_amp})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    train_records, val_records = load_sft_from_manifest(manifest_path)
    if train_limit is not None:
        train_records = train_records[: int(train_limit)]
    if val_limit is not None:
        val_records = val_records[: int(val_limit)]

    def collate_fn(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [r["prompt"] for r in rows]
        responses = [r["response"] for r in rows]
        return build_sft_batch(tokenizer, prompts, responses, max_length=max_length)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_records,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_records,
        batch_size=per_device_batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_opt_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_opt_steps = min(total_opt_steps, int(max_steps))
    warmup_steps = max(1, int(round(total_opt_steps * warmup_ratio)))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    best_val_loss = float("inf")
    global_step = 0
    skipped_steps = 0
    all_losses_finite = True
    val_pass_times: list[float] = []

    def log_metric(record: dict[str, Any]) -> None:
        with open(metrics_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(record) + "\n")
        print("METRIC:", json.dumps(record))

    # Step 0 baseline validation
    init_val_loss, init_val_sec = evaluate_sft(model, val_loader, use_amp=use_amp)
    val_pass_times.append(init_val_sec)
    if not math.isfinite(init_val_loss):
        all_losses_finite = False
    best_val_loss = init_val_loss
    log_metric(
        {
            "step": 0,
            "train_loss": None,
            "val_loss": round(init_val_loss, 6),
            "val_pass_sec": round(init_val_sec, 3),
            "scaler_scale": scaler.get_scale() if use_amp else 1.0,
            "skipped_steps": skipped_steps,
            "all_losses_finite": all_losses_finite,
            "lr": lr,
            "elapsed_sec": round(time.time() - start_time, 2),
        }
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    micro_steps = 0
    step_start_time = time.time()
    step_times: list[float] = []

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = compute_sft_loss_from_batch(model, batch)
                scaled_loss = loss / grad_accum_steps

            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                all_losses_finite = False

            scaler.scale(scaled_loss).backward()
            running_loss += loss_val
            micro_steps += 1

            is_last_batch = (batch_idx + 1) == len(train_loader)
            if micro_steps == grad_accum_steps or is_last_batch:
                scale_before = scaler.get_scale() if use_amp else 1.0
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale() if use_amp else 1.0

                step_was_skipped = (use_amp and scale_after < scale_before) or (not math.isfinite(grad_norm))
                if step_was_skipped:
                    skipped_steps += 1
                else:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                step_dt = time.time() - step_start_time
                step_times.append(step_dt)
                avg_train_loss = running_loss / max(1, micro_steps)
                if not math.isfinite(avg_train_loss):
                    all_losses_finite = False
                running_loss = 0.0
                micro_steps = 0

                peak_mem_gb = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                )

                should_eval = (
                    (global_step % eval_every_steps == 0)
                    or (global_step == total_opt_steps)
                )
                val_loss = None
                val_sec = None
                if should_eval:
                    val_loss, val_sec = evaluate_sft(model, val_loader, use_amp=use_amp)
                    val_pass_times.append(val_sec)
                    if not math.isfinite(val_loss):
                        all_losses_finite = False
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, tokenizer, output_dir / "best")

                log_metric(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": round(avg_train_loss, 6),
                        "val_loss": round(val_loss, 6) if val_loss is not None else None,
                        "best_val_loss": round(best_val_loss, 6),
                        "val_pass_sec": round(val_sec, 3) if val_sec is not None else None,
                        "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else "inf/nan",
                        "scaler_scale": scale_after,
                        "skipped_steps": skipped_steps,
                        "all_losses_finite": all_losses_finite,
                        "lr": scheduler.get_last_lr()[0],
                        "sec_per_step": round(step_dt, 3),
                        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
                        "elapsed_sec": round(time.time() - start_time, 2),
                    }
                )

                # Reset step timer AFTER validation/checkpointing so step_dt measures pure train step time
                step_start_time = time.time()

                if global_step >= total_opt_steps:
                    break
        if global_step >= total_opt_steps:
            break

    final_ckpt_mb = save_checkpoint(model, tokenizer, output_dir / "final")
    if not (output_dir / "best").exists():
        best_ckpt_mb = save_checkpoint(model, tokenizer, output_dir / "best")
    else:
        best_ckpt_mb = get_dir_size_mb(output_dir / "best")

    steady_times = step_times[2:] if len(step_times) > 2 else step_times
    sec_per_step_steady = sum(steady_times) / max(1, len(steady_times))
    sec_per_step_all = sum(step_times) / max(1, len(step_times))
    mean_val_pass_sec = sum(val_pass_times) / max(1, len(val_pass_times))

    # Full run has 938 optimizer steps (ceil(30000 / 32)) and 11 validation passes on 1,000 examples
    # (steps 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 938)
    val_scale_to_1000 = 1000.0 / float(max(1, len(val_records)))
    projected_1000_val_pass_sec = mean_val_pass_sec * val_scale_to_1000
    full_train_sec = 938.0 * sec_per_step_steady
    full_eval_sec = 11.0 * projected_1000_val_pass_sec
    projected_full_run_hours = (full_train_sec + full_eval_sec) / 3600.0

    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    summary = {
        "total_steps_run": global_step,
        "train_examples_used": len(train_records),
        "val_examples_used": len(val_records),
        "best_val_loss": round(best_val_loss, 6),
        "all_losses_finite": all_losses_finite,
        "skipped_steps": skipped_steps,
        "final_scaler_scale": scaler.get_scale() if use_amp else 1.0,
        "sec_per_step_excluding_first_2": round(sec_per_step_steady, 3),
        "sec_per_step_all": round(sec_per_step_all, 3),
        "val_pass_sec_for_n_val": round(mean_val_pass_sec, 3),
        "projected_1000_val_pass_sec": round(projected_1000_val_pass_sec, 3),
        "projected_full_run_hours_with_evals": round(projected_full_run_hours, 3),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "best_checkpoint_size_mb": best_ckpt_mb,
        "final_checkpoint_size_mb": final_ckpt_mb,
        "elapsed_sec": round(time.time() - start_time, 2),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("SFT SUMMARY:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SFT model with custom PyTorch loop.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_sft(config)


if __name__ == "__main__":
    main()
