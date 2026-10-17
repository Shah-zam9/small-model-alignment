"""From-scratch PyTorch Direct Preference Optimization (DPO) training loop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.optim import AdamW, RMSprop
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.align.batching import build_dpo_batch
from src.align.data import load_hh_dpo_from_manifest, load_uf_dpo_from_manifest
from src.align.logprobs import compute_response_logprobs_from_batch
from src.align.losses import compute_dpo_loss_from_logps, compute_dpo_metrics_from_logps


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def resolve_model_path(path_str: str) -> str:
    """Resolve local or Kaggle kernel_sources mount paths automatically."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists() and path_str.startswith("/kaggle/input/"):
        parts = p.parts  # ('/', 'kaggle', 'input', '<slug>', ...)
        if len(parts) > 4:
            rel_suffix = Path(*parts[4:])
            matches = list(kaggle_in.rglob(str(rel_suffix / "config.json")))
            if matches:
                resolved = str(matches[0].parent)
                print(f"Resolved Kaggle model path {path_str} -> {resolved}")
                return resolved
    return path_str


@torch.no_grad()
def evaluate_dpo(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    val_loader: DataLoader,
    beta: float,
    use_amp: bool,
    amp_device_type: str = "cuda",
) -> dict[str, Any]:
    """Evaluate DPO loss, reward accuracy, reward margin, and reference logprob finiteness on val_loader."""
    t0 = time.time()
    policy_model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    total_chosen_rew = 0.0
    total_rejected_rew = 0.0
    total_examples = 0
    ref_logps_finite = True

    for dpo_batch in val_loader:
        bsz = dpo_batch["chosen"]["input_ids"].shape[0]
        with torch.amp.autocast(amp_device_type, dtype=torch.float16, enabled=use_amp):
            pi_c = compute_response_logprobs_from_batch(policy_model, dpo_batch["chosen"])
            pi_r = compute_response_logprobs_from_batch(policy_model, dpo_batch["rejected"])
            ref_c = compute_response_logprobs_from_batch(reference_model, dpo_batch["chosen"])
            ref_r = compute_response_logprobs_from_batch(reference_model, dpo_batch["rejected"])

        if not (torch.isfinite(ref_c).all() and torch.isfinite(ref_r).all()):
            ref_logps_finite = False

        loss = compute_dpo_loss_from_logps(pi_c, pi_r, ref_c, ref_r, beta=beta)
        metrics = compute_dpo_metrics_from_logps(pi_c, pi_r, ref_c, ref_r, beta=beta)

        total_loss += float(loss.item()) * bsz
        total_acc += float(metrics["reward_accuracy"].item()) * bsz
        total_margin += float(metrics["reward_margin"].item()) * bsz
        total_chosen_rew += float(metrics["chosen_rewards"].mean().item()) * bsz
        total_rejected_rew += float(metrics["rejected_rewards"].mean().item()) * bsz
        total_examples += bsz

    policy_model.train()
    denom = max(1, total_examples)
    return {
        "val_loss": total_loss / denom,
        "val_reward_accuracy": total_acc / denom,
        "val_reward_margin": total_margin / denom,
        "val_chosen_reward": total_chosen_rew / denom,
        "val_rejected_reward": total_rejected_rew / denom,
        "ref_logps_finite": ref_logps_finite,
        "val_pass_sec": time.time() - t0,
    }


def train_dpo(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    policy_model_path = resolve_model_path(config["policy_model_path"])
    reference_model_path = resolve_model_path(config.get("reference_model_path", policy_model_path))
    track = config.get("track", "hh_rlhf")
    manifest_path = config["manifest_path"]
    output_dir = Path(config.get("output_dir", f"outputs/dpo_{track}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    beta = float(config.get("beta", 0.1))
    max_length = int(config.get("max_length", 512))
    per_device_batch_size = int(config.get("per_device_batch_size", 1))
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 32))
    lr = float(config.get("learning_rate", 1e-6))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_name = config.get("optimizer", "adamw").lower()
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    epochs = int(config.get("epochs", 1))
    eval_every_steps = int(config.get("eval_every_steps", 100))
    max_steps = config.get("max_steps", None)
    train_limit = config.get("train_limit", None)
    val_limit = config.get("val_limit", None)
    save_ckpts = bool(config.get("save_checkpoints", True))
    force_cpu = bool(config.get("force_cpu", False))

    has_cuda = torch.cuda.is_available() and not force_cpu
    amp_device_type = "cuda" if has_cuda else "cpu"
    num_gpus = torch.cuda.device_count() if has_cuda else 0
    policy_device = torch.device("cuda:0" if has_cuda else "cpu")
    ref_device = torch.device("cuda:1" if num_gpus > 1 else policy_device)
    use_amp = has_cuda and bool(config.get("fp16", True))

    print(f"Loading tokenizer from {policy_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading trainable policy model in fp32 on {policy_device}: {policy_model_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_path, torch_dtype=torch.float32
    ).to(policy_device)
    if config.get("gradient_checkpointing", False):
        policy_model.gradient_checkpointing_enable()

    ref_dtype_cfg = str(config.get("reference_dtype", "fp16")).lower()
    ref_dtype = (
        torch.float16
        if (has_cuda and ref_dtype_cfg == "fp16")
        else torch.float32
    )
    print(f"Loading frozen reference model in {ref_dtype} on {ref_device}: {reference_model_path}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path, torch_dtype=ref_dtype
    ).to(ref_device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    if track == "ultrafeedback":
        train_records, val_records = load_uf_dpo_from_manifest(manifest_path)
    else:
        train_records, val_records = load_hh_dpo_from_manifest(manifest_path)

    if train_limit is not None:
        train_records = train_records[: int(train_limit)]
    if val_limit is not None:
        val_records = val_records[: int(val_limit)]

    def collate_fn(rows: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        prompts = [r["prompt"] for r in rows]
        chosen = [r["chosen"] for r in rows]
        rejected = [r["rejected"] for r in rows]
        return build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length=max_length)

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

    if optimizer_name == "rmsprop":
        optimizer = RMSprop(policy_model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if has_cuda:
        for dev_idx in range(num_gpus):
            torch.cuda.reset_peak_memory_stats(dev_idx)

    start_time = time.time()
    best_val_acc = -1.0
    global_step = 0
    skipped_steps = 0
    all_losses_finite = True
    ref_logps_all_finite = True
    val_pass_times: list[float] = []

    def get_gpu_mem() -> tuple[float, float]:
        if not has_cuda:
            return 0.0, 0.0
        m0 = torch.cuda.max_memory_allocated(0) / (1024**3)
        m1 = torch.cuda.max_memory_allocated(1) / (1024**3) if num_gpus > 1 else 0.0
        return round(m0, 3), round(m1, 3)

    def log_metric(record: dict[str, Any]) -> None:
        with open(metrics_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(record) + "\n")
        print("METRIC:", json.dumps(record))

    # Step 0 baseline validation
    eval0 = evaluate_dpo(
        policy_model,
        reference_model,
        val_loader,
        beta=beta,
        use_amp=use_amp,
        amp_device_type=amp_device_type,
    )
    last_eval = eval0
    val_pass_times.append(eval0["val_pass_sec"])
    if not eval0["ref_logps_finite"]:
        ref_logps_all_finite = False
    if not math.isfinite(eval0["val_loss"]):
        all_losses_finite = False
    m0, m1 = get_gpu_mem()
    log_metric(
        {
            "step": 0,
            "train_loss": None,
            "val_loss": round(eval0["val_loss"], 6),
            "val_reward_accuracy": round(eval0["val_reward_accuracy"], 6),
            "val_reward_margin": round(eval0["val_reward_margin"], 6),
            "val_chosen_reward": round(eval0["val_chosen_reward"], 6),
            "val_rejected_reward": round(eval0["val_rejected_reward"], 6),
            "val_pass_sec": round(eval0["val_pass_sec"], 3),
            "ref_logps_all_finite": ref_logps_all_finite,
            "all_losses_finite": all_losses_finite,
            "scaler_scale": scaler.get_scale() if use_amp else 1.0,
            "skipped_steps": skipped_steps,
            "peak_gpu0_mem_gb": m0,
            "peak_gpu1_mem_gb": m1,
            "lr": lr,
            "elapsed_sec": round(time.time() - start_time, 2),
        }
    )

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_acc = 0.0
    running_margin = 0.0
    micro_steps = 0
    step_start_time = time.time()
    step_times: list[float] = []

    for epoch in range(epochs):
        for batch_idx, dpo_batch in enumerate(train_loader):
            with torch.no_grad(), torch.amp.autocast(amp_device_type, dtype=torch.float16, enabled=use_amp):
                ref_c = compute_response_logprobs_from_batch(reference_model, dpo_batch["chosen"])
                ref_r = compute_response_logprobs_from_batch(reference_model, dpo_batch["rejected"])
                if not (torch.isfinite(ref_c).all() and torch.isfinite(ref_r).all()):
                    ref_logps_all_finite = False

            with torch.amp.autocast(amp_device_type, dtype=torch.float16, enabled=use_amp):
                pi_c = compute_response_logprobs_from_batch(policy_model, dpo_batch["chosen"])
                pi_r = compute_response_logprobs_from_batch(policy_model, dpo_batch["rejected"])
                loss = compute_dpo_loss_from_logps(pi_c, pi_r, ref_c, ref_r, beta=beta)
                scaled_loss = loss / grad_accum_steps

            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                all_losses_finite = False

            scaler.scale(scaled_loss).backward()

            with torch.no_grad():
                m_step = compute_dpo_metrics_from_logps(pi_c, pi_r, ref_c, ref_r, beta=beta)
                running_acc += float(m_step["reward_accuracy"].item())
                running_margin += float(m_step["reward_margin"].item())

            running_loss += loss_val
            micro_steps += 1

            is_last_batch = (batch_idx + 1) == len(train_loader)
            if micro_steps == grad_accum_steps or is_last_batch:
                scale_before = scaler.get_scale() if use_amp else 1.0
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm).item())
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale() if use_amp else 1.0

                if (use_amp and scale_after < scale_before) or (not math.isfinite(grad_norm)):
                    skipped_steps += 1

                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                step_dt = time.time() - step_start_time
                step_times.append(step_dt)
                avg_train_loss = running_loss / max(1, micro_steps)
                avg_train_acc = running_acc / max(1, micro_steps)
                avg_train_margin = running_margin / max(1, micro_steps)
                if not math.isfinite(avg_train_loss):
                    all_losses_finite = False
                running_loss = 0.0
                running_acc = 0.0
                running_margin = 0.0
                micro_steps = 0

                should_eval = (
                    (global_step % eval_every_steps == 0)
                    or (global_step == total_opt_steps)
                )
                eval_res = None
                if should_eval:
                    eval_res = evaluate_dpo(
                        policy_model,
                        reference_model,
                        val_loader,
                        beta=beta,
                        use_amp=use_amp,
                        amp_device_type=amp_device_type,
                    )
                    last_eval = eval_res
                    val_pass_times.append(eval_res["val_pass_sec"])
                    if not eval_res["ref_logps_finite"]:
                        ref_logps_all_finite = False
                    if not math.isfinite(eval_res["val_loss"]):
                        all_losses_finite = False
                    if eval_res["val_reward_accuracy"] > best_val_acc:
                        best_val_acc = eval_res["val_reward_accuracy"]
                        if save_ckpts:
                            save_checkpoint(policy_model, tokenizer, output_dir / "best")

                m0, m1 = get_gpu_mem()
                log_metric(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": round(avg_train_loss, 6),
                        "train_reward_accuracy": round(avg_train_acc, 6),
                        "train_reward_margin": round(avg_train_margin, 6),
                        "val_loss": round(eval_res["val_loss"], 6) if eval_res else None,
                        "val_reward_accuracy": round(eval_res["val_reward_accuracy"], 6) if eval_res else None,
                        "val_reward_margin": round(eval_res["val_reward_margin"], 6) if eval_res else None,
                        "val_chosen_reward": round(eval_res["val_chosen_reward"], 6) if eval_res else None,
                        "val_rejected_reward": round(eval_res["val_rejected_reward"], 6) if eval_res else None,
                        "best_val_reward_accuracy": round(best_val_acc, 6) if best_val_acc >= 0 else None,
                        "val_pass_sec": round(eval_res["val_pass_sec"], 3) if eval_res else None,
                        "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else "inf/nan",
                        "scaler_scale": scale_after,
                        "skipped_steps": skipped_steps,
                        "ref_logps_all_finite": ref_logps_all_finite,
                        "all_losses_finite": all_losses_finite,
                        "lr": lr,
                        "sec_per_step": round(step_dt, 3),
                        "peak_gpu0_mem_gb": m0,
                        "peak_gpu1_mem_gb": m1,
                        "elapsed_sec": round(time.time() - start_time, 2),
                    }
                )

                step_start_time = time.time()

                if global_step >= total_opt_steps:
                    break
        if global_step >= total_opt_steps:
            break

    if save_ckpts:
        final_ckpt_mb = save_checkpoint(policy_model, tokenizer, output_dir / "final")
        if not (output_dir / "best").exists():
            best_ckpt_mb = save_checkpoint(policy_model, tokenizer, output_dir / "best")
        else:
            best_ckpt_mb = get_dir_size_mb(output_dir / "best")
    else:
        final_ckpt_mb = 0.0
        best_ckpt_mb = 0.0

    steady_times = step_times[2:] if len(step_times) > 2 else step_times
    sec_per_step_steady = sum(steady_times) / max(1, len(steady_times))
    sec_per_step_all = sum(step_times) / max(1, len(step_times))
    mean_val_pass_sec = sum(val_pass_times) / max(1, len(val_pass_times))

    # Full DPO run has 469 optimizer steps (ceil(15000 / 32)) and 6 validation passes on 1,000 pairs
    # (steps 0, 100, 200, 300, 400, 469)
    val_scale_to_1000 = 1000.0 / float(max(1, len(val_records)))
    projected_1000_val_pass_sec = mean_val_pass_sec * val_scale_to_1000
    full_train_sec = 469.0 * sec_per_step_steady
    full_eval_sec = 6.0 * projected_1000_val_pass_sec
    projected_full_run_hours = (full_train_sec + full_eval_sec) / 3600.0

    m0, m1 = get_gpu_mem()
    summary = {
        "track": track,
        "total_steps_run": global_step,
        "train_pairs_used": len(train_records),
        "val_pairs_used": len(val_records),
        "step_0_val_loss": round(eval0["val_loss"], 6),
        "step_0_val_reward_accuracy": round(eval0["val_reward_accuracy"], 6),
        "step_0_val_reward_margin": round(eval0["val_reward_margin"], 6),
        "final_val_loss": round(last_eval["val_loss"], 6),
        "final_val_reward_accuracy": round(last_eval["val_reward_accuracy"], 6),
        "final_val_reward_margin": round(last_eval["val_reward_margin"], 6),
        "best_val_reward_accuracy": round(best_val_acc, 6),
        "ref_logps_all_finite": ref_logps_all_finite,
        "all_losses_finite": all_losses_finite,
        "skipped_steps": skipped_steps,
        "final_scaler_scale": scaler.get_scale() if use_amp else 1.0,
        "sec_per_step_excluding_first_2": round(sec_per_step_steady, 3),
        "sec_per_step_all": round(sec_per_step_all, 3),
        "val_pass_sec_for_n_val": round(mean_val_pass_sec, 3),
        "projected_1000_val_pass_sec": round(projected_1000_val_pass_sec, 3),
        "projected_full_run_hours_with_evals": round(projected_full_run_hours, 3),
        "peak_gpu0_mem_gb": m0,
        "peak_gpu1_mem_gb": m1,
        "best_checkpoint_size_mb": best_ckpt_mb,
        "final_checkpoint_size_mb": final_ckpt_mb,
        "elapsed_sec": round(time.time() - start_time, 2),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("DPO SUMMARY:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPO model with custom PyTorch loop.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--policy_model_path", type=str, default=None, help="Override policy_model_path.")
    parser.add_argument("--reference_model_path", type=str, default=None, help="Override reference_model_path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.policy_model_path:
        config["policy_model_path"] = args.policy_model_path
    if args.reference_model_path:
        config["reference_model_path"] = args.reference_model_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.cpu:
        config["force_cpu"] = True
    train_dpo(config)


if __name__ == "__main__":
    main()
