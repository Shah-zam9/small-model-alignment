"""Generate training plots, configuration/results tables, and qualitative examples for the final report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ema_smooth(values: list[float], alpha: float = 0.15) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1.0 - alpha) * smoothed[-1])
    return smoothed


def plot_sft(metrics_path: Path, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    rows = load_jsonl(metrics_path)
    if not rows:
        print(f"[skip] No SFT metrics found at {metrics_path}")
        return None

    train_steps = [r["step"] for r in rows if r.get("train_loss") is not None]
    train_losses = [r["train_loss"] for r in rows if r.get("train_loss") is not None]
    val_steps = [r["step"] for r in rows if r.get("val_loss") is not None]
    val_losses = [r["val_loss"] for r in rows if r.get("val_loss") is not None]

    best_step = None
    if val_losses:
        min_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
        best_step = val_steps[min_idx]
        best_val = val_losses[min_idx]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    if train_losses:
        ax.plot(
            train_steps,
            train_losses,
            color="#90caf9",
            alpha=0.45,
            linewidth=1.0,
            label="Train Loss (raw per-step)",
        )
        ax.plot(
            train_steps,
            ema_smooth(train_losses, alpha=0.15),
            color="#1565c0",
            linewidth=2.0,
            label="Train Loss (EMA α=0.15)",
        )
    if val_losses:
        ax.plot(
            val_steps,
            val_losses,
            color="#d32f2f",
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="Validation Loss (1,000 UltraChat val)",
        )
    if best_step is not None:
        ax.axvline(
            best_step,
            color="#2e7d32",
            linestyle="--",
            linewidth=1.5,
            label=f"Selected Best Checkpoint (step {best_step}, val={best_val:.4f})",
        )

    ax.set_title("Supervised Fine-Tuning (SFT) on Qwen2.5-0.5B (UltraChat 30k)")
    ax.set_xlabel("Optimizer Step (effective batch size = 32)")
    ax.set_ylabel("Token-Level Cross-Entropy Loss (NLL)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()

    out_path = out_dir / "sft_loss.png"
    fig.savefig(out_path)
    fig.savefig(out_dir / "sft_loss.pdf")
    plt.close(fig)
    print(f"Saved SFT plot to {out_path}")
    return out_path


def plot_dpo_run(metrics_path: Path, run_label: str, filename_stem: str, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    rows = load_jsonl(metrics_path)
    if not rows:
        print(f"[skip] No DPO metrics found at {metrics_path}")
        return None

    train_steps = [r["step"] for r in rows if r.get("train_loss") is not None]
    train_losses = [r["train_loss"] for r in rows if r.get("train_loss") is not None]
    train_accs = [r["train_reward_accuracy"] for r in rows if r.get("train_reward_accuracy") is not None]
    train_margins = [r["train_reward_margin"] for r in rows if r.get("train_reward_margin") is not None]

    val_rows = [r for r in rows if r.get("val_loss") is not None]
    val_steps = [r["step"] for r in val_rows]
    val_losses = [r["val_loss"] for r in val_rows]
    val_accs = [r["val_reward_accuracy"] for r in val_rows]
    val_margins = [r["val_reward_margin"] for r in val_rows]
    val_chosen = [r.get("val_chosen_reward", 0.0) for r in val_rows]
    val_rejected = [r.get("val_rejected_reward", 0.0) for r in val_rows]

    best_step = None
    best_acc = -1.0
    for r in val_rows:
        if r["step"] > 0 and r["val_reward_accuracy"] > best_acc:
            best_acc = r["val_reward_accuracy"]
            best_step = r["step"]
    if best_step is None and val_rows:
        best_step = val_rows[-1]["step"]
        best_acc = val_rows[-1]["val_reward_accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel 1: DPO Loss
    ax = axes[0]
    if train_losses:
        ax.plot(train_steps, train_losses, color="#90caf9", alpha=0.4, linewidth=1.0, label="Train Loss (raw)")
        ax.plot(
            train_steps,
            ema_smooth(train_losses, alpha=0.15),
            color="#1565c0",
            linewidth=2.0,
            label="Train Loss (EMA α=0.15)",
        )
    if val_losses:
        ax.plot(val_steps, val_losses, color="#d32f2f", marker="o", linewidth=2.0, label="Val Loss")
    ax.axhline(math.log(2.0), color="gray", linestyle=":", linewidth=1.2, label="ln(2) ≈ 0.6931")
    if best_step is not None:
        ax.axvline(best_step, color="#2e7d32", linestyle="--", linewidth=1.4, label=f"Best Step ({best_step})")
    ax.set_title(f"{run_label}: DPO Loss")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("DPO Logistic Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7.5)

    # Panel 2: Reward Accuracy
    ax = axes[1]
    if train_accs:
        ax.plot(
            train_steps,
            ema_smooth(train_accs, alpha=0.15),
            color="#1565c0",
            linewidth=1.8,
            alpha=0.8,
            label="Train Reward Acc (EMA α=0.15)",
        )
    if val_accs:
        ax.plot(val_steps, val_accs, color="#d32f2f", marker="s", linewidth=2.0, label="Val Reward Acc")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random tie-break (0.5)")
    if best_step is not None:
        ax.axvline(
            best_step,
            color="#2e7d32",
            linestyle="--",
            linewidth=1.4,
            label=f"Selected Best (step {best_step}, acc={best_acc:.3f})",
        )
    ax.set_title(f"{run_label}: Implicit Reward Accuracy")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Fraction with r_w > r_l")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7.5)

    # Panel 3: Reward Margin & Chosen/Rejected Rewards
    ax = axes[2]
    if train_margins:
        ax.plot(
            train_steps,
            ema_smooth(train_margins, alpha=0.15),
            color="#1565c0",
            linewidth=1.8,
            alpha=0.8,
            label="Train Margin (EMA α=0.15)",
        )
    if val_margins:
        ax.plot(val_steps, val_margins, color="#d32f2f", marker="^", linewidth=2.0, label="Val Margin (r_w - r_l)")
        ax.plot(val_steps, val_chosen, color="#2e7d32", linestyle="-.", marker=".", linewidth=1.4, label="Val Chosen r_w")
        ax.plot(val_steps, val_rejected, color="#ef6c00", linestyle="-.", marker=".", linewidth=1.4, label="Val Rejected r_l")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    if best_step is not None:
        ax.axvline(best_step, color="#2e7d32", linestyle="--", linewidth=1.4, label=f"Best Step ({best_step})")
    ax.set_title(f"{run_label}: Implicit Reward Margin")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Implicit Reward (β = 0.1)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out_path = out_dir / f"{filename_stem}.png"
    fig.savefig(out_path)
    fig.savefig(out_dir / f"{filename_stem}.pdf")
    plt.close(fig)
    print(f"Saved DPO plot to {out_path}")
    return out_path


def plot_lora_persona(metrics_path: Path, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    rows = load_jsonl(metrics_path)
    if not rows:
        print(f"[skip] No LoRA Persona metrics found at {metrics_path}")
        return None

    train_rows = [r for r in rows if "loss" in r and "eval_loss" not in r]
    eval_rows = [r for r in rows if "eval_loss" in r]

    t_steps = [r.get("step", i) for i, r in enumerate(train_rows)]
    t_loss = [r["loss"] for r in train_rows]
    t_acc = [r.get("rewards/accuracies", None) for r in train_rows]
    t_margin = [r.get("rewards/margins", None) for r in train_rows]

    e_steps = [r.get("step", 0) for r in eval_rows]
    e_loss = [r["eval_loss"] for r in eval_rows]
    e_acc = [r.get("eval_rewards/accuracies", None) for r in eval_rows]
    e_margin = [r.get("eval_rewards/margins", None) for r in eval_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=300)

    axes[0].plot(t_steps, t_loss, color="#1565c0", marker=".", label="Train Loss")
    if e_loss:
        axes[0].plot(e_steps, e_loss, color="#d32f2f", marker="o", linewidth=2.0, label="Val Loss")
    axes[0].axhline(math.log(2.0), color="gray", linestyle=":", label="ln(2)")
    axes[0].set_title("LoRA Persona DPO: Loss")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("DPO Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)

    if any(x is not None for x in t_acc):
        axes[1].plot(t_steps, t_acc, color="#1565c0", marker=".", label="Train Reward Acc")
    if any(x is not None for x in e_acc):
        axes[1].plot(e_steps, e_acc, color="#d32f2f", marker="s", linewidth=2.0, label="Val Reward Acc")
    axes[1].set_title("LoRA Persona DPO: Reward Accuracy")
    axes[1].set_xlabel("Optimizer Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)

    if any(x is not None for x in t_margin):
        axes[2].plot(t_steps, t_margin, color="#1565c0", marker=".", label="Train Reward Margin")
    if any(x is not None for x in e_margin):
        axes[2].plot(e_steps, e_margin, color="#d32f2f", marker="^", linewidth=2.0, label="Val Reward Margin")
    axes[2].set_title("LoRA Persona DPO: Reward Margin")
    axes[2].set_xlabel("Optimizer Step")
    axes[2].set_ylabel("Margin")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "lora_persona_training.png"
    fig.savefig(out_path)
    fig.savefig(out_dir / "lora_persona_training.pdf")
    plt.close(fig)
    print(f"Saved LoRA Persona plot to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training plots for SFT, DPO, and LoRA-DPO.")
    parser.add_argument(
        "--sft_metrics",
        type=str,
        default="outputs/kaggle/align-sft/outputs/sft/metrics.jsonl",
    )
    parser.add_argument(
        "--dpo_hh_metrics",
        type=str,
        default="outputs/kaggle/align-dpo-hh/outputs/dpo_hh/metrics.jsonl",
    )
    parser.add_argument(
        "--dpo_uf_metrics",
        type=str,
        default="outputs/kaggle/align-dpo-uf/outputs/dpo_uf/metrics.jsonl",
    )
    parser.add_argument(
        "--lora_metrics",
        type=str,
        default="outputs/kaggle/align-lora-persona/outputs/lora_persona/metrics.jsonl",
    )
    parser.add_argument("--figures_dir", type=str, default="report/figures")
    args = parser.parse_args()

    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sft(Path(args.sft_metrics), out_dir)
    plot_dpo_run(Path(args.dpo_hh_metrics), "HH-RLHF DPO", "dpo_hh_training", out_dir)
    plot_dpo_run(Path(args.dpo_uf_metrics), "UltraFeedback DPO", "dpo_uf_training", out_dir)
    plot_lora_persona(Path(args.lora_metrics), out_dir)


if __name__ == "__main__":
    main()
