#!/usr/bin/env python3
"""Print validation tables and hyperparameters directly from metrics.jsonl and summary.json."""

import json

def print_run_details(name, metrics_path, summary_path):
    print("=" * 90)
    print(f"RUN: {name}")
    print("=" * 90)
    with open(summary_path, "r") as sf:
        summary = json.load(sf)
    
    val_records = []
    with open(metrics_path, "r") as mf:
        for line in mf:
            d = json.loads(line)
            vl = d.get("val_loss") or d.get("eval_loss")
            if vl is not None:
                val_records.append(d)
                
    print(f"Summary Metadata:")
    for k in ["total_steps_run", "best_val_loss", "best_val_reward_accuracy", "learning_rate", "epochs", "effective_batch_size", "train_runtime_sec", "elapsed_sec"]:
        if k in summary:
            print(f"  {k}: {summary[k]}")
    if "longest_pair_probe" in summary:
        print("  Longest Pair Probe:")
        print(f"    Gradient Checkpointing Enabled: {summary['longest_pair_probe'].get('gradient_checkpointing_enabled')}")
        print(f"    Decision: {summary['longest_pair_probe'].get('decision')}")

    print("\nValidation Table:")
    # Header
    print(f"{'Step':>6} | {'Epoch':>5} | {'Val Loss':>10} | {'Reward Acc':>10} | {'Margin':>10} | {'Chosen R':>10} | {'Rejected R':>10}")
    print("-" * 80)
    for r in val_records:
        step = r.get("step")
        epoch = r.get("epoch", "-")
        loss = r.get("val_loss") or r.get("eval_loss")
        acc = r.get("val_reward_accuracy") or r.get("eval_rewards/accuracies")
        margin = r.get("val_reward_margin") or r.get("eval_rewards/margins")
        chosen = r.get("val_chosen_reward") or r.get("eval_rewards/chosen")
        rej = r.get("val_rejected_reward") or r.get("eval_rewards/rejected")
        
        loss_str = f"{loss:.6f}" if loss is not None else "-"
        acc_str = f"{acc:.4f}" if acc is not None else "-"
        margin_str = f"{margin:.6f}" if margin is not None else "-"
        chosen_str = f"{chosen:.6f}" if chosen is not None else "-"
        rej_str = f"{rej:.6f}" if rej is not None else "-"
        epoch_str = f"{epoch:.1f}" if isinstance(epoch, (int, float)) else str(epoch)
        print(f"{step:>6} | {epoch_str:>5} | {loss_str:>10} | {acc_str:>10} | {margin_str:>10} | {chosen_str:>10} | {rej_str:>10}")

def main():
    runs = [
        ("SFT (UltraChat)", "outputs/sft/metrics.jsonl", "outputs/sft/summary.json"),
        ("DPO-HH (Track A)", "outputs/dpo_hh/metrics.jsonl", "outputs/dpo_hh/summary.json"),
        ("DPO-UF (Track B)", "outputs/dpo_uf/metrics.jsonl", "outputs/dpo_uf/summary.json"),
        ("LoRA-3ep (SmolLM2)", "outputs/lora_persona/metrics.jsonl", "outputs/lora_persona/summary.json"),
        ("LoRA-1ep (SmolLM2)", "outputs/lora_persona_1ep/metrics.jsonl", "outputs/lora_persona_1ep/summary.json"),
    ]
    for name, mp, sp in runs:
        print_run_details(name, mp, sp)

if __name__ == "__main__":
    main()
