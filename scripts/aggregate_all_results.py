import json
from pathlib import Path

def main():
    judgments_dir = Path("outputs/judgments")
    files = [
        # Tier a
        ("Tier (a): AlpacaEval 300", "SFT vs DPO-UF (rp=1.1)", "judge_alpaca_sft_vs_dpo_uf_rp11_summary.json"),
        ("Tier (a): HH-RLHF 300", "SFT vs DPO-HH (rp=1.1)", "judge_hh_sft_vs_dpo_hh_rp11_summary.json"),
        ("Tier (a): Persona 24", "Base vs 3-Ep LoRA (rp=1.1)", "judge_persona_base_vs_lora_3ep_rp11_summary.json"),
        # Tier b
        ("Tier (b): AlpacaEval 50 Swapped", "SFT vs DPO-UF Swapped", "judge_swap50_alpaca_sft_vs_dpo_uf_rp11_summary.json"),
        ("Tier (b): HH-RLHF 50 Swapped", "SFT vs DPO-HH Swapped", "judge_swap50_hh_sft_vs_dpo_hh_rp11_summary.json"),
        # Tier c
        ("Tier (c): Persona 24", "Base vs 1-Ep LoRA (rp=1.1)", "judge_persona_base_vs_lora_1ep_rp11_summary.json"),
        ("Tier (c): Persona 24", "1-Ep vs 3-Ep LoRA (rp=1.1)", "judge_persona_lora_1ep_vs_3ep_rp11_summary.json"),
        ("Tier (c): AlpacaEval 300", "Base vs SFT (rp=1.1)", "judge_alpaca_base_vs_sft_rp11_summary.json"),
        ("Tier (c): HH-RLHF 300", "Base vs SFT (rp=1.1)", "judge_hh_base_vs_sft_rp11_summary.json"),
    ]

    print("========================= ALL JUDGMENT RESULTS =========================")
    print(f"{'Tier & Set':32s} | {'Comparison':24s} | {'N':4s} | {'Wins':5s} | {'Losses':6s} | {'Ties':5s} | {'Unk':4s} | {'Win Rate (excl)':16s} | {'Win Rate (half)':16s}")
    print("-" * 125)

    for tier_label, comp_label, fname in files:
        fpath = judgments_dir / fname
        if not fpath.exists():
            print(f"{tier_label:32s} | {comp_label:24s} | MISSING FILE")
            continue
        s = json.load(open(fpath, encoding="utf-8"))
        n = s.get("n", 0)
        wins = s.get("wins", 0)
        losses = s.get("losses", 0)
        ties = s.get("ties", 0)
        unk = s.get("unknown", 0)
        w_excl = s.get("win_rate_excluding_ties", 0.0) * 100
        w_half = s.get("win_rate_ties_as_half", 0.0) * 100
        print(f"{tier_label:32s} | {comp_label:24s} | {n:4d} | {wins:5d} | {losses:6d} | {ties:5d} | {unk:4d} | {w_excl:15.2f}% | {w_half:15.2f}%")

if __name__ == "__main__":
    main()
