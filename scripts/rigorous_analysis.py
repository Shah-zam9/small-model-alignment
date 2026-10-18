import json
import math
from pathlib import Path
from scipy import stats

def binomial_analysis(wins: int, losses: int, ties: int, total_n: int):
    decisive = wins + losses
    valid = wins + losses + ties
    wr_excl = wins / decisive if decisive > 0 else 0.0
    wr_half = (wins + 0.5 * ties) / valid if valid > 0 else 0.0

    # Exact binomial test on decisive outcomes: H0: p = 0.5
    if decisive > 0:
        res = stats.binomtest(wins, decisive, p=0.5, alternative="two-sided")
        p_val = res.pvalue
        ci_low, ci_high = res.proportion_ci(confidence_level=0.95, method="exact")
    else:
        p_val = 1.0
        ci_low, ci_high = 0.0, 0.0

    # Wilson score interval for ties-as-half (or decisive)
    z = 1.95996
    # Wilson for decisive
    if decisive > 0:
        p_hat = wins / decisive
        denom = 1 + z**2 / decisive
        center = (p_hat + z**2 / (2 * decisive)) / denom
        diff = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * decisive)) / decisive) / denom
        w_ci_low, w_ci_high = max(0.0, center - diff), min(1.0, center + diff)
    else:
        w_ci_low, w_ci_high = 0.0, 0.0

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "decisive": decisive,
        "wr_excl": wr_excl,
        "wr_half": wr_half,
        "p_val": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilson_low": w_ci_low,
        "wilson_high": w_ci_high,
    }

def main():
    judgments_dir = Path("outputs/judgments")

    files = [
        ("Tier (a): AlpacaEval 300", "DPO-UF vs SFT", "qwen2.5-0.5b-dpo-ultrafeedback-rp11", "qwen2.5-0.5b-sft-rp11", "judge_alpaca_sft_vs_dpo_uf_rp11.jsonl"),
        ("Tier (a): HH-RLHF 300", "DPO-HH vs SFT", "qwen2.5-0.5b-dpo-hh-rp11", "qwen2.5-0.5b-sft-rp11", "judge_hh_sft_vs_dpo_hh_rp11.jsonl"),
        ("Tier (a): Persona 24 (Overall)", "3-Ep LoRA vs Base", "smollm2-360m-instruct-ramsay-lora-dpo-rp11", "smollm2-360m-instruct-base-rp11", "judge_persona_base_vs_lora_3ep_rp11.jsonl"),
        ("Tier (c): AlpacaEval 300", "SFT vs Base", "qwen2.5-0.5b-sft-rp11", "qwen2.5-0.5b-base-rp11", "judge_alpaca_base_vs_sft_rp11.jsonl"),
        ("Tier (c): HH-RLHF 300", "SFT vs Base", "qwen2.5-0.5b-sft-rp11", "qwen2.5-0.5b-base-rp11", "judge_hh_base_vs_sft_rp11.jsonl"),
        ("Tier (c): Persona 24 (Overall)", "1-Ep LoRA vs Base", "smollm2-360m-instruct-ramsay-lora-1ep-rp11", "smollm2-360m-instruct-base-rp11", "judge_persona_base_vs_lora_1ep_rp11.jsonl"),
        ("Tier (c): Persona 24 (Overall)", "3-Ep vs 1-Ep LoRA", "smollm2-360m-instruct-ramsay-lora-dpo-rp11", "smollm2-360m-instruct-ramsay-lora-1ep-rp11", "judge_persona_lora_1ep_vs_3ep_rp11.jsonl"),
    ]

    print("=== 1 & 2. STATISTICAL HYPOTHESIS TESTING (EXPLICIT DIRECTION & 95% CIs) ===")
    print(f"{'Comparison':24s} | {'N':4s} | {'W-L-T':12s} | {'Win Rate (excl)':16s} | {'95% Exact CI':16s} | {'p-value':10s} | {'Significance (alpha=0.05)'}")
    print("-" * 115)

    for tier_label, comp_name, cand, base, fn in files:
        rows = [json.loads(l) for l in open(judgments_dir / fn, encoding="utf-8") if l.strip()]
        wins = sum(1 for r in rows if r["winner_model"] == cand)
        losses = sum(1 for r in rows if r["winner_model"] == base)
        ties = sum(1 for r in rows if r["winner_model"] == "tie")
        total_n = len(rows)

        stat = binomial_analysis(wins, losses, ties, total_n)
        sig = "Significant (p < 0.05)" if stat["p_val"] < 0.05 else "Not significant (p >= 0.05)"
        ci_str = f"[{stat['ci_low']*100:.1f}%, {stat['ci_high']*100:.1f}%]"
        wlt = f"{wins}-{losses}-{ties}"
        print(f"{comp_name:24s} | {total_n:4d} | {wlt:12s} | {stat['wr_excl']*100:15.2f}% | {ci_str:16s} | {stat['p_val']:10.4e} | {sig}")

    # Item 3: Overall "A" rate (position bias check across all pairs in each file)
    print("\n=== 3. POSITION BIAS: OVERALL 'A' CHOICE RATE ===")
    for tier_label, comp_name, cand, base, fn in files:
        rows = [json.loads(l) for l in open(judgments_dir / fn, encoding="utf-8") if l.strip()]
        a_count = sum(1 for r in rows if r.get("winner") == "A")
        b_count = sum(1 for r in rows if r.get("winner") == "B")
        tie_count = sum(1 for r in rows if r.get("winner") == "tie")
        decisive_ab = a_count + b_count
        a_rate_excl = a_count / decisive_ab * 100 if decisive_ab > 0 else 0.0
        a_rate_all = a_count / len(rows) * 100 if rows else 0.0
        print(f"{comp_name:24s} | Total: {len(rows):3d} | A: {a_count:3d} ({a_rate_all:5.1f}%) | B: {b_count:3d} ({b_count/len(rows)*100:5.1f}%) | Tie: {tie_count:3d} | A-rate (excl ties): {a_rate_excl:5.1f}%")

    # Item 4: Length split EXCLUDING 512-token hits
    print("\n=== 4. DPO-UF vs SFT LENGTH SPLIT EXCLUDING 512-TOKEN HITS ===")
    uf_gen = {r["id"]: r for r in json.load(open("outputs/generations/alpaca_dpo_uf_rp11.json", encoding="utf-8"))}
    sft_gen = {r["id"]: r for r in json.load(open("outputs/generations/alpaca_sft_rp11.json", encoding="utf-8"))}
    rows = [json.loads(l) for l in open(judgments_dir / "judge_alpaca_sft_vs_dpo_uf_rp11.jsonl", encoding="utf-8") if l.strip()]

    # Filter out any pair where EITHER model hit 512 tokens
    clean_rows = []
    excluded_count = 0
    for r in rows:
        pid = r["pair_id"]
        uf_hit = uf_gen[pid].get("hit_max_new_tokens", uf_gen[pid]["num_generated_tokens"] >= 512)
        sft_hit = sft_gen[pid].get("hit_max_new_tokens", sft_gen[pid]["num_generated_tokens"] >= 512)
        if uf_hit or sft_hit:
            excluded_count += 1
        else:
            clean_rows.append(r)

    cand = "qwen2.5-0.5b-dpo-ultrafeedback-rp11"
    base = "qwen2.5-0.5b-sft-rp11"
    print(f"Total pairs: {len(rows)}, Excluded (hit 512 in either model): {excluded_count}, Clean pairs: {len(clean_rows)}")

    splits = {
        "DPO-UF longer than SFT (no hits)": [r for r in clean_rows if uf_gen[r["pair_id"]]["num_generated_tokens"] > sft_gen[r["pair_id"]]["num_generated_tokens"]],
        "DPO-UF shorter than SFT (no hits)": [r for r in clean_rows if uf_gen[r["pair_id"]]["num_generated_tokens"] < sft_gen[r["pair_id"]]["num_generated_tokens"]],
        "Equal length (no hits)": [r for r in clean_rows if uf_gen[r["pair_id"]]["num_generated_tokens"] == sft_gen[r["pair_id"]]["num_generated_tokens"]],
    }

    print(f"{'Condition':36s} | {'N':4s} | {'Wins':5s} | {'Losses':6s} | {'Ties':5s} | {'Win Rate (excl)':16s} | {'Win Rate (half)'}")
    print("-" * 95)
    for name, group in splits.items():
        n = len(group)
        w = sum(1 for r in group if r["winner_model"] == cand)
        l = sum(1 for r in group if r["winner_model"] == base)
        t = sum(1 for r in group if r["winner_model"] == "tie")
        dec = w + l
        wr_e = w / dec * 100 if dec > 0 else 0.0
        wr_h = (w + 0.5 * t) / n * 100 if n > 0 else 0.0
        print(f"{name:36s} | {n:4d} | {w:5d} | {l:6d} | {t:5d} | {wr_e:15.2f}% | {wr_h:14.2f}%")

if __name__ == "__main__":
    main()
