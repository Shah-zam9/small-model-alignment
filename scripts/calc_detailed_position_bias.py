import json
from pathlib import Path

def analyze_track(judge_file, cand_name, baseline_name):
    rows = [json.loads(line) for line in Path(judge_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    
    cand_as_a = []
    cand_as_b = []
    
    for r in rows:
        meta = r.get("meta", {})
        a_model = meta.get("a_model", "")
        b_model = meta.get("b_model", "")
        winner = r.get("winner", "")
        winner_model = r.get("winner_model", "")
        
        # Determine slot
        if cand_name in a_model:
            cand_as_a.append(r)
        elif cand_name in b_model:
            cand_as_b.append(r)
        else:
            print(f"Warning: neither matches {cand_name}: a={a_model}, b={b_model}")
            
    def get_stats(subset, slot_label):
        n = len(subset)
        wins = sum(1 for r in subset if r.get("winner_model") == cand_name)
        losses = sum(1 for r in subset if r.get("winner_model") == baseline_name)
        ties = sum(1 for r in subset if r.get("winner") == "tie" or r.get("winner_model") == "tie")
        unknowns = n - (wins + losses + ties)
        decisive = wins + losses
        win_rate_excl = (wins / decisive * 100) if decisive > 0 else 0.0
        win_rate_half = ((wins + 0.5 * ties) / n * 100) if n > 0 else 0.0
        return {
            "slot": slot_label,
            "N": n,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "unknowns": unknowns,
            "win_rate_excl": win_rate_excl,
            "win_rate_half": win_rate_half
        }
        
    stats_a = get_stats(cand_as_a, "Response A")
    stats_b = get_stats(cand_as_b, "Response B")
    return stats_a, stats_b

print("=== AlpacaEval 300: DPO-UF vs SFT ===")
uf_a, uf_b = analyze_track(
    "outputs/judgments/judge_alpaca_sft_vs_dpo_uf_rp11.jsonl",
    cand_name="qwen2.5-0.5b-dpo-ultrafeedback-rp11",
    baseline_name="qwen2.5-0.5b-sft-rp11"
)
print(f"DPO-UF as Response A: N={uf_a['N']}, Wins={uf_a['wins']}, Losses={uf_a['losses']}, Ties={uf_a['ties']}, Win%(excl)={uf_a['win_rate_excl']:.2f}%, Win%(half)={uf_a['win_rate_half']:.2f}%")
print(f"DPO-UF as Response B: N={uf_b['N']}, Wins={uf_b['wins']}, Losses={uf_b['losses']}, Ties={uf_b['ties']}, Win%(excl)={uf_b['win_rate_excl']:.2f}%, Win%(half)={uf_b['win_rate_half']:.2f}%")

print("\n=== HH-RLHF 300: DPO-HH vs SFT ===")
hh_a, hh_b = analyze_track(
    "outputs/judgments/judge_hh_sft_vs_dpo_hh_rp11.jsonl",
    cand_name="qwen2.5-0.5b-dpo-hh-rp11",
    baseline_name="qwen2.5-0.5b-sft-rp11"
)
print(f"DPO-HH as Response A: N={hh_a['N']}, Wins={hh_a['wins']}, Losses={hh_a['losses']}, Ties={hh_a['ties']}, Win%(excl)={hh_a['win_rate_excl']:.2f}%, Win%(half)={hh_a['win_rate_half']:.2f}%")
print(f"DPO-HH as Response B: N={hh_b['N']}, Wins={hh_b['wins']}, Losses={hh_b['losses']}, Ties={hh_b['ties']}, Win%(excl)={hh_b['win_rate_excl']:.2f}%, Win%(half)={hh_b['win_rate_half']:.2f}%")

def compare_swapped_50(orig_file, swap_file, cand_name, baseline_name):
    orig_rows = {r["pair_id"]: r for r in [json.loads(line) for line in Path(orig_file).read_text(encoding="utf-8").splitlines() if line.strip()]}
    swap_rows = {r["pair_id"]: r for r in [json.loads(line) for line in Path(swap_file).read_text(encoding="utf-8").splitlines() if line.strip()]}
    
    common_ids = list(swap_rows.keys())
    print(f"\nAnalyzing 50 swapped pairs for {cand_name} (found {len(common_ids)} in swapped):")
    
    # Stats on original
    orig_subset = [orig_rows[pid] for pid in common_ids if pid in orig_rows]
    swap_subset = [swap_rows[pid] for pid in common_ids]
    
    def calc_cand(subset):
        w = sum(1 for r in subset if r.get("winner_model") == cand_name)
        l = sum(1 for r in subset if r.get("winner_model") == baseline_name)
        t = sum(1 for r in subset if r.get("winner") == "tie" or r.get("winner_model") == "tie")
        d = w + l
        rate = (w / d * 100) if d > 0 else 0.0
        return w, l, t, rate
        
    ow, ol, ot, orate = calc_cand(orig_subset)
    sw, sl, st, srate = calc_cand(swap_subset)
    
    print(f"  ORIGINAL Order on 50 pairs: Wins={ow}, Losses={ol}, Ties={ot}, Win%(excl)={orate:.2f}%")
    print(f"  SWAPPED Order on 50 pairs:  Wins={sw}, Losses={sl}, Ties={st}, Win%(excl)={srate:.2f}%")

compare_swapped_50(
    "outputs/judgments/judge_alpaca_sft_vs_dpo_uf_rp11.jsonl",
    "outputs/judgments/judge_swap50_alpaca_sft_vs_dpo_uf_rp11.jsonl",
    cand_name="qwen2.5-0.5b-dpo-ultrafeedback-rp11",
    baseline_name="qwen2.5-0.5b-sft-rp11"
)

compare_swapped_50(
    "outputs/judgments/judge_hh_sft_vs_dpo_hh_rp11.jsonl",
    "outputs/judgments/judge_swap50_hh_sft_vs_dpo_hh_rp11.jsonl",
    cand_name="qwen2.5-0.5b-dpo-hh-rp11",
    baseline_name="qwen2.5-0.5b-sft-rp11"
)
