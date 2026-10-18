import json
from pathlib import Path

def evaluate_agreement(orig_path: Path, swap_path: Path, label: str):
    orig_rows = {json.loads(line)["pair_id"]: json.loads(line) for line in open(orig_path, encoding="utf-8") if line.strip()}
    swap_rows = {json.loads(line)["pair_id"]: json.loads(line) for line in open(swap_path, encoding="utf-8") if line.strip()}

    common_ids = sorted(set(orig_rows.keys()) & set(swap_rows.keys()))
    print(f"\n=== POSITION-SWAP CONSISTENCY: {label} (N={len(common_ids)}) ===")

    exact_matches = 0
    ties_in_either = 0
    flips = 0

    for pid in common_ids:
        orig_w = orig_rows[pid]["winner_model"]
        swap_w = swap_rows[pid]["winner_model"]

        if orig_w == swap_w:
            exact_matches += 1
        elif orig_w == "tie" or swap_w == "tie":
            ties_in_either += 1
        else:
            flips += 1

    agr_rate = round(exact_matches / len(common_ids) * 100, 2)
    decisive_matches = exact_matches - sum(1 for pid in common_ids if orig_rows[pid]["winner_model"] == "tie" and swap_rows[pid]["winner_model"] == "tie")
    print(f"Total pairs compared: {len(common_ids)}")
    print(f"Exact model agreement: {exact_matches}/{len(common_ids)} ({agr_rate}%)")
    print(f"Disagreements involving ties: {ties_in_either}/{len(common_ids)} ({round(ties_in_either/len(common_ids)*100, 2)}%)")
    print(f"Direct candidate/baseline flips: {flips}/{len(common_ids)} ({round(flips/len(common_ids)*100, 2)}%)")

def main():
    evaluate_agreement(
        Path("outputs/judgments/judge_alpaca_sft_vs_dpo_uf_rp11.jsonl"),
        Path("outputs/judgments/judge_swap50_alpaca_sft_vs_dpo_uf_rp11.jsonl"),
        "AlpacaEval 300 (SFT vs DPO-UF)",
    )
    evaluate_agreement(
        Path("outputs/judgments/judge_hh_sft_vs_dpo_hh_rp11.jsonl"),
        Path("outputs/judgments/judge_swap50_hh_sft_vs_dpo_hh_rp11.jsonl"),
        "HH-RLHF 300 (SFT vs DPO-HH)",
    )

if __name__ == "__main__":
    main()
