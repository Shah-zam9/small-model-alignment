import json
from pathlib import Path

def main():
    judge_file = Path("outputs/judgments/judge_alpaca_sft_vs_dpo_uf_rp11.jsonl")
    uf_gen_file = Path("outputs/generations/alpaca_dpo_uf_rp11.json")
    sft_gen_file = Path("outputs/generations/alpaca_sft_rp11.json")

    uf_data = {r["id"]: r for r in json.load(open(uf_gen_file, encoding="utf-8"))}
    sft_data = {r["id"]: r for r in json.load(open(sft_gen_file, encoding="utf-8"))}

    rows = []
    with open(judge_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    candidate = "qwen2.5-0.5b-dpo-ultrafeedback-rp11"
    baseline = "qwen2.5-0.5b-sft-rp11"

    splits = {
        "DPO-UF longer than SFT": [],
        "DPO-UF shorter than SFT": [],
        "Equal length": [],
    }

    for r in rows:
        pid = r["pair_id"]
        uf_len = uf_data[pid]["num_generated_tokens"]
        sft_len = sft_data[pid]["num_generated_tokens"]

        if uf_len > sft_len:
            splits["DPO-UF longer than SFT"].append(r)
        elif uf_len < sft_len:
            splits["DPO-UF shorter than SFT"].append(r)
        else:
            splits["Equal length"].append(r)

    print("=== DPO-UF vs SFT WIN RATE SPLIT BY LENGTH ===")
    print(f"{'Split':26s} | {'N':4s} | {'Wins':5s} | {'Losses':6s} | {'Ties':5s} | {'Win Rate (excl ties)':20s} | {'Win Rate (ties half)':20s}")
    print("-" * 100)

    for name, group in splits.items():
        n = len(group)
        if n == 0:
            print(f"{name:26s} | {0:4d} | {0:5d} | {0:6d} | {0:5d} | {0.0:20.2f}% | {0.0:20.2f}%")
            continue
        wins = sum(1 for item in group if item["winner_model"] == candidate)
        losses = sum(1 for item in group if item["winner_model"] == baseline)
        ties = sum(1 for item in group if item["winner_model"] == "tie")
        decisive = wins + losses
        valid = wins + losses + ties
        wr_excl = round(wins / decisive * 100, 2) if decisive > 0 else 0.0
        wr_half = round((wins + 0.5 * ties) / valid * 100, 2) if valid > 0 else 0.0
        print(f"{name:26s} | {n:4d} | {wins:5d} | {losses:6d} | {ties:5d} | {wr_excl:19.2f}% | {wr_half:19.2f}%")

if __name__ == "__main__":
    main()
