import json
import random
from pathlib import Path

def create_swap_file(input_path: Path, output_path: Path, n: int = 50, seed: int = 42):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))

    swapped_rows = []
    for r in sample:
        meta = dict(r.get("meta", {}))
        orig_a_model = meta.get("a_model")
        orig_b_model = meta.get("b_model")

        swapped_meta = dict(meta)
        swapped_meta["a_model"] = orig_b_model
        swapped_meta["b_model"] = orig_a_model
        swapped_meta["is_swapped"] = True

        swapped_row = {
            "pair_id": r["pair_id"],
            "instruction": r["instruction"],
            "response_a": r["response_b"],
            "response_b": r["response_a"],
            "dataset": r.get("dataset", ""),
            "meta": swapped_meta,
        }
        if "category" in r:
            swapped_row["category"] = r["category"]
        swapped_rows.append(swapped_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for r in swapped_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(swapped_rows)} swapped pairs to {output_path}")

def main():
    create_swap_file(
        Path("outputs/pairwise/pairs_alpaca_sft_vs_dpo_uf_rp11.jsonl"),
        Path("outputs/pairwise/swap50_alpaca_sft_vs_dpo_uf_rp11.jsonl"),
        n=50,
        seed=42,
    )
    create_swap_file(
        Path("outputs/pairwise/pairs_hh_sft_vs_dpo_hh_rp11.jsonl"),
        Path("outputs/pairwise/swap50_hh_sft_vs_dpo_hh_rp11.jsonl"),
        n=50,
        seed=42,
    )

if __name__ == "__main__":
    main()
