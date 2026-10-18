import json
from pathlib import Path

def extract_track_examples(judge_file, gen_cand, gen_base, cand_name, base_name, track_name):
    j_rows = [json.loads(l) for l in open(judge_file, encoding="utf-8") if l.strip()]
    cand_data = {r["id"]: r for r in json.load(open(gen_cand, encoding="utf-8"))}
    base_data = {r["id"]: r for r in json.load(open(gen_base, encoding="utf-8"))}

    improvements = []
    regressions = []

    for r in j_rows:
        pid = r["pair_id"]
        w = r["winner_model"]
        inst = r["instruction"]
        reason = r.get("reason", "")
        c_out = cand_data[pid]["output"]
        b_out = base_data[pid]["output"]

        # Only select examples with substantive, interesting reasons
        if w == cand_name and len(c_out) > 50 and len(b_out) > 50 and len(reason) > 30:
            improvements.append({
                "id": pid,
                "instruction": inst,
                "cand_output": c_out,
                "base_output": b_out,
                "reason": reason,
            })
        elif w == base_name and len(c_out) > 50 and len(b_out) > 50 and len(reason) > 30:
            regressions.append({
                "id": pid,
                "instruction": inst,
                "cand_output": c_out,
                "base_output": b_out,
                "reason": reason,
            })

    print(f"\n=================== {track_name} ===================")
    print(f"Found {len(improvements)} improvement candidates, {len(regressions)} regression candidates.")
    
    print("\n--- 2 IMPROVEMENT EXAMPLES (Candidate > Baseline) ---")
    for i, ex in enumerate(improvements[:2], 1):
        print(f"\n[Improvement {i}] ID: {ex['id']}")
        print(f"Instruction: {ex['instruction'][:120]}...")
        print(f"Candidate ({cand_name[:20]}): {ex['cand_output'][:200]}...")
        print(f"Baseline ({base_name[:20]}): {ex['base_output'][:200]}...")
        print(f"Judge Reason: {ex['reason']}")

    print("\n--- 2 REGRESSION EXAMPLES (Baseline > Candidate) ---")
    for i, ex in enumerate(regressions[:2], 1):
        print(f"\n[Regression {i}] ID: {ex['id']}")
        print(f"Instruction: {ex['instruction'][:120]}...")
        print(f"Candidate ({cand_name[:20]}): {ex['cand_output'][:200]}...")
        print(f"Baseline ({base_name[:20]}): {ex['base_output'][:200]}...")
        print(f"Judge Reason: {ex['reason']}")

    return improvements[:2], regressions[:2]

def main():
    uf_imp, uf_reg = extract_track_examples(
        "outputs/judgments/judge_alpaca_sft_vs_dpo_uf_rp11.jsonl",
        "outputs/generations/alpaca_dpo_uf_rp11.json",
        "outputs/generations/alpaca_sft_rp11.json",
        "qwen2.5-0.5b-dpo-ultrafeedback-rp11",
        "qwen2.5-0.5b-sft-rp11",
        "Track B: UltraFeedback (Helpfulness/General)",
    )

    hh_imp, hh_reg = extract_track_examples(
        "outputs/judgments/judge_hh_sft_vs_dpo_hh_rp11.jsonl",
        "outputs/generations/hh_dpo_hh_rp11.json",
        "outputs/generations/hh_sft_rp11.json",
        "qwen2.5-0.5b-dpo-hh-rp11",
        "qwen2.5-0.5b-sft-rp11",
        "Track A: Anthropic HH-RLHF (Safety/Harmlessness)",
    )

    # Save to json for exact inclusion in LaTeX script
    out = {
        "uf_improvements": uf_imp,
        "uf_regressions": uf_reg,
        "hh_improvements": hh_imp,
        "hh_regressions": hh_reg,
    }
    with open("scratch/qualitative_quotes.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved qualitative quotes to scratch/qualitative_quotes.json")

if __name__ == "__main__":
    main()
