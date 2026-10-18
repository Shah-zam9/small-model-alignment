import json
from pathlib import Path

eval_sets = {
    "alpaca": json.load(open("eval_sets/alpaca_eval_300_seed42.json", encoding="utf-8")),
    "hh": json.load(open("eval_sets/hh_rlhf_eval_300_seed42.json", encoding="utf-8")),
    "persona": json.load(open("eval_sets/persona_style_eval.json", encoding="utf-8")),
}

# Build id -> instruction map for each eval set
eval_maps = {}
for name, rows in eval_sets.items():
    mapping = {}
    for r in rows:
        pid = str(r.get("id"))
        inst = r.get("instruction")
        mapping[pid] = inst
    eval_maps[name] = mapping
    print(f"Eval set '{name}': {len(mapping)} prompts loaded.")

pairwise_dir = Path("outputs/pairwise")
pairwise_files = sorted(pairwise_dir.glob("*.jsonl"))
print(f"\nChecking {len(pairwise_files)} pairwise files...")

total_mismatches = 0
total_checked = 0

for pf in pairwise_files:
    fname = pf.name
    if "alpaca" in fname:
        target_eval = "alpaca"
    elif "hh" in fname:
        target_eval = "hh"
    elif "persona" in fname:
        target_eval = "persona"
    else:
        print(f"UNKNOWN DATASET FOR {fname}")
        continue

    ref_map = eval_maps[target_eval]
    mismatches = []
    line_count = 0
    with open(pf, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            line_count += 1
            total_checked += 1
            row = json.loads(line)
            pid = str(row["pair_id"])
            p_inst = row["instruction"]
            if pid not in ref_map:
                mismatches.append((pid, "ID NOT FOUND IN EVAL SET", p_inst[:50]))
            elif ref_map[pid] != p_inst:
                mismatches.append((pid, ref_map[pid][:50], p_inst[:50]))

    if mismatches:
        print(f"[FAIL] {fname}: {len(mismatches)}/{line_count} mismatches!")
        for m in mismatches[:5]:
            print(f"   Mismatch in {m[0]}: expected {m[1]!r} vs got {m[2]!r}")
        total_mismatches += len(mismatches)
    else:
        print(f"[PASS] {fname}: {line_count}/{line_count} rows match exact eval set prompt verbatim.")

print(f"\nTOTAL ROWS CHECKED: {total_checked}")
print(f"TOTAL MISMATCHES: {total_mismatches}")
