import json
import glob
from pathlib import Path

eval_sets = {
    "AlpacaEval 300": [
        "outputs/generations/alpaca_base.json",
        "outputs/generations/alpaca_base_rp11.json",
        "outputs/generations/alpaca_sft.json",
        "outputs/generations/alpaca_sft_rp11.json",
        "outputs/generations/alpaca_dpo_uf.json",
        "outputs/generations/alpaca_dpo_uf_rp11.json",
    ],
    "HH-RLHF 300": [
        "outputs/generations/hh_base.json",
        "outputs/generations/hh_base_rp11.json",
        "outputs/generations/hh_sft.json",
        "outputs/generations/hh_sft_rp11.json",
        "outputs/generations/hh_dpo_hh.json",
        "outputs/generations/hh_dpo_hh_rp11.json",
    ],
    "Persona 24": [
        "outputs/lora_persona/persona_base_eval.json",
        "outputs/lora_persona/persona_base_eval_rp11.json",
        "outputs/lora_persona_1ep/persona_lora_1ep_eval.json",
        "outputs/lora_persona_1ep/persona_lora_1ep_eval_rp11.json",
        "outputs/lora_persona/persona_lora_eval.json",
        "outputs/lora_persona/persona_lora_eval_rp11.json",
    ],
}

print("=== EXACT-MATCH OUTPUT OVERLAP MATRICES ===")
for set_name, file_list in eval_sets.items():
    print(f"\n==================== {set_name} ====================")
    data = {}
    names = []
    for f in file_list:
        p = Path(f)
        if not p.exists():
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
            data[p.name] = {row["id"]: row["output"] for row in d}
            names.append(p.name)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Header
    print(f"{'File':30s} | " + " | ".join(f"{n[:12]:12s}" for n in names))
    print("-" * (32 + 15 * len(names)))

    for n1 in names:
        row_str = f"{n1:30s} | "
        for n2 in names:
            d1 = data[n1]
            d2 = data[n2]
            common_ids = set(d1.keys()) & set(d2.keys())
            exact_matches = sum(1 for pid in common_ids if d1[pid] == d2[pid])
            row_str += f"{exact_matches:12d} | "
        print(row_str)
