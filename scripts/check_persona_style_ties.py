import json

for fname in [
    'outputs/judgments/judge_persona_base_vs_lora_3ep_rp11.jsonl',
    'outputs/judgments/judge_persona_base_vs_lora_1ep_rp11.jsonl',
    'outputs/judgments/judge_persona_lora_1ep_vs_3ep_rp11.jsonl'
]:
    with open(fname, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    print(f"=== {fname} ===")
    unknown_style = sum(1 for r in rows if r.get('style_winner') == 'unknown')
    unknown_useful = sum(1 for r in rows if r.get('usefulness_winner') == 'unknown')
    unknown_safe = sum(1 for r in rows if r.get('safety_winner') == 'unknown')
    style_ties = [r for r in rows if r.get('style_winner') == 'tie']
    
    print(f"Total rows: {len(rows)}")
    print(f"Unknowns: style={unknown_style}, useful={unknown_useful}, safety={unknown_safe}")
    print(f"Style ties count: {len(style_ties)}")
    print("5 style-tie rows with reason:")
    for i, r in enumerate(style_ties[:5], 1):
        pid = r.get("pair_id")
        w = r.get("winner")
        sw = r.get("style_winner")
        uw = r.get("usefulness_winner")
        sfw = r.get("safety_winner")
        reason = r.get("reason")
        print(f"  [{i}] Pair: {pid} | Overall: {w} | Style: {sw} | Useful: {uw} | Safety: {sfw}")
        print(f"      Reason: {reason}\n")
