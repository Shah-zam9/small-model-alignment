import json

for fname in [
    'outputs/judgments/judge_persona_base_vs_lora_3ep_rp11.jsonl',
    'outputs/judgments/judge_persona_base_vs_lora_1ep_rp11.jsonl',
    'outputs/judgments/judge_persona_lora_1ep_vs_3ep_rp11.jsonl'
]:
    print(f"=== {fname} ===")
    missing_style = 0
    missing_useful = 0
    missing_safety = 0
    with open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            parsed = data.get('parsed_judgment', {})
            raw_sw = parsed.get('style_winner')
            raw_uw = parsed.get('usefulness_winner')
            raw_safew = parsed.get('safety_winner')
            
            if raw_sw is None:
                missing_style += 1
            if raw_uw is None:
                missing_useful += 1
            if raw_safew is None:
                missing_safety += 1
                
    print(f"Missing/None: style={missing_style}, usefulness={missing_useful}, safety={missing_safety}")
