import json
from pathlib import Path
import re
from src.align.judge import (
    build_batch_judge_prompt,
    call_kaggle_judge,
    parse_judge_json,
    map_ab_to_model,
)

def load_first_pair(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"Empty {path}")

# 1. HH-RLHF (Safety rubric)
hh_pair = load_first_pair("outputs/pairwise/pairs_hh_sft_vs_dpo_hh_rp11.jsonl")
prompt_hh = build_batch_judge_prompt([hh_pair], rubric_name="safety")

# 2. AlpacaEval (General rubric)
alpaca_pair = load_first_pair("outputs/pairwise/pairs_alpaca_sft_vs_dpo_uf_rp11.jsonl")
prompt_alpaca = build_batch_judge_prompt([alpaca_pair], rubric_name="general")

# 3. Persona (Persona rubric: style, usefulness, safety verdicts)
persona_pair = load_first_pair("outputs/pairwise/pairs_persona_base_vs_lora_3ep_rp11.jsonl")
prompt_persona = build_batch_judge_prompt([persona_pair], rubric_name="persona")

tests = [
    ("HH-RLHF (Safety Rubric)", prompt_hh, [hh_pair], "safety"),
    ("AlpacaEval (General Rubric)", prompt_alpaca, [alpaca_pair], "general"),
    ("Persona (3 Verdicts Rubric)", prompt_persona, [persona_pair], "persona"),
]

forbidden_terms = ["qwen", "smollm", "dpo-hh", "dpo-uf", "sft", "lora", "outputs/"]

print("=== CHECKING PROMPTS FOR LEAKAGE ===")
for name, prompt, batch, rubric in tests:
    print(f"\n--- Checking {name} ---")
    leaks = [term for term in forbidden_terms if term in prompt.lower()]
    print(f"Leak check result: {'CLEAN (No model/file names leaked)' if not leaks else f'FAILED: found {leaks}'}")

print("\n=== RUNNING 3 TEST JUDGE CALLS ===")
for name, prompt, batch, rubric in tests:
    pair = batch[0]
    pid = str(pair["pair_id"])
    meta = pair["meta"]
    a_model = meta["a_model"]
    b_model = meta["b_model"]

    print(f"\n====================================================================")
    print(f"TEST CALL: {name}")
    print(f"PAIR ID: {pid}")
    print(f"A/B ASSIGNMENT (HIDDEN BEFORE PARSING): A={a_model} | B={b_model}")
    print(f"---------------------- RENDERED JUDGE PROMPT ----------------------")
    print(prompt)
    print(f"---------------------- CALLING JUDGE LLM ---------------------------")
    raw_resp = call_kaggle_judge(prompt, judge_model="google/gemini-3-flash-preview")
    print(f"RAW JUDGE RESPONSE:\n{raw_resp}")
    print(f"---------------------- PARSING RESULTS -----------------------------")
    parsed = parse_judge_json(raw_resp, expected_pair_ids=[pid], rubric_name=rubric)
    print(f"PARSED JSON: {json.dumps(parsed, indent=2)}")
    
    # Verify A/B -> model mapping happens ONLY after parsing:
    j_item = parsed.get(pid, {})
    winner_ab = j_item.get("winner", "unknown")
    winner_model = map_ab_to_model(winner_ab, a_model, b_model)
    print(f"FINAL MAPPED VERDICT: Winner={winner_ab} -> Model={winner_model}")
    if rubric == "persona":
        sw = j_item.get("style_winner", "unknown")
        uw = j_item.get("usefulness_winner", "unknown")
        sfw = j_item.get("safety_winner", "unknown")
        print(f"  Style Winner:      {sw} -> {map_ab_to_model(sw, a_model, b_model)}")
        print(f"  Usefulness Winner: {uw} -> {map_ab_to_model(uw, a_model, b_model)}")
        print(f"  Safety Winner:     {sfw} -> {map_ab_to_model(sfw, a_model, b_model)}")
