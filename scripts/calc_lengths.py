import json
import statistics
from pathlib import Path

files = [
    # AlpacaEval 300
    ("AlpacaEval 300", "Qwen2.5-0.5B Base (rp=1.1)", "outputs/generations/alpaca_base_rp11.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B SFT (rp=1.1)", "outputs/generations/alpaca_sft_rp11.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B DPO-UF (rp=1.1)", "outputs/generations/alpaca_dpo_uf_rp11.json"),

    # HH-RLHF 300
    ("HH-RLHF 300", "Qwen2.5-0.5B Base (rp=1.1)", "outputs/generations/hh_base_rp11.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B SFT (rp=1.1)", "outputs/generations/hh_sft_rp11.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B DPO-HH (rp=1.1)", "outputs/generations/hh_dpo_hh_rp11.json"),

    # Persona Style 24
    ("Persona Style (24)", "SmolLM2-360M Base (rp=1.1)", "outputs/lora_persona/persona_base_eval_rp11.json"),
    ("Persona Style (24)", "SmolLM2-360M 1-Ep LoRA (rp=1.1)", "outputs/lora_persona_1ep/persona_lora_1ep_eval_rp11.json"),
    ("Persona Style (24)", "SmolLM2-360M 3-Ep LoRA (rp=1.1)", "outputs/lora_persona/persona_lora_eval_rp11.json"),
]

print(f"{'Eval Set':20s} | {'Model':35s} | {'N':4s} | {'Mean Tokens':11s} | {'Median Tokens':13s} | {'Min':5s} | {'Max':5s}")
print("-" * 105)
for eval_set, model_label, fpath in files:
    data = json.load(open(fpath, encoding="utf-8"))
    lengths = [item["num_generated_tokens"] for item in data]
    mean_len = round(statistics.mean(lengths), 1)
    median_len = round(statistics.median(lengths), 1)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"{eval_set:20s} | {model_label:35s} | {len(lengths):4d} | {mean_len:11.1f} | {median_len:13.1f} | {min_len:5d} | {max_len:5d}")
