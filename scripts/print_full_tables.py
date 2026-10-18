import json
import statistics
from pathlib import Path

files = [
    # AlpacaEval 300
    ("AlpacaEval 300", "Qwen2.5-0.5B Base (greedy rp=1.0)", "outputs/generations/alpaca_base.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B Base (rp=1.1)", "outputs/generations/alpaca_base_rp11.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B SFT (greedy rp=1.0)", "outputs/generations/alpaca_sft.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B SFT (rp=1.1)", "outputs/generations/alpaca_sft_rp11.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B DPO-UF (greedy rp=1.0)", "outputs/generations/alpaca_dpo_uf.json"),
    ("AlpacaEval 300", "Qwen2.5-0.5B DPO-UF (rp=1.1)", "outputs/generations/alpaca_dpo_uf_rp11.json"),

    # HH-RLHF 300
    ("HH-RLHF 300", "Qwen2.5-0.5B Base (greedy rp=1.0)", "outputs/generations/hh_base.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B Base (rp=1.1)", "outputs/generations/hh_base_rp11.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B SFT (greedy rp=1.0)", "outputs/generations/hh_sft.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B SFT (rp=1.1)", "outputs/generations/hh_sft_rp11.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B DPO-HH (greedy rp=1.0)", "outputs/generations/hh_dpo_hh.json"),
    ("HH-RLHF 300", "Qwen2.5-0.5B DPO-HH (rp=1.1)", "outputs/generations/hh_dpo_hh_rp11.json"),

    # Persona Style 24
    ("Persona Style (24)", "SmolLM2-360M Base (greedy rp=1.0)", "outputs/lora_persona/persona_base_eval.json"),
    ("Persona Style (24)", "SmolLM2-360M Base (rp=1.1)", "outputs/lora_persona/persona_base_eval_rp11.json"),
    ("Persona Style (24)", "SmolLM2-360M 1-Ep LoRA (greedy rp=1.0)", "outputs/lora_persona_1ep/persona_lora_1ep_eval.json"),
    ("Persona Style (24)", "SmolLM2-360M 1-Ep LoRA (rp=1.1)", "outputs/lora_persona_1ep/persona_lora_1ep_eval_rp11.json"),
    ("Persona Style (24)", "SmolLM2-360M 3-Ep LoRA (greedy rp=1.0)", "outputs/lora_persona/persona_lora_eval.json"),
    ("Persona Style (24)", "SmolLM2-360M 3-Ep LoRA (rp=1.1)", "outputs/lora_persona/persona_lora_eval_rp11.json"),
]

header = f"{'Eval Set':18s} | {'Model':35s} | {'N':4s} | {'Mean':6s} | {'Median':6s} | {'Min':4s} | {'Max':4s} | {'Hit 512':7s} | {'Loop Rate':9s}"
print(header)
print("-" * len(header))
for eval_set, model_label, fpath in files:
    data = json.load(open(fpath, encoding="utf-8"))
    lengths = [item["num_generated_tokens"] for item in data]
    hits = sum(1 for item in data if item.get("hit_max_new_tokens", item["num_generated_tokens"] >= 512))
    loop_rate = round(hits / len(data) * 100, 2)
    mean_len = round(statistics.mean(lengths), 1)
    median_len = round(statistics.median(lengths), 1)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"{eval_set:18s} | {model_label:35s} | {len(lengths):4d} | {mean_len:6.1f} | {median_len:6.1f} | {min_len:4d} | {max_len:4d} | {hits:7d} | {loop_rate:8.2f}%")
