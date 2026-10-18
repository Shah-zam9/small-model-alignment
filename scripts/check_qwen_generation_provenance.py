import json
from pathlib import Path

qwen_files = [
    # Baseline generation (Task 2)
    ("Task 2 (align-baseline-gen)", "outputs/generations/alpaca_base.json"),
    ("Task 2 (align-baseline-gen)", "outputs/generations/hh_base.json"),
    # SFT generation (align-generate)
    ("align-generate", "outputs/generations/alpaca_sft.json"),
    ("align-generate", "outputs/generations/alpaca_sft_rp11.json"),
    ("align-generate", "outputs/generations/hh_sft.json"),
    ("align-generate", "outputs/generations/hh_sft_rp11.json"),
    # DPO-HH generation (align-generate)
    ("align-generate", "outputs/generations/hh_dpo_hh.json"),
    ("align-generate", "outputs/generations/hh_dpo_hh_rp11.json"),
    # DPO-UF generation (align-generate-uf)
    ("align-generate-uf", "outputs/generations/alpaca_dpo_uf.json"),
    ("align-generate-uf", "outputs/generations/alpaca_dpo_uf_rp11.json"),
    # Renewed Base generation (align-generate-base)
    ("align-generate-base", "outputs/generations/alpaca_base_rp11.json"),
    ("align-generate-base", "outputs/generations/hh_base_rp11.json"),
]

print(f"{'Source Run':22s} | {'File':26s} | {'Environment / Hardware':24s} | {'Device':7s} | {'Dtype':8s} | {'Prompt Template'}")
print("-" * 125)

for src, fp in qwen_files:
    p = Path(fp)
    if not p.exists():
        print(f"{src:22s} | {p.name:26s} | MISSING FILE")
        continue
    data = json.load(open(p, encoding="utf-8"))
    first_item = data[0] if data else {}
    gset = first_item.get("generation_settings", {})
    env = gset.get("environment", "N/A")
    dev = gset.get("device", "N/A")
    dtype = gset.get("dtype", "N/A")
    ptpl = repr(gset.get("prompt_template", "N/A"))[:35]
    print(f"{src:22s} | {p.name:26s} | {env:24s} | {dev:7s} | {dtype:8s} | {ptpl}")
