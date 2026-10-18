import json
import re
from pathlib import Path

print("=== CHECKING ADAPTER CONFIGS ===")
for p in [
    "outputs/lora_persona/best_adapter/adapter_config.json",
    "outputs/lora_persona_1ep/best_adapter/adapter_config.json",
]:
    data = json.load(open(p, encoding="utf-8"))
    print(f"{p}: base_model_name_or_path = {data.get('base_model_name_or_path')!r}")

print("\n=== RESOLVED MODEL PATHS IN LOGS ===")
logs = {
    "align-dpo-hh (TRAINING)": "outputs/kaggle/align-dpo-hh/align-dpo-hh.log",
    "align-dpo-uf (TRAINING)": "outputs/kaggle/align-dpo-uf/align-dpo-uf.log",
    "align-generate (GENERATION)": "outputs/kaggle/align-generate/align-generate.log",
    "align-generate-uf (GENERATION)": "outputs/kaggle/align-generate-uf/align-generate-uf.log",
}

for label, log_path in logs.items():
    print(f"\n--- {label} ({log_path}) ---")
    p = Path(log_path)
    if not p.exists():
        print(f"Log not found: {log_path}")
        continue
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Find all occurrences of Resolved or Using model path
    matches = re.findall(r"(?:Resolved model path.*?|Using model path.*?|Loading trainable policy model.*?|Loading frozen reference model.*?)[\r\n]", text)
    for m in matches:
        clean = m.strip().replace('"', '').replace('\\n', '').replace('\\r', '')
        if clean:
            print("  " + clean)
