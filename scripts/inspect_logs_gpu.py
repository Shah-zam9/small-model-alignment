import json
from pathlib import Path

logs = [
    ("align-baseline-gen", "outputs/kaggle/align-baseline-gen/align-baseline-gen.log"),
    ("align-generate", "outputs/kaggle/align-generate/align-generate.log"),
    ("align-generate-uf", "outputs/kaggle/align-generate-uf/align-generate-uf.log"),
    ("align-generate-base", "outputs/kaggle/align-generate-base/align-generate-base.log"),
]

for name, log_path in logs:
    p = Path(log_path)
    if not p.exists():
        print(f"Missing {log_path}")
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    gpu_lines = [l for l in text.splitlines() if any(k in l for k in ["Tesla", "T4", "cuda:0", "torch_dtype", "float16", "NVIDIA"])]
    print(f"=== {name} ===")
    for gl in gpu_lines[:5]:
        print("  ", gl.strip())
