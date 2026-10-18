#!/usr/bin/env python3
"""Package the final assignment submission archive (submission_27100328.zip).

Includes:
- report_27100328.pdf (compiled 8-page final report)
- submission_adapter.py (verified adapter passing 21/21 public tests)
- README.md (with repo URL and full inventory)
- configs/, scripts/, tests/, eval_sets/, and essential outputs
Excludes:
- Heavy model weights, checkpoints, large raw logs, and hidden git/cache folders.
"""

import os
import zipfile
from pathlib import Path

def package():
    zip_name = "submission_27100328.zip"
    exclude_dirs = {".git", ".gemini", "__pycache__", ".pytest_cache", "venv", "env", "wandb", "checkpoints", "best"}
    exclude_exts = {".bin", ".pt", ".safetensors", ".zip"}

    # Ensure report/main.pdf exists
    if not Path("report/main.pdf").exists():
        raise FileNotFoundError("report/main.pdf not found. Please compile it first.")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        # Add report copy at root
        z.write("report/main.pdf", arcname="report_27100328.pdf")
        
        # Traverse workspace
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
            for file in files:
                p = Path(root) / file
                if p.suffix in exclude_exts or file.startswith("."):
                    continue
                # Skip heavy kaggle subfolder logs or huge files > 10MB
                if "outputs/kaggle" in root or p.stat().st_size > 10_000_000:
                    continue
                rel = p.relative_to(".")
                if str(rel) in {zip_name, "report_overleaf.zip"}:
                    continue
                z.write(p, arcname=str(rel))

    zip_path = Path(zip_name)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {zip_name} ({size_mb:.2f} MB).")

if __name__ == "__main__":
    package()
