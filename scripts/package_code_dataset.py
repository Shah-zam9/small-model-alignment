"""Package repository files and COMMIT.txt into outputs/kaggle/align_code_pkg for Kaggle dataset upload."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


INCLUDE_ITEMS = [
    "src",
    "scripts",
    "configs",
    "eval_sets",
    "data",
    "tests",
    "submission_adapter.py",
    "pyproject.toml",
    "requirements.txt",
    "DEVIATIONS.md",
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pkg_dir = root / "outputs" / "kaggle" / "align_code_pkg"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )
    (pkg_dir / "COMMIT.txt").write_text(commit_hash + "\n", encoding="utf-8")

    for name in INCLUDE_ITEMS:
        src_path = root / name
        if not src_path.exists():
            continue
        dst_path = pkg_dir / name
        if src_path.is_dir():
            shutil.copytree(
                src_path,
                dst_path,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        else:
            shutil.copy2(src_path, dst_path)

    metadata = {
        "title": "align-code",
        "id": "shahzaibali21/align-code",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(pkg_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Packaged align-code at {pkg_dir} with COMMIT={commit_hash}")


if __name__ == "__main__":
    main()
