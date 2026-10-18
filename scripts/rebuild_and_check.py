import json
import subprocess
from pathlib import Path

def run():
    print("=== 1. Checking align-generate-base kernel status ===")
    res = subprocess.run(["kaggle", "kernels", "status", "shahzaibali21/align-generate-base"], capture_output=True, text=True)
    print(res.stdout.strip())
    if "COMPLETE" not in res.stdout:
        print("Kernel not yet complete.")
        return False

    print("=== 2. Downloading kernel outputs ===")
    out_dir = Path("outputs/kaggle/align-generate-base")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", "shahzaibali21/align-generate-base", "-p", str(out_dir)], check=True)

    # Check downloaded files
    print("Downloaded files in outputs/kaggle/align-generate-base:")
    for f in out_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size} bytes)")

    # Copy alpaca_base_rp11.json and hh_base_rp11.json to outputs/generations/
    src_alpaca = out_dir / "outputs" / "generations" / "alpaca_base_rp11.json"
    if not src_alpaca.exists():
        src_alpaca = out_dir / "alpaca_base_rp11.json"
    src_hh = out_dir / "outputs" / "generations" / "hh_base_rp11.json"
    if not src_hh.exists():
        src_hh = out_dir / "hh_base_rp11.json"

    import shutil
    shutil.copy(src_alpaca, "outputs/generations/alpaca_base_rp11.json")
    shutil.copy(src_hh, "outputs/generations/hh_base_rp11.json")
    print("Copied renewed base rp11 files to outputs/generations/")

    # 3. Rebuild pairwise files
    print("=== 3. Rebuilding affected pairwise files ===")
    from src.align.pairs import write_pairwise_jsonl
    p1 = write_pairwise_jsonl(
        "outputs/generations/alpaca_base_rp11.json",
        "outputs/generations/alpaca_sft_rp11.json",
        "outputs/pairwise/pairs_alpaca_base_vs_sft_rp11.jsonl",
        seed=42,
        dataset_override="alpaca_eval",
    )
    print(f"Wrote {len(p1)} pairs to outputs/pairwise/pairs_alpaca_base_vs_sft_rp11.jsonl")

    p2 = write_pairwise_jsonl(
        "outputs/generations/hh_base_rp11.json",
        "outputs/generations/hh_sft_rp11.json",
        "outputs/pairwise/pairs_hh_base_vs_sft_rp11.jsonl",
        seed=42,
        dataset_override="hh_rlhf",
    )
    print(f"Wrote {len(p2)} pairs to outputs/pairwise/pairs_hh_base_vs_sft_rp11.jsonl")

    return True

if __name__ == "__main__":
    run()
