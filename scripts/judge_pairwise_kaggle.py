"""Resumable batch pairwise evaluator using Kaggle Benchmarks / Model Proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repo root is on sys.path when invoked as `python scripts/judge_pairwise_kaggle.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.align.judge import (
    build_batch_judge_prompt,
    call_kaggle_judge,
    detect_rubric,
    map_ab_to_model,
    parse_judge_json,
    summarize_judgments,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_pairwise_judging(
    pairs_path: str | Path,
    output_path: str | Path,
    judge_model: str = "google/gemini-3.6-flash",
    batch_size: int = 5,
    sleep_seconds: float = 5.0,
    max_retries: int = 3,
    reuse_judgments: bool = False,
    rubric: str = "auto",
    limit: int | None = None,
    show_prompts: bool = False,
) -> dict[str, Any]:
    pairs_file = Path(pairs_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = out_file.with_name(out_file.stem + "_failures.jsonl")
    summary_file = out_file.with_name(out_file.stem + "_summary.json")

    all_pairs = load_jsonl(pairs_file)
    if limit is not None:
        all_pairs = all_pairs[:limit]

    rubric_name = detect_rubric(all_pairs, pairs_file) if rubric == "auto" else rubric

    existing_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if reuse_judgments and out_file.exists():
        existing_rows = load_jsonl(out_file)
        for r in existing_rows:
            if r.get("winner") in ("A", "B", "tie"):
                completed_ids.add(str(r["pair_id"]))
        print(f"Loaded {len(completed_ids)} completed judgments from {out_file}")
    elif not reuse_judgments and out_file.exists():
        out_file.unlink()

    pending_pairs = [p for p in all_pairs if str(p["pair_id"]) not in completed_ids]
    print(
        f"Judging {len(pending_pairs)} pending pairs (out of {len(all_pairs)}) "
        f"with model={judge_model!r}, rubric={rubric_name!r}, batch_size={batch_size}"
    )

    judged_map: dict[str, dict[str, Any]] = {str(r["pair_id"]): r for r in existing_rows}
    total_retries = 0
    failures_count = 0

    for start in range(0, len(pending_pairs), batch_size):
        batch = pending_pairs[start : start + batch_size]
        expected_ids = [str(p["pair_id"]) for p in batch]
        prompt_text = build_batch_judge_prompt(batch, rubric_name=rubric_name)

        if show_prompts:
            print(f"\n================ RENDERED JUDGE PROMPT (Batch {start // batch_size + 1}) ================")
            print(prompt_text)
            print("=========================================================================\n")

        parsed_batch: dict[str, dict[str, str]] = {}
        raw_response = ""
        last_error = ""

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                total_retries += 1
            try:
                raw_response = call_kaggle_judge(prompt_text, judge_model=judge_model)
                parsed_batch = parse_judge_json(
                    raw_response,
                    expected_pair_ids=expected_ids,
                    rubric_name=rubric_name,
                )
                if len(parsed_batch) == len(batch) and all(
                    parsed_batch[pid]["winner"] in ("A", "B", "tie") for pid in expected_ids
                ):
                    break
                last_error = f"Incomplete or unparseable JSON batch (got {len(parsed_batch)}/{len(batch)} valid items)"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            print(f"[Batch {start // batch_size + 1}] Attempt {attempt}/{max_retries} failed: {last_error}")
            time.sleep(max(5.0, float(sleep_seconds)))

        # Record batch results and map A/B back to model identities AFTER judging
        with open(out_file, "a", encoding="utf-8") as fout:
            for pair in batch:
                pid = str(pair["pair_id"])
                meta = pair.get("meta", {})
                a_model = str(meta.get("a_model", "model_a"))
                b_model = str(meta.get("b_model", "model_b"))

                j_item = parsed_batch.get(pid)
                if j_item is None or j_item.get("winner") not in ("A", "B", "tie"):
                    winner_ab = "unknown"
                    reason = j_item.get("reason", f"Unresolved after {max_retries} attempts: {last_error}") if j_item else f"Unresolved: {last_error}"
                    failures_count += 1
                    with open(failures_file, "a", encoding="utf-8") as ffail:
                        ffail.write(
                            json.dumps(
                                {
                                    "pair_id": pid,
                                    "error": last_error,
                                    "raw_response": raw_response[:1000],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                else:
                    winner_ab = j_item["winner"]
                    reason = j_item.get("reason", "")

                winner_model = map_ab_to_model(winner_ab, a_model, b_model)
                out_row: dict[str, Any] = {
                    "pair_id": pid,
                    "instruction": pair["instruction"],
                    "dataset": pair.get("dataset", ""),
                    "winner": winner_ab,
                    "winner_model": winner_model,
                    "reason": reason,
                    "judge_model": judge_model,
                    "rubric": rubric_name,
                    "meta": meta,
                }
                if "category" in pair:
                    out_row["category"] = pair["category"]
                if rubric_name == "persona":
                    sw_ab = j_item.get("style_winner", "unknown") if j_item else "unknown"
                    uw_ab = j_item.get("usefulness_winner", "unknown") if j_item else "unknown"
                    sfw_ab = j_item.get("safety_winner", "unknown") if j_item else "unknown"
                    out_row["style_winner"] = sw_ab
                    out_row["style_winner_model"] = map_ab_to_model(sw_ab, a_model, b_model)
                    out_row["usefulness_winner"] = uw_ab
                    out_row["usefulness_winner_model"] = map_ab_to_model(uw_ab, a_model, b_model)
                    out_row["safety_winner"] = sfw_ab
                    out_row["safety_winner_model"] = map_ab_to_model(sfw_ab, a_model, b_model)

                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                judged_map[pid] = out_row

        if start + batch_size < len(pending_pairs):
            time.sleep(max(5.0, float(sleep_seconds)))

    ordered_judged = [judged_map[str(p["pair_id"])] for p in all_pairs if str(p["pair_id"]) in judged_map]
    summary = summarize_judgments(
        judged_rows=ordered_judged,
        failures_count=failures_count,
        total_retries=total_retries,
        judge_model=judge_model,
        rubric_name=rubric_name,
    )
    with open(summary_file, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("JUDGE SUMMARY:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge pairwise A/B responses using Kaggle Benchmarks.")
    parser.add_argument("--pairs_path", type=str, required=True, help="Input pairwise JSONL path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output judged JSONL path.")
    parser.add_argument("--judge_model", type=str, default="google/gemini-3.6-flash", help="Judge model slug.")
    parser.add_argument("--batch_size", type=int, default=5, help="Pairs per judge request.")
    parser.add_argument("--sleep_seconds", type=float, default=5.0, help="Sleep seconds between requests.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per batch.")
    parser.add_argument("--reuse_judgments", action="store_true", help="Resume from existing judged JSONL.")
    parser.add_argument("--rubric", type=str, default="auto", choices=["auto", "general", "safety", "persona"])
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs.")
    parser.add_argument("--show_prompts", action="store_true", help="Print rendered judge prompts.")
    args = parser.parse_args()

    run_pairwise_judging(
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        reuse_judgments=args.reuse_judgments,
        rubric=args.rubric,
        limit=args.limit,
        show_prompts=args.show_prompts,
    )


if __name__ == "__main__":
    main()
