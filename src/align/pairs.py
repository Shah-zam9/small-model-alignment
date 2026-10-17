"""Pairwise comparison builder with deterministic A/B randomization and hidden metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


def build_pairwise_records(
    gen_x: list[dict[str, Any]],
    gen_y: list[dict[str, Any]],
    seed: int = 42,
    dataset_override: str | None = None,
) -> list[dict[str, Any]]:
    """Build randomized A/B comparison pairs from two aligned model generation lists.

    Each row has:
        {
            "pair_id": str,
            "instruction": str,
            "response_a": str,
            "response_b": str,
            "dataset": str,
            "category": str | None,
            "meta": {
                "a_model": str,
                "b_model": str,
                "prompt_id": str,
                "seed": int,
            }
        }
    """
    if len(gen_x) != len(gen_y):
        raise ValueError(
            f"Generation lists must have equal length, got {len(gen_x)} vs {len(gen_y)}."
        )

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    for idx, (rx, ry) in enumerate(zip(gen_x, gen_y)):
        id_x = str(rx.get("id", f"prompt_{idx:04d}"))
        id_y = str(ry.get("id", f"prompt_{idx:04d}"))
        if id_x != id_y:
            raise ValueError(f"Mismatched prompt IDs at index {idx}: {id_x!r} vs {id_y!r}")

        inst_x = rx["instruction"]
        inst_y = ry["instruction"]
        if inst_x != inst_y:
            raise ValueError(f"Mismatched instructions for ID {id_x!r}")

        model_x = str(rx.get("generator", "model_x"))
        model_y = str(ry.get("generator", "model_y"))
        if model_x == model_y:
            raise ValueError(
                f"Both generation files have identical generator name {model_x!r}."
            )

        out_x = str(rx.get("output", ""))
        out_y = str(ry.get("output", ""))
        dataset_name = dataset_override or str(rx.get("dataset", "eval"))
        category = rx.get("category") or ry.get("category")

        # Flip a deterministic fair coin for A/B order
        if rng.random() < 0.5:
            resp_a, resp_b = out_x, out_y
            a_model, b_model = model_x, model_y
        else:
            resp_a, resp_b = out_y, out_x
            a_model, b_model = model_y, model_x

        pair_record: dict[str, Any] = {
            "pair_id": id_x,
            "instruction": inst_x,
            "response_a": resp_a,
            "response_b": resp_b,
            "dataset": dataset_name,
            "meta": {
                "a_model": a_model,
                "b_model": b_model,
                "prompt_id": id_x,
                "seed": seed,
            },
        }
        if category is not None:
            pair_record["category"] = category
        pairs.append(pair_record)

    return pairs


def write_pairwise_jsonl(
    model_x_path: str | Path,
    model_y_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    dataset_override: str | None = None,
) -> list[dict[str, Any]]:
    """Load two generation JSON files, build randomized A/B pairs, and save as JSONL."""
    with open(model_x_path, "r", encoding="utf-8") as fx:
        gen_x = json.load(fx)
    with open(model_y_path, "r", encoding="utf-8") as fy:
        gen_y = json.load(fy)

    pairs = build_pairwise_records(
        gen_x=gen_x,
        gen_y=gen_y,
        seed=seed,
        dataset_override=dataset_override,
    )
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fout:
        for row in pairs:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build randomized A/B pairwise JSONL files.")
    parser.add_argument("--model_x_path", type=str, required=True, help="First model generation JSON.")
    parser.add_argument("--model_y_path", type=str, required=True, help="Second model generation JSON.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for A/B assignment.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name override.")
    args = parser.parse_args()

    pairs = write_pairwise_jsonl(
        model_x_path=args.model_x_path,
        model_y_path=args.model_y_path,
        output_path=args.output_path,
        seed=args.seed,
        dataset_override=args.dataset,
    )
    print(f"Wrote {len(pairs)} randomized pairs to {args.output_path}")


if __name__ == "__main__":
    main()
