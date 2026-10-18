import os
from pathlib import Path
import pytest
from src.align.generate import resolve_model_path


def test_hf_hub_id_not_hijacked_by_kaggle_input(tmp_path, monkeypatch):
    """Confirm HF Hub IDs are returned as-is and never search /kaggle/input."""
    # Create fake /kaggle/input with a rogue best/config.json
    fake_kaggle_in = tmp_path / "kaggle" / "input"
    rogue_best = fake_kaggle_in / "some_kernel" / "outputs" / "dpo" / "best"
    rogue_best.mkdir(parents=True)
    (rogue_best / "config.json").write_text("{}", encoding="utf-8")

    # Monkeypatch Path("/kaggle/input") to fake_kaggle_in
    orig_path_init = Path
    monkeypatch.setattr(
        "src.align.generate.Path",
        lambda *args: fake_kaggle_in if args == ("/kaggle/input",) else orig_path_init(*args),
    )

    # 1. Qwen hub ID
    resolved_qwen = resolve_model_path("Qwen/Qwen2.5-0.5B")
    assert resolved_qwen == "Qwen/Qwen2.5-0.5B"

    # 2. SmolLM2 hub ID
    resolved_smollm = resolve_model_path("HuggingFaceTB/SmolLM2-360M-Instruct")
    assert resolved_smollm == "HuggingFaceTB/SmolLM2-360M-Instruct"


def test_resolve_local_existing_path(tmp_path):
    """Local existing path resolves directly."""
    local_dir = tmp_path / "my_model"
    local_dir.mkdir()
    res = resolve_model_path(str(local_dir))
    assert res == str(local_dir.resolve())


def test_resolve_unique_match_in_kaggle_input(tmp_path, monkeypatch):
    """Path starting with outputs/ resolves if exactly one match exists under /kaggle/input."""
    fake_kaggle_in = tmp_path / "kaggle" / "input"
    sft_dir = fake_kaggle_in / "dataset_kernel" / "outputs" / "sft" / "best"
    sft_dir.mkdir(parents=True)
    (sft_dir / "config.json").write_text("{}", encoding="utf-8")

    orig_path_init = Path
    monkeypatch.setattr(
        "src.align.generate.Path",
        lambda *args: fake_kaggle_in if args == ("/kaggle/input",) else orig_path_init(*args),
    )

    resolved = resolve_model_path("outputs/sft/best")
    assert resolved == str(sft_dir.resolve())


def test_generic_best_fallback_deleted_raises_filenotfound(tmp_path, monkeypatch):
    """Generic best/config.json must NOT match when looking for a different checkpoint."""
    fake_kaggle_in = tmp_path / "kaggle" / "input"
    dpo_dir = fake_kaggle_in / "kernel_1" / "outputs" / "dpo_hh" / "best"
    dpo_dir.mkdir(parents=True)
    (dpo_dir / "config.json").write_text("{}", encoding="utf-8")

    orig_path_init = Path
    monkeypatch.setattr(
        "src.align.generate.Path",
        lambda *args: fake_kaggle_in if args == ("/kaggle/input",) else orig_path_init(*args),
    )

    # Looking for outputs/sft/best when only outputs/dpo_hh/best exists must fail
    with pytest.raises(FileNotFoundError, match="no match found under /kaggle/input"):
        resolve_model_path("outputs/sft/best")


def test_ambiguous_matches_raise_value_error(tmp_path, monkeypatch):
    """Multiple matches for the same outputs path under /kaggle/input must raise ValueError."""
    fake_kaggle_in = tmp_path / "kaggle" / "input"
    match1 = fake_kaggle_in / "mount_1" / "outputs" / "sft" / "best"
    match2 = fake_kaggle_in / "mount_2" / "outputs" / "sft" / "best"
    match1.mkdir(parents=True)
    match2.mkdir(parents=True)
    (match1 / "config.json").write_text("{}", encoding="utf-8")
    (match2 / "config.json").write_text("{}", encoding="utf-8")

    orig_path_init = Path
    monkeypatch.setattr(
        "src.align.generate.Path",
        lambda *args: fake_kaggle_in if args == ("/kaggle/input",) else orig_path_init(*args),
    )

    with pytest.raises(ValueError, match="Ambiguous model path resolution"):
        resolve_model_path("outputs/sft/best")
