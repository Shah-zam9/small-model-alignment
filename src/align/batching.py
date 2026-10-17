"""SFT and DPO batch construction with response-only label masking."""

from __future__ import annotations

import torch


def _resolve_pad_token_id(tokenizer) -> int:
    """Resolve padding token ID with fallback chain: pad_token_id -> eos_token_id -> 0."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def build_sft_batch(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build right-padded input_ids, attention_mask, and response-only labels.

    Math / Masking rules:
    - Prompt is tokenized with default special tokens; response with add_special_tokens=False.
    - Note: max_prompt_length is kept None in adapter and training (long DPO prompts are
      filtered out at dataset construction time so the '### Response:' suffix is never cut).
    - If tokenizer.eos_token_id is not None, append it to response_ids BEFORE truncation.
    - Concatenate ids = prompt_ids + response_ids, then truncate from the right to max_length.
    - Position-based masking: labels are -100 on prompt positions [:len(prompt_ids)] and
      on right-padding positions [len(full_ids):], and equal to input_ids on response
      positions (including EOS even when pad_token_id == eos_token_id).
    """
    if len(prompts) != len(responses):
        raise ValueError(f"prompts ({len(prompts)}) and responses ({len(responses)}) must have equal length")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = _resolve_pad_token_id(tokenizer)

    rows_ids: list[list[int]] = []
    rows_labels: list[list[int]] = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = list(tokenizer.encode(prompt))
        response_ids = list(tokenizer.encode(response, add_special_tokens=False))
        if eos_token_id is not None:
            response_ids.append(int(eos_token_id))

        full_ids = (prompt_ids + response_ids)[:max_length]
        full_labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]

        rows_ids.append(full_ids)
        rows_labels.append(full_labels)

    batch_width = max((len(ids) for ids in rows_ids), default=0)

    padded_input_ids: list[list[int]] = []
    padded_attention_mask: list[list[int]] = []
    padded_labels: list[list[int]] = []

    for ids, labels in zip(rows_ids, rows_labels):
        pad_len = batch_width - len(ids)
        padded_input_ids.append(ids + [pad_token_id] * pad_len)
        padded_attention_mask.append([1] * len(ids) + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def build_dpo_batch(
    tokenizer,
    prompts: list[str],
    chosen_responses: list[str],
    rejected_responses: list[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build a dictionary with 'chosen' and 'rejected' SFT-formatted LM batches."""
    return {
        "chosen": build_sft_batch(
            tokenizer,
            prompts,
            chosen_responses,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        ),
        "rejected": build_sft_batch(
            tokenizer,
            prompts,
            rejected_responses,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        ),
    }
