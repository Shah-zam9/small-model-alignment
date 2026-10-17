"""Response-only sequence log-probability computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.align.batching import build_sft_batch


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the model's first parameter or buffer (fallback CPU)."""
    for param in model.parameters():
        return param.device
    for buf in model.buffers():
        return buf.device
    return torch.device("cpu")


def compute_response_logprobs_from_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute summed response-token log-probabilities per row from a prebuilt batch.

    Math:
        log pi(y | x) = sum_{t in response} log p(y_t | x, y_{<t})
    Uses causal next-token shift (logits[:, :-1] predicts labels[:, 1:]) and
    F.cross_entropy(..., ignore_index=-100, reduction="none") in fp32 so that
    masked positions (-100) contribute exactly 0 without gathering on -100.
    """
    device = get_model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    if logits.shape[1] <= 1:
        return torch.zeros(logits.shape[0], dtype=torch.float32, device=device)

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    batch_size, seq_len_minus_1, vocab_size = shift_logits.shape

    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(batch_size, seq_len_minus_1)

    return -token_nll.sum(dim=1)


def compute_response_logprobs(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int,
) -> torch.Tensor:
    """Tokenize prompts/responses and return one summed response log-prob per row (shape [B])."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length=max_length)
    return compute_response_logprobs_from_batch(model, batch)
