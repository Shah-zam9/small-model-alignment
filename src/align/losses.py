"""From-scratch SFT and DPO loss functions and preference reward metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.align.batching import build_dpo_batch, build_sft_batch
from src.align.logprobs import compute_response_logprobs_from_batch, get_model_device


def compute_sft_loss_from_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute token-level mean negative log-likelihood over unmasked response tokens.

    Math:
        L_SFT = ( -sum_{i, t in response} log p_theta(y_{i,t} | x_i, y_{i,<t}) )
                / max(1, count(labels[:, 1:] != -100))
    """
    device = get_model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    if logits.shape[1] <= 1:
        return logits.sum() * 0.0

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.shape[-1]

    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )

    response_token_count = shift_labels.ne(-100).sum().clamp_min(1)
    return token_nll.sum() / response_token_count


def compute_sft_loss(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int,
) -> torch.Tensor:
    """Build SFT batch and return scalar token-mean response-only SFT loss."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length=max_length)
    return compute_sft_loss_from_batch(model, batch)


def compute_dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute scalar batch-mean DPO loss from per-example response log-probabilities.

    Math:
        Delta_theta = log pi_theta(y_w | x) - log pi_theta(y_l | x)
        Delta_ref   = log pi_ref(y_w | x)   - log pi_ref(y_l | x)
        L_DPO       = -mean( log sigma( beta * (Delta_theta - Delta_ref) ) )
    """
    pi_c = policy_chosen_logps.float()
    pi_r = policy_rejected_logps.float()
    ref_c = reference_chosen_logps.detach().to(pi_c.device).float()
    ref_r = reference_rejected_logps.detach().to(pi_c.device).float()

    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    return -F.logsigmoid(logits).mean()


def compute_dpo_metrics_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Compute DPO implicit rewards, strict reward accuracy (r_w > r_l), and reward margin."""
    pi_c = policy_chosen_logps.detach().float()
    pi_r = policy_rejected_logps.detach().float()
    ref_c = reference_chosen_logps.detach().to(pi_c.device).float()
    ref_r = reference_rejected_logps.detach().to(pi_c.device).float()

    chosen_rewards = beta * (pi_c - ref_c)
    rejected_rewards = beta * (pi_r - ref_r)
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    reward_margins = chosen_rewards - rejected_rewards

    return {
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_accuracy": reward_accuracies.mean(),
        "reward_margin": reward_margins.mean(),
    }


def compute_dpo_loss(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    chosen_responses: list[str],
    rejected_responses: list[str],
    beta: float,
    max_length: int,
) -> torch.Tensor:
    """Compute scalar batch-mean DPO loss with frozen reference model under torch.no_grad()."""
    dpo_batch = build_dpo_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length=max_length,
    )

    policy_chosen_logps = compute_response_logprobs_from_batch(policy_model, dpo_batch["chosen"])
    policy_rejected_logps = compute_response_logprobs_from_batch(policy_model, dpo_batch["rejected"])

    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs_from_batch(reference_model, dpo_batch["chosen"])
        reference_rejected_logps = compute_response_logprobs_from_batch(reference_model, dpo_batch["rejected"])

    return compute_dpo_loss_from_logps(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
        beta=beta,
    )
