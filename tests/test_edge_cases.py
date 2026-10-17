"""Comprehensive offline CPU edge-case tests for SFT/DPO batching, logprobs, and losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from src.align.losses import compute_dpo_metrics_from_logps
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoEosTokenizer(ToyTokenizer):
    """Tokenizer variant with eos_token_id = None."""

    eos_token_id = None


class NoPadTokenizer(ToyTokenizer):
    """Tokenizer variant with pad_token_id = None (falls back to eos_token_id)."""

    pad_token_id = None


class NoPadNoEosTokenizer(ToyTokenizer):
    """Tokenizer variant with both pad_token_id = None and eos_token_id = None (falls back to 0)."""

    pad_token_id = None
    eos_token_id = None


class BosDefaultTokenizer(ToyTokenizer):
    """Tokenizer that prepends a BOS token by default unless add_special_tokens=False."""

    bos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        base = super().encode(text, add_special_tokens=False)
        if add_special_tokens:
            return [self.bos_token_id] + base
        return base


def test_prompt_longer_than_max_length_produces_zero_logprob_and_finite_sft_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    prompts = ["This prompt is definitely longer than eight characters"]
    responses = ["answer"]

    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    assert batch["labels"].eq(-100).all()

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=8)
    assert logps.shape == (1,)
    assert torch.isfinite(logps).all()
    torch.testing.assert_close(logps, torch.zeros(1))

    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_response_truncation_drops_eos_when_sequence_hits_max_length():
    tokenizer = ToyTokenizer()
    # Prompt length = 2 ("Q:"), response length = 3 ("abc"), +1 EOS = 6 total tokens.
    # With max_length = 5, EOS must be truncated off the right.
    batch = adapter.build_sft_batch(tokenizer, ["Q:"], ["abc"], max_length=5)
    assert batch["input_ids"].shape == (1, 5)
    assert batch["labels"][0, :2].eq(-100).all()
    assert batch["labels"][0, 2:].ne(-100).all()
    assert (batch["labels"][0] != tokenizer.eos_token_id).all()


def test_tokenizer_with_no_eos_token():
    tokenizer = NoEosTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: "], ["ok"], max_length=10)
    # Prompt "Q: " -> 3 tokens; response "ok" -> 2 tokens (no EOS appended).
    assert batch["input_ids"].shape == (1, 5)
    assert batch["labels"][0, :3].eq(-100).all()
    assert batch["labels"][0, 3:5].ne(-100).all()


def test_tokenizer_pad_fallbacks_to_eos_then_zero():
    tok_no_pad = NoPadTokenizer()
    batch_eos_pad = adapter.build_sft_batch(tok_no_pad, ["Q:", "Q:"], ["a", "abcd"], max_length=12)
    # Row 0 is shorter and should be padded with eos_token_id (1)
    assert batch_eos_pad["input_ids"][0, -1].item() == tok_no_pad.eos_token_id
    assert batch_eos_pad["attention_mask"][0, -1].item() == 0
    assert batch_eos_pad["labels"][0, -1].item() == -100

    tok_neither = NoPadNoEosTokenizer()
    batch_zero_pad = adapter.build_sft_batch(tok_neither, ["Q:", "Q:"], ["a", "abcd"], max_length=12)
    assert batch_zero_pad["input_ids"][0, -1].item() == 0
    assert batch_zero_pad["attention_mask"][0, -1].item() == 0
    assert batch_zero_pad["labels"][0, -1].item() == -100


def test_pad_equals_eos_keeps_response_eos_labeled_and_masks_padding():
    """Simulate Qwen2.5-0.5B where pad_token_id == eos_token_id (both 151643 / 1)."""
    tok_equal = NoPadTokenizer()  # pad_token_id is None -> resolves to eos_token_id (1)
    tok_equal.pad_token_id = tok_equal.eos_token_id
    batch = adapter.build_sft_batch(tok_equal, ["Q:", "Q:"], ["a", "abcd"], max_length=12)

    # Row 0: prompt "Q:" (2 tokens) + response "a" (1 token) + EOS (1 token at index 3) + 3 pad tokens (indices 4..6)
    assert batch["input_ids"][0, 3].item() == tok_equal.eos_token_id
    assert batch["labels"][0, 3].item() == tok_equal.eos_token_id  # EOS kept labeled!
    assert (batch["input_ids"][0, 4:] == tok_equal.pad_token_id).all()
    assert (batch["labels"][0, 4:] == -100).all()  # Padding masked -100!
    assert (batch["attention_mask"][0, :4] == 1).all()
    assert (batch["attention_mask"][0, 4:] == 0).all()


def test_prompt_uses_default_special_tokens_and_response_disables_special_tokens():
    tokenizer = BosDefaultTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Hi"], ["ok"], max_length=12)
    # Prompt "Hi" -> [BOS, 'H', 'i'] (3 tokens, all masked -100)
    # Response "ok" -> ['o', 'k', EOS] (no second BOS, 3 unmasked tokens)
    assert batch["input_ids"].shape == (1, 6)
    assert batch["input_ids"][0, 0].item() == tokenizer.bos_token_id
    assert batch["labels"][0, :3].eq(-100).all()
    assert batch["labels"][0, 3:].ne(-100).all()
    assert batch["labels"][0, 3].item() != tokenizer.bos_token_id


def test_mixed_lengths_and_row_permutation_invariance():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.6)
    prompts = ["P1: ", "Longer prompt two: ", "P3: "]
    responses = ["short", "medium response", "x"]
    perm = [2, 0, 1]

    logps_orig = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=32)
    logps_perm = adapter.compute_response_logprobs(
        model,
        tokenizer,
        [prompts[i] for i in perm],
        [responses[i] for i in perm],
        max_length=32,
    )
    torch.testing.assert_close(logps_orig[perm], logps_perm)


def test_hand_computed_three_token_response_logprob():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.8)
    # Prompt "A" (1 token), response "bc" + EOS = 3 response tokens
    prompt = "A"
    response = "bc"
    logps = adapter.compute_response_logprobs(model, tokenizer, [prompt], [response], max_length=10)

    # ToyCausalLM produces identical logits = model.logit_bias at every position
    log_probs_vocab = F.log_softmax(model.logit_bias.float(), dim=-1)
    expected_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert len(expected_ids) == 3
    hand_sum = sum(log_probs_vocab[token_id] for token_id in expected_ids)

    torch.testing.assert_close(logps[0], hand_sum, rtol=1e-5, atol=1e-6)


def test_dpo_loss_equals_log2_and_gradients_reach_policy_only():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    with torch.no_grad():
        reference.logit_bias.copy_(policy.logit_bias)

    prompts = ["Q1: ", "Q2: "]
    chosen = ["yes", "good"]
    rejected = ["no", "bad"]

    loss = adapter.compute_dpo_loss(
        policy,
        reference,
        tokenizer,
        prompts,
        chosen,
        rejected,
        beta=0.1,
        max_length=16,
    )
    torch.testing.assert_close(loss, torch.tensor(math.log(2.0)), rtol=1e-5, atol=1e-6)

    loss.backward()
    assert policy.logit_bias.grad is not None
    assert policy.logit_bias.grad.abs().sum() > 0
    assert reference.logit_bias.grad is None

    # Verify strict reward accuracy starts at 0.0 when policy == reference
    with torch.no_grad():
        pi_c = adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, max_length=16)
        pi_r = adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, max_length=16)
        metrics = compute_dpo_metrics_from_logps(pi_c, pi_r, pi_c.clone(), pi_r.clone(), beta=0.1)
        assert metrics["reward_accuracy"].item() == 0.0
        assert metrics["reward_margin"].item() == 0.0
