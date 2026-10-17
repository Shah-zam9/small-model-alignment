"""Single shared prompt template used across SFT, DPO, and evaluation generation."""

from __future__ import annotations

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Format a single user instruction into the canonical prompt prefix."""
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
