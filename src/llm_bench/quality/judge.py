"""LLM-as-judge quality regression check.

Given outputs from a baseline config and a candidate config on the *same*
prompts, pairwise-judge them with a strong reference model. Returns the rate
at which the candidate is judged at least as good as the baseline. Used to
catch quality cliffs from aggressive quantization that perplexity-only
benchmarks miss.

This is a scaffold — the actual judge call is left for later. The interface
is the contract; downstream code should depend only on this surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QualityCheck:
    baseline_name: str
    candidate_name: str
    judge_model: str
    n_pairs: int
    candidate_win_rate: float
    candidate_loss_rate: float
    tie_rate: float


def judge_outputs(
    baseline_outputs: Sequence[str],
    candidate_outputs: Sequence[str],
    prompts: Sequence[str],
    *,
    judge_model: str = "gpt-4o-mini",
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> QualityCheck:
    if not (len(baseline_outputs) == len(candidate_outputs) == len(prompts)):
        raise ValueError("baseline, candidate, and prompts must be aligned")
    # TODO: dispatch to an API client (OpenAI / Anthropic / a local judge),
    # randomize A/B order per pair, parse winner, aggregate.
    raise NotImplementedError(
        "judge_outputs is scaffolded — implement the judge client before the "
        "quality regression check can run."
    )
