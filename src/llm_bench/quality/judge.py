"""LLM-as-judge quality regression check.

Given outputs from a baseline and a candidate config on the *same* prompts,
pairwise-judge them with a reference model. Default judge is Anthropic's
claude-haiku-4-5 with prompt caching on the rubric. The judge call is a
pluggable callable so tests can drive the orchestrator without hitting an API.

To avoid position bias: per pair, the candidate is shown as "A" or "B" with
50/50 random probability seeded by `seed`.

To avoid leaking quantization artifacts into the rubric: the prompt is
included verbatim, but the rubric never mentions which output came from which
config — the judge is blind.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

Verdict = Literal["A", "B", "TIE"]


@dataclass(frozen=True)
class QualityCheck:
    baseline_name: str
    candidate_name: str
    judge_model: str
    n_pairs: int
    candidate_win_rate: float
    candidate_loss_rate: float
    tie_rate: float


class JudgeFn(Protocol):
    def __call__(self, prompt: str, a: str, b: str) -> Verdict: ...


JUDGE_SYSTEM = """You are a careful evaluator comparing two AI assistant outputs to the same user prompt.

Decision criteria, in order of priority:
1. Correctness — is the output factually right and free of fabrications?
2. Responsiveness — does it actually address what the user asked?
3. Clarity — is it well-organized and easy to follow?
4. Concision — does it avoid unnecessary padding?

Ignore: style preferences, tone differences, output length unless one is clearly truncated, the order in which the outputs are presented.

Respond with exactly one token: A, B, or TIE.
- A — output A is meaningfully better
- B — output B is meaningfully better
- TIE — neither is meaningfully better, including small stylistic differences"""


def make_anthropic_judge(
    model: str = "claude-haiku-4-5",
    *,
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> JudgeFn:
    """Build a judge callable backed by the Anthropic Messages API.

    The rubric (system message) is sent with prompt-caching enabled, so after
    the first pair every subsequent call hits the cache and is ~10x cheaper.
    """
    try:
        from anthropic import Anthropic, APIStatusError, APITimeoutError
    except ImportError as exc:
        raise ImportError(
            "anthropic SDK not installed. Install with `pip install -e '.[judge]'`."
        ) from exc

    client = Anthropic()

    def _judge(prompt: str, a: str, b: str) -> Verdict:
        attempt = 0
        while True:
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=4,
                    system=[{
                        "type": "text",
                        "text": JUDGE_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{
                        "role": "user",
                        "content": (
                            f"USER PROMPT:\n{prompt}\n\n"
                            f"OUTPUT A:\n{a}\n\n"
                            f"OUTPUT B:\n{b}\n\n"
                            "Which is better? Reply A, B, or TIE."
                        ),
                    }],
                )
                break
            except (APIStatusError, APITimeoutError):
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(base_delay * (2 ** (attempt - 1)))

        text = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        ).strip().upper()
        if text.startswith("TIE"):
            return "TIE"
        if text.startswith("A"):
            return "A"
        if text.startswith("B"):
            return "B"
        # Judge produced an unparseable verdict — treat as tie so noise doesn't
        # bias the win rate in either direction.
        return "TIE"

    return _judge


def judge_outputs(
    baseline_outputs: Sequence[str],
    candidate_outputs: Sequence[str],
    prompts: Sequence[str],
    *,
    judge: JudgeFn | None = None,
    judge_model: str = "claude-haiku-4-5",
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    seed: int = 0,
) -> QualityCheck:
    if not (len(baseline_outputs) == len(candidate_outputs) == len(prompts)):
        raise ValueError("baseline, candidate, and prompts must be aligned")
    if judge is None:
        judge = make_anthropic_judge(model=judge_model)

    rng = random.Random(seed)
    wins = losses = ties = 0
    for prompt, base, cand in zip(prompts, baseline_outputs, candidate_outputs):
        cand_is_a = rng.random() < 0.5
        a, b = (cand, base) if cand_is_a else (base, cand)
        verdict = judge(prompt, a, b)
        if verdict == "TIE":
            ties += 1
        elif (verdict == "A" and cand_is_a) or (verdict == "B" and not cand_is_a):
            wins += 1
        else:
            losses += 1

    n = len(prompts)
    return QualityCheck(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        judge_model=judge_model,
        n_pairs=n,
        candidate_win_rate=wins / n,
        candidate_loss_rate=losses / n,
        tie_rate=ties / n,
    )
