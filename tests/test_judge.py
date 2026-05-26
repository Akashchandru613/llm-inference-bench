"""Judge orchestrator tests — drive the loop with synthetic judges.

The Anthropic-backed judge is exercised on-line and not from CI; these tests
pin the win/loss accounting and the position-bias defeat property.
"""
import pytest

from llm_bench.quality import judge_outputs


def test_smart_judge_finds_candidate_when_better():
    def smart_judge(prompt, a, b):
        # Whichever slot contains the candidate token wins.
        return "A" if "CAND" in a else "B"

    result = judge_outputs(
        baseline_outputs=["BASE"] * 32,
        candidate_outputs=["CAND"] * 32,
        prompts=["p"] * 32,
        judge=smart_judge,
        seed=0,
    )
    assert result.candidate_win_rate == 1.0
    assert result.candidate_loss_rate == 0.0
    assert result.tie_rate == 0.0


def test_smart_judge_finds_baseline_when_better():
    def smart_judge(prompt, a, b):
        # Always prefer whichever side carries the baseline token.
        return "A" if "BASE" in a else "B"

    result = judge_outputs(
        baseline_outputs=["BASE"] * 32,
        candidate_outputs=["CAND"] * 32,
        prompts=["p"] * 32,
        judge=smart_judge,
        seed=0,
    )
    assert result.candidate_loss_rate == 1.0


def test_position_bias_evens_out_after_shuffling():
    # A judge that *always* says A (worst-case position bias).
    # Random A/B order shuffling should give a win rate near 0.5.
    def biased_judge(prompt, a, b):
        return "A"

    result = judge_outputs(
        baseline_outputs=["b"] * 400,
        candidate_outputs=["c"] * 400,
        prompts=["p"] * 400,
        judge=biased_judge,
        seed=42,
    )
    assert abs(result.candidate_win_rate - 0.5) < 0.06


def test_tie_judge_yields_full_tie_rate():
    def tie_judge(prompt, a, b):
        return "TIE"

    result = judge_outputs(
        baseline_outputs=["x"] * 10,
        candidate_outputs=["y"] * 10,
        prompts=["p"] * 10,
        judge=tie_judge,
        seed=0,
    )
    assert result.tie_rate == 1.0
    assert result.candidate_win_rate == 0.0


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError):
        judge_outputs(
            baseline_outputs=["a", "b"],
            candidate_outputs=["x"],
            prompts=["p1", "p2"],
            judge=lambda p, a, b: "TIE",
        )


def test_judge_call_count_matches_n_pairs():
    calls = []

    def counting_judge(prompt, a, b):
        calls.append((prompt, a, b))
        return "TIE"

    judge_outputs(
        baseline_outputs=["b1", "b2", "b3"],
        candidate_outputs=["c1", "c2", "c3"],
        prompts=["p1", "p2", "p3"],
        judge=counting_judge,
        seed=0,
    )
    assert len(calls) == 3
