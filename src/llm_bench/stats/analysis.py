"""Significance tests and effect sizes used in pairwise comparisons.

Latency distributions are heavy-tailed and non-normal; default to a
non-parametric test (Mann-Whitney U) and bootstrap CIs over the median,
not the mean. Welch's t is exposed for cases where the user has a reason
to assume Gaussian residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    confidence: float


def bootstrap_ci(
    samples: Sequence[float],
    *,
    statistic: Literal["median", "mean"] = "median",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    rng_seed: int = 0,
) -> BootstrapCI:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return BootstrapCI(point=float("nan"), low=float("nan"), high=float("nan"), confidence=confidence)
    rng = np.random.default_rng(rng_seed)
    func = np.median if statistic == "median" else np.mean
    point = float(func(arr))
    resampled = func(rng.choice(arr, size=(n_resamples, arr.size), replace=True), axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(resampled, alpha))
    high = float(np.quantile(resampled, 1.0 - alpha))
    return BootstrapCI(point=point, low=low, high=high, confidence=confidence)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    pooled = np.sqrt(((aa.size - 1) * aa.var(ddof=1) + (bb.size - 1) * bb.var(ddof=1)) / (aa.size + bb.size - 2))
    if pooled == 0:
        return float("nan")
    return float((aa.mean() - bb.mean()) / pooled)


@dataclass(frozen=True)
class ComparisonResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float
    a_median: float
    b_median: float
    median_delta: float
    relative_change: float


def compare_distributions(
    a: Sequence[float],
    b: Sequence[float],
    *,
    test: Literal["mannwhitney", "welch"] = "mannwhitney",
) -> ComparisonResult:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if test == "mannwhitney":
        u, p = stats.mannwhitneyu(aa, bb, alternative="two-sided")
        statistic = float(u)
    else:
        t, p = stats.ttest_ind(aa, bb, equal_var=False)
        statistic = float(t)
    a_med, b_med = float(np.median(aa)), float(np.median(bb))
    delta = a_med - b_med
    rel = delta / b_med if b_med != 0 else float("nan")
    return ComparisonResult(
        test=test,
        statistic=statistic,
        p_value=float(p),
        effect_size=cohens_d(aa, bb),
        a_median=a_med,
        b_median=b_med,
        median_delta=delta,
        relative_change=rel,
    )
