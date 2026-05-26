import numpy as np

from llm_bench.stats import bootstrap_ci, cohens_d, compare_distributions


def test_bootstrap_ci_brackets_median():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=10.0, scale=1.0, size=500).tolist()
    ci = bootstrap_ci(samples, statistic="median", confidence=0.95, rng_seed=0)
    assert ci.low < ci.point < ci.high
    assert abs(ci.point - 10.0) < 0.2


def test_compare_distributions_detects_real_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=20.0, scale=2.0, size=200).tolist()
    b = rng.normal(loc=22.0, scale=2.0, size=200).tolist()
    res = compare_distributions(a, b, test="mannwhitney")
    assert res.p_value < 0.01
    assert res.median_delta < 0
    # Cohen's d magnitude ~1 for a 2 sigma shift on equal-variance Gaussians.
    assert abs(res.effect_size) > 0.7


def test_compare_distributions_no_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(loc=5.0, scale=1.0, size=200).tolist()
    b = rng.normal(loc=5.0, scale=1.0, size=200).tolist()
    res = compare_distributions(a, b, test="mannwhitney")
    assert res.p_value > 0.05


def test_cohens_d_zero_for_identical_samples():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cohens_d(a, a) == 0.0
