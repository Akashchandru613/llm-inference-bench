import math

from llm_bench.metrics import cost_per_million_tokens, summarize_latency, summarize_throughput
from llm_bench.metrics.latency import RequestMeasurement


def _mk(i: int, ttft: float, tpot: float, out_tokens: int = 100, submit: float = 0.0) -> RequestMeasurement:
    first = submit + ttft
    end = first + tpot * (out_tokens - 1)
    return RequestMeasurement(
        request_id=i,
        input_tokens=64,
        output_tokens=out_tokens,
        submit_time=submit,
        first_token_time=first,
        end_time=end,
    )


def test_latency_percentiles_use_milliseconds():
    measurements = [_mk(i, ttft=0.100 + 0.001 * i, tpot=0.020) for i in range(100)]
    s = summarize_latency(measurements)
    assert s.n == 100
    assert math.isclose(s.ttft_p50_ms, 149.5, abs_tol=1.0)
    assert s.ttft_p99_ms > s.ttft_p95_ms > s.ttft_p50_ms
    assert math.isclose(s.tpot_p50_ms, 20.0, abs_tol=0.5)


def test_short_outputs_excluded_from_tpot():
    measurements = [
        _mk(0, ttft=0.05, tpot=0.0, out_tokens=1),
        _mk(1, ttft=0.05, tpot=0.02, out_tokens=10),
    ]
    s = summarize_latency(measurements)
    # First request has no decode phase; only the second contributes to TPOT.
    assert math.isclose(s.tpot_p50_ms, 20.0, abs_tol=0.5)


def test_throughput_system_vs_per_request():
    measurements = [_mk(i, ttft=0.05, tpot=0.02, out_tokens=50) for i in range(8)]
    s = summarize_throughput(measurements, wall_time_s=2.0)
    assert s.total_output_tokens == 400
    assert math.isclose(s.system_output_tps, 200.0, abs_tol=0.1)
    # Per-request throughput is lower than system throughput when batched.
    assert s.per_request_output_tps_mean > 0


def test_cost_per_million_tokens_t4():
    # 1 hour on T4 at $0.35/hr producing 1M tokens -> $0.35 per M.
    cost = cost_per_million_tokens(
        wall_time_s=3600.0, total_output_tokens=1_000_000, hardware="T4"
    )
    assert math.isclose(cost, 0.35, rel_tol=1e-6)


def test_cost_unknown_hardware_raises():
    import pytest

    with pytest.raises(KeyError):
        cost_per_million_tokens(wall_time_s=10.0, total_output_tokens=100, hardware="MOCK")
