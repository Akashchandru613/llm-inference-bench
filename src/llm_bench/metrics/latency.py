"""Per-request latency aggregation.

TTFT: submit -> first token. TPOT: per-token cost during decode, computed as
(end_time - first_token_time) / (output_tokens - 1). Requests with output_tokens<2
contribute to TTFT only — they have no decode phase to measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RequestMeasurement:
    request_id: int
    input_tokens: int
    output_tokens: int
    submit_time: float
    first_token_time: float
    end_time: float

    @property
    def ttft_s(self) -> float:
        return self.first_token_time - self.submit_time

    @property
    def tpot_s(self) -> float | None:
        if self.output_tokens < 2:
            return None
        return (self.end_time - self.first_token_time) / (self.output_tokens - 1)

    @property
    def latency_s(self) -> float:
        return self.end_time - self.submit_time


@dataclass(frozen=True)
class LatencySummary:
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    request_latency_p50_ms: float
    request_latency_p95_ms: float
    request_latency_p99_ms: float
    n: int


def _percentile_ms(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values) * 1000.0, q))


def summarize_latency(measurements: Sequence[RequestMeasurement]) -> LatencySummary:
    ttfts = [m.ttft_s for m in measurements]
    tpots = [m.tpot_s for m in measurements if m.tpot_s is not None]
    lats = [m.latency_s for m in measurements]
    return LatencySummary(
        ttft_p50_ms=_percentile_ms(ttfts, 50),
        ttft_p95_ms=_percentile_ms(ttfts, 95),
        ttft_p99_ms=_percentile_ms(ttfts, 99),
        tpot_p50_ms=_percentile_ms(tpots, 50),
        tpot_p95_ms=_percentile_ms(tpots, 95),
        tpot_p99_ms=_percentile_ms(tpots, 99),
        request_latency_p50_ms=_percentile_ms(lats, 50),
        request_latency_p95_ms=_percentile_ms(lats, 95),
        request_latency_p99_ms=_percentile_ms(lats, 99),
        n=len(measurements),
    )
