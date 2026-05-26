"""Throughput aggregation.

System throughput is total output tokens divided by wall time across the whole
measured run. Per-request throughput is the average across requests — these can
diverge meaningfully when batching is in play, so we report both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .latency import RequestMeasurement


@dataclass(frozen=True)
class ThroughputSummary:
    total_output_tokens: int
    total_input_tokens: int
    wall_time_s: float
    system_output_tps: float
    per_request_output_tps_mean: float


def summarize_throughput(
    measurements: Sequence[RequestMeasurement],
    wall_time_s: float,
) -> ThroughputSummary:
    total_out = sum(m.output_tokens for m in measurements)
    total_in = sum(m.input_tokens for m in measurements)
    system_tps = total_out / wall_time_s if wall_time_s > 0 else 0.0
    per_req_tps = [
        m.output_tokens / m.latency_s
        for m in measurements
        if m.latency_s > 0
    ]
    per_req_mean = sum(per_req_tps) / len(per_req_tps) if per_req_tps else 0.0
    return ThroughputSummary(
        total_output_tokens=total_out,
        total_input_tokens=total_in,
        wall_time_s=wall_time_s,
        system_output_tps=system_tps,
        per_request_output_tps_mean=per_req_mean,
    )
