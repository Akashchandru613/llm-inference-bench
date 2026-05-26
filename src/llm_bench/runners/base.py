"""Runner interface.

A runner takes a frozen RunConfig and a list of prompts, executes them, and
returns per-request measurements plus the wall time of the measured window.
Warmup runs happen before the measured window and are not returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import RunConfig
from ..metrics.latency import RequestMeasurement
from ..prompts import PromptSample


@dataclass(frozen=True)
class RunOutput:
    measurements: list[RequestMeasurement]
    wall_time_s: float
    hardware: str


class Runner(Protocol):
    name: str

    def __init__(self, config: RunConfig) -> None: ...

    def warmup(self, prompts: Sequence[PromptSample]) -> None: ...

    def run(self, prompts: Sequence[PromptSample]) -> RunOutput: ...

    def shutdown(self) -> None: ...
