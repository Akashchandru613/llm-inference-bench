"""Synthetic runner for smoke tests and harness CI.

Produces deterministic per-request measurements that vary plausibly with config
(batch size, quantization, speculative decoding, context length) so the
end-to-end pipeline can be exercised without a GPU. The numbers are fake —
never report them as benchmark results.
"""
from __future__ import annotations

import random
from typing import Sequence

from ..config import RunConfig
from ..metrics.latency import RequestMeasurement
from ..prompts import PromptSample
from .base import RunOutput


class MockRunner:
    name = "mock"

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._rng = random.Random(config.controls.seed)

    def warmup(self, prompts: Sequence[PromptSample]) -> None:
        for _ in prompts:
            self._rng.random()

    def run(self, prompts: Sequence[PromptSample]) -> RunOutput:
        c = self.config
        # Per-token decode cost in seconds, hand-tuned to be roughly T4-like.
        base_tpot = 0.040
        if c.quantization in {"awq", "gptq", "fp8"}:
            base_tpot *= 0.65
        # Batching is memory-bandwidth amortized: tpot shrinks sublinearly.
        base_tpot *= 1.0 + 0.4 * (1.0 - 1.0 / max(1, c.batch_size) ** 0.5)
        if c.speculative_decoding.enabled:
            # Spec wins at small batch, shrinks at high batch, inverts past ~32.
            scale = max(0.6, 1.0 - 0.35 / max(1, c.batch_size) ** 0.5)
            if c.batch_size > 32:
                scale = 1.1
            base_tpot *= scale

        # Prefill scales linearly with context length and sublinearly with batch.
        prefill_per_token = 0.0006
        base_ttft = prefill_per_token * c.context_tokens
        base_ttft *= 1.0 + 0.25 * (c.batch_size ** 0.5 - 1.0)

        clock = 0.0
        measurements: list[RequestMeasurement] = []
        for i, prompt in enumerate(prompts):
            submit = clock
            ttft = base_ttft * self._rng.uniform(0.9, 1.15)
            output_tokens = max(2, int(c.sampling.max_new_tokens * self._rng.uniform(0.85, 1.0)))
            decode = base_tpot * (output_tokens - 1) * self._rng.uniform(0.97, 1.06)
            first = submit + ttft
            end = first + decode
            measurements.append(RequestMeasurement(
                request_id=i,
                input_tokens=prompt.approx_input_tokens,
                output_tokens=output_tokens,
                submit_time=submit,
                first_token_time=first,
                end_time=end,
            ))
            # Requests overlap when batched — wall clock advances by ~1/batch.
            clock = submit + (end - submit) / max(1, c.batch_size)

        return RunOutput(measurements=measurements, wall_time_s=clock, hardware="MOCK")

    def shutdown(self) -> None:
        pass
