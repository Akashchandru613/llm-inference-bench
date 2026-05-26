"""vLLM-backed runner.

Constructs an AsyncLLMEngine sized for the config and submits prompts
concurrently so per-request TTFT can be measured directly from the token
stream. Model loading and engine driving are stubbed (raise NotImplementedError)
until we're running on a GPU host — the surrounding integration is real so
filling those in is the only remaining work.
"""
from __future__ import annotations

import time
from typing import Sequence

from ..config import RunConfig
from ..metrics.latency import RequestMeasurement
from ..prompts import PromptSample
from .base import RunOutput


class VLLMRunner:
    name = "vllm"

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._engine = None
        self._hardware = "UNKNOWN"

    def _engine_args(self) -> dict:
        c = self.config
        args: dict = {
            "model": c.model.name,
            "dtype": "auto",
            "seed": c.controls.seed,
            "max_model_len": c.context_tokens + c.sampling.max_new_tokens,
            "trust_remote_code": c.model.trust_remote_code,
        }
        if c.quantization in {"awq", "gptq", "fp8"}:
            args["quantization"] = c.quantization
        if c.model.revision:
            args["revision"] = c.model.revision
        if c.speculative_decoding.enabled:
            args["speculative_model"] = c.speculative_decoding.draft_model
            args["num_speculative_tokens"] = c.speculative_decoding.num_speculative_tokens
        return args

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        # TODO(gpu): instantiate AsyncLLMEngine and capture self._hardware from
        # torch.cuda.get_device_name(0). Deferred until running on a GPU host.
        #
        #     from vllm import AsyncEngineArgs, AsyncLLMEngine
        #     import torch
        #     self._engine = AsyncLLMEngine.from_engine_args(
        #         AsyncEngineArgs(**self._engine_args())
        #     )
        #     self._hardware = torch.cuda.get_device_name(0)
        raise NotImplementedError(
            "VLLMRunner model loading is intentionally deferred. "
            "Fill in _ensure_engine() on a GPU host (see TODO in source)."
        )

    def warmup(self, prompts: Sequence[PromptSample]) -> None:
        self._ensure_engine()
        # TODO(gpu): drive the engine through `prompts` once, discard timings.

    def run(self, prompts: Sequence[PromptSample]) -> RunOutput:
        self._ensure_engine()
        # TODO(gpu): submit all prompts to the async engine concurrently up to
        # `self.config.batch_size`, record submit/first-token/end timestamps
        # per request from the streaming output, then return RunOutput.
        #
        # Sketch:
        #     measurements = []
        #     start = time.monotonic()
        #     async def _one(i, p):
        #         submit = time.monotonic()
        #         first = None
        #         async for out in self._engine.generate(p.text, sampling, i):
        #             if first is None and out.outputs[0].token_ids:
        #                 first = time.monotonic()
        #         end = time.monotonic()
        #         return RequestMeasurement(...)
        #     measurements = asyncio.run(_gather_bounded(_one, prompts, self.config.batch_size))
        #     return RunOutput(measurements, time.monotonic() - start, self._hardware)
        _ = time.monotonic
        raise NotImplementedError("VLLMRunner.run is deferred until running on a GPU host.")

    def shutdown(self) -> None:
        self._engine = None
