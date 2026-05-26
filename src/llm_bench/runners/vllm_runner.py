"""vLLM-backed runner.

Drives an `AsyncLLMEngine` with a worker pool that keeps `batch_size` requests
in flight at all times — this matches the steady-state batching regime servers
actually operate in, and lets the engine's scheduler batch up to `max_num_seqs`
on every forward pass.

Per-request TTFT is read from the streaming output: the timestamp of the first
chunk with any decoded tokens. Decode timings come from the final
`RequestOutput` carrying `finished=True`.

vLLM and torch are imported lazily so the module loads on a no-GPU host.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Sequence

from ..config import RunConfig
from ..metrics.latency import RequestMeasurement
from ..prompts import PromptSample
from .base import RunOutput

if TYPE_CHECKING:
    from vllm import AsyncLLMEngine


class VLLMRunner:
    name = "vllm"

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._engine: "AsyncLLMEngine | None" = None
        self._hardware = "UNKNOWN"

    def _engine_args(self) -> dict:
        c = self.config
        args: dict = {
            "model": c.model.name,
            "dtype": "auto",
            "seed": c.controls.seed,
            "max_model_len": c.context_tokens + c.sampling.max_new_tokens,
            "max_num_seqs": c.batch_size,
            "trust_remote_code": c.model.trust_remote_code,
            "enforce_eager": False,
            "disable_log_requests": True,
        }
        if c.quantization in {"awq", "gptq", "fp8"}:
            args["quantization"] = c.quantization
        if c.model.revision:
            args["revision"] = c.model.revision
        if c.speculative_decoding.enabled:
            args["speculative_model"] = c.speculative_decoding.draft_model
            args["num_speculative_tokens"] = c.speculative_decoding.num_speculative_tokens
        return args

    def _sampling_params(self):
        from vllm import SamplingParams

        c = self.config.sampling
        return SamplingParams(
            temperature=c.temperature,
            top_p=c.top_p,
            max_tokens=c.max_new_tokens,
            seed=self.config.controls.seed,
        )

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import torch

        self._engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**self._engine_args())
        )
        self._hardware = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )

    async def _generate_one(
        self,
        request_id: str,
        prompt: PromptSample,
        sampling,
    ) -> RequestMeasurement:
        assert self._engine is not None
        submit = time.monotonic()
        first_token: float | None = None
        prompt_tokens = 0
        output_tokens = 0
        async for request_output in self._engine.generate(prompt.text, sampling, request_id):
            if first_token is None and request_output.outputs and request_output.outputs[0].token_ids:
                first_token = time.monotonic()
            if request_output.prompt_token_ids is not None:
                prompt_tokens = len(request_output.prompt_token_ids)
            if request_output.outputs:
                output_tokens = len(request_output.outputs[0].token_ids)
        end = time.monotonic()
        if first_token is None:
            # Edge case: a 0-token output. Treat first_token as end so TTFT covers the whole call.
            first_token = end
        return RequestMeasurement(
            request_id=int(request_id.rsplit("-", 1)[-1], 16) % (2**31),
            input_tokens=prompt_tokens or prompt.approx_input_tokens,
            output_tokens=max(output_tokens, 1),
            submit_time=submit,
            first_token_time=first_token,
            end_time=end,
        )

    async def _run_async(self, prompts: Sequence[PromptSample]) -> tuple[list[RequestMeasurement], float]:
        sampling = self._sampling_params()
        queue: asyncio.Queue[tuple[int, PromptSample]] = asyncio.Queue()
        for i, p in enumerate(prompts):
            queue.put_nowait((i, p))

        measurements: list[RequestMeasurement] = []
        lock = asyncio.Lock()

        async def worker(worker_id: int) -> None:
            while True:
                try:
                    i, prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rid = f"{worker_id}-{i:08x}"
                m = await self._generate_one(rid, prompt, sampling)
                async with lock:
                    measurements.append(m)

        start = time.monotonic()
        workers = [asyncio.create_task(worker(w)) for w in range(self.config.batch_size)]
        await asyncio.gather(*workers)
        wall = time.monotonic() - start

        measurements.sort(key=lambda m: m.submit_time)
        for stable_id, m in enumerate(measurements):
            measurements[stable_id] = RequestMeasurement(
                request_id=stable_id,
                input_tokens=m.input_tokens,
                output_tokens=m.output_tokens,
                submit_time=m.submit_time,
                first_token_time=m.first_token_time,
                end_time=m.end_time,
            )
        return measurements, wall

    def warmup(self, prompts: Sequence[PromptSample]) -> None:
        self._ensure_engine()
        if not prompts:
            return
        # Reset memory peak so warmup allocations don't pollute the post-run reading.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        asyncio.run(self._run_async(prompts))
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def run(self, prompts: Sequence[PromptSample]) -> RunOutput:
        self._ensure_engine()
        measurements, wall = asyncio.run(self._run_async(prompts))
        return RunOutput(measurements=measurements, wall_time_s=wall, hardware=self._hardware)

    def shutdown(self) -> None:
        if self._engine is None:
            return
        # vLLM doesn't expose an explicit close; drop the reference and let
        # CUDA contexts unwind when the process exits or the next engine is built.
        self._engine = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
