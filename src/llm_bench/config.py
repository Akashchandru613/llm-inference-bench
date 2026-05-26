"""Benchmark configuration schema.

A `RunConfig` fully describes one measurement: model, quantization, speculative
decoding, batch size, context length, sampling params, and run controls.
Hashing is deterministic over the benchmark-relevant fields only — two configs
with the same hash should produce comparable measurements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Quantization = Literal["fp16", "bf16", "awq", "gptq", "fp8"]
ContextBucket = Literal["short", "medium", "long"]

CONTEXT_TOKENS: dict[ContextBucket, int] = {
    "short": 256,
    "medium": 2048,
    "long": 8192,
}


class ModelSpec(BaseModel):
    name: str = Field(..., description="HF model id, e.g. 'Qwen/Qwen2.5-7B-Instruct'")
    revision: str | None = None
    trust_remote_code: bool = False


class SpeculativeDecoding(BaseModel):
    enabled: bool = False
    draft_model: str | None = None
    num_speculative_tokens: int = 5

    @model_validator(mode="after")
    def _validate(self) -> "SpeculativeDecoding":
        if self.enabled and not self.draft_model:
            raise ValueError("speculative_decoding.enabled requires draft_model")
        return self


class SamplingParams(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256


class RunControls(BaseModel):
    num_prompts: int = 100
    num_warmup: int = 8
    seed: int = 0
    repeats: int = 1


class PromptSource(BaseModel):
    dataset: str = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    split: str = "train"
    context_bucket: ContextBucket = "short"


class RunConfig(BaseModel):
    name: str
    model: ModelSpec
    quantization: Quantization = "fp16"
    speculative_decoding: SpeculativeDecoding = SpeculativeDecoding()
    batch_size: int = 1
    sampling: SamplingParams = SamplingParams()
    prompts: PromptSource = PromptSource()
    controls: RunControls = RunControls()

    @property
    def context_tokens(self) -> int:
        return CONTEXT_TOKENS[self.prompts.context_bucket]

    def fingerprint(self) -> str:
        payload = {
            "model": self.model.model_dump(),
            "quantization": self.quantization,
            "speculative_decoding": self.speculative_decoding.model_dump(),
            "batch_size": self.batch_size,
            "sampling": self.sampling.model_dump(),
            "prompts": self.prompts.model_dump(),
            "controls": self.controls.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class SweepConfig(BaseModel):
    name: str
    runs: list[RunConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "SweepConfig":
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)


def load_run_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text())
    return RunConfig.model_validate(data)
