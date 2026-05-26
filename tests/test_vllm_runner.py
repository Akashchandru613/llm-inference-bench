"""Structural tests for VLLMRunner — verifies the surface without loading vLLM.

The actual engine-driving path is exercised on a GPU host. These tests pin the
config -> engine-args mapping so a vLLM API drift is caught even before
running on hardware.
"""
from pathlib import Path

import pytest

from llm_bench.config import RunConfig, load_run_config
from llm_bench.runners import VLLMRunner

REPO = Path(__file__).resolve().parents[1]


def test_engine_args_picks_up_quantization():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    cfg = RunConfig.model_validate({
        **cfg.model_dump(),
        "model": {"name": "Qwen/Qwen2.5-7B-Instruct-AWQ"},
        "quantization": "awq",
    })
    args = VLLMRunner(cfg)._engine_args()
    assert args["quantization"] == "awq"
    assert args["max_num_seqs"] == cfg.batch_size
    assert args["max_model_len"] >= cfg.context_tokens + cfg.sampling.max_new_tokens


def test_engine_args_includes_speculative_decoding_when_enabled():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    cfg = RunConfig.model_validate({
        **cfg.model_dump(),
        "speculative_decoding": {
            "enabled": True,
            "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 5,
        },
    })
    args = VLLMRunner(cfg)._engine_args()
    assert args["speculative_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args["num_speculative_tokens"] == 5


def test_engine_args_omits_quantization_for_fp16():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    args = VLLMRunner(cfg)._engine_args()
    assert "quantization" not in args
    assert "speculative_model" not in args


def test_ensure_engine_fails_cleanly_without_vllm():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    runner = VLLMRunner(cfg)
    # vLLM is in the optional [gpu] extras; in CI it isn't installed.
    # The runner should fail with a clear ImportError, not at module-load time.
    with pytest.raises((ImportError, ModuleNotFoundError)):
        runner._ensure_engine()
