from llm_bench.config import load_run_config
from llm_bench.prompts import PromptSample
from llm_bench.runners import MockRunner
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _prompts(n: int, tokens: int = 256) -> list[PromptSample]:
    return [PromptSample(text=f"prompt {i}", approx_input_tokens=tokens) for i in range(n)]


def test_mock_runner_produces_consistent_measurement_shape():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    runner = MockRunner(cfg)
    runner.warmup(_prompts(cfg.controls.num_warmup))
    out = runner.run(_prompts(cfg.controls.num_prompts))
    assert len(out.measurements) == cfg.controls.num_prompts
    assert out.wall_time_s > 0
    for m in out.measurements:
        assert m.submit_time <= m.first_token_time <= m.end_time
        assert m.output_tokens >= 2


def test_mock_runner_is_deterministic_for_same_seed():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    prompts = _prompts(cfg.controls.num_prompts)
    a = MockRunner(cfg).run(prompts).measurements
    b = MockRunner(cfg).run(prompts).measurements
    assert [m.end_time for m in a] == [m.end_time for m in b]


def test_mock_runner_reflects_spec_decoding_at_low_batch():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    baseline = MockRunner(cfg).run(_prompts(cfg.controls.num_prompts))
    spec_cfg = cfg.model_copy(update={
        "speculative_decoding": {
            "enabled": True,
            "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 5,
        }
    })
    spec = MockRunner(spec_cfg).run(_prompts(cfg.controls.num_prompts))
    # Synthetic model: spec decoding lowers per-token cost at batch=1.
    assert spec.wall_time_s < baseline.wall_time_s
