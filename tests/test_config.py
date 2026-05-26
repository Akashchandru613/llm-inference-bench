from pathlib import Path

import pytest

from llm_bench.config import RunConfig, SweepConfig, load_run_config


REPO = Path(__file__).resolve().parents[1]


def test_smoke_config_loads():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    assert isinstance(cfg, RunConfig)
    assert cfg.batch_size == 1
    assert cfg.prompts.context_bucket == "short"
    assert cfg.context_tokens == 256


def test_sweep_config_loads():
    sweep = SweepConfig.from_yaml(REPO / "configs" / "sweep.yaml")
    assert sweep.name == "sweep-v1"
    assert len(sweep.runs) >= 1
    # All run names must be unique within a sweep.
    names = [r.name for r in sweep.runs]
    assert len(set(names)) == len(names)


def test_fingerprint_stable_and_sensitive():
    cfg = load_run_config(REPO / "configs" / "smoke.yaml")
    fp = cfg.fingerprint()
    assert cfg.fingerprint() == fp
    bumped = cfg.model_copy(update={"batch_size": cfg.batch_size + 1})
    assert bumped.fingerprint() != fp


def test_speculative_decoding_requires_draft_model():
    with pytest.raises(ValueError):
        RunConfig(
            name="bad",
            model={"name": "x/y"},
            speculative_decoding={"enabled": True},
        )
