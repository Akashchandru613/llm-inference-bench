.PHONY: help install install-gpu install-judge ci test lint smoke sweep analyze clean

PY ?= python
RESULTS ?= results/runs

# Belt-and-suspenders: also include src/ on PYTHONPATH so the harness works
# even when the editable install is broken (some sandboxed envs strip .pth
# processing).
export PYTHONPATH := src:$(PYTHONPATH)

help:
	@echo "Targets:"
	@echo "  install       pip install harness deps (CPU-only, no torch/vllm)"
	@echo "  install-gpu   pip install GPU deps on top (torch, vllm, autoawq)"
	@echo "  install-judge pip install the LLM-as-judge deps (anthropic)"
	@echo "  test          run unit tests on the harness"
	@echo "  lint          ruff check"
	@echo "  ci            lint + tests + smoke (what GH Actions runs)"
	@echo "  smoke         run the single-config smoke test (MockRunner, no GPU)"
	@echo "  sweep         run the full sweep matrix (needs GPU)"
	@echo "  analyze       aggregate results/runs/* into results/summary"
	@echo "  charts        generate matplotlib charts under docs/charts/"
	@echo "  clean         remove caches and build artifacts"

install:
	$(PY) -m pip install -e ".[dev]"

install-gpu:
	$(PY) -m pip install -e ".[gpu]"

install-judge:
	$(PY) -m pip install -e ".[judge]"

ci: lint test smoke

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

smoke:
	$(PY) -m llm_bench run --config configs/smoke.yaml --runner mock --output $(RESULTS)

sweep:
	$(PY) -m llm_bench sweep --config configs/sweep.yaml --runner vllm --output $(RESULTS)

analyze:
	$(PY) -m llm_bench analyze --input $(RESULTS) --output results/summary

charts:
	$(PY) scripts/make_charts.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
