.PHONY: help install install-gpu test lint smoke sweep analyze clean

PY ?= python
RESULTS ?= results/runs

help:
	@echo "Targets:"
	@echo "  install       pip install harness deps (CPU-only, no torch/vllm)"
	@echo "  install-gpu   pip install GPU deps on top (torch, vllm, autoawq)"
	@echo "  test          run unit tests on the harness"
	@echo "  lint          ruff check"
	@echo "  smoke         run the single-config smoke test (MockRunner, no GPU)"
	@echo "  sweep         run the full sweep matrix (needs GPU)"
	@echo "  analyze       aggregate results/runs/* into results/summary"
	@echo "  clean         remove caches and build artifacts"

install:
	$(PY) -m pip install -e ".[dev]"

install-gpu:
	$(PY) -m pip install -e ".[gpu]"

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

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
