#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/sweep.yaml}"
RUNNER="${2:-vllm}"
HARDWARE="${3:-T4}"
python -m llm_bench sweep \
  --config "$CONFIG" \
  --runner "$RUNNER" \
  --hardware "$HARDWARE" \
  --output results/runs
python -m llm_bench analyze --input results/runs --output results/summary
