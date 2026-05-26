#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/smoke.yaml}"
RUNNER="${2:-mock}"
HARDWARE="${3:-T4}"
python -m llm_bench run \
  --config "$CONFIG" \
  --runner "$RUNNER" \
  --hardware "$HARDWARE" \
  --output results/runs
