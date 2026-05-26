# llm-inference-bench

[![tests](https://github.com/Akashchandru613/llm-inference-bench/actions/workflows/test.yml/badge.svg)](https://github.com/Akashchandru613/llm-inference-bench/actions/workflows/test.yml)

A reproducible benchmark suite for open-weight LLM inference, comparing the
optimizations that matter in production: **quantization × batch size × context
length**, with rigorous methodology and real measurements on free-tier
infrastructure.

## Headline results

Mistral-7B-Instruct-v0.2-AWQ on NVIDIA T4 (Kaggle, 16 GB), vLLM 0.5.5, XFormers
attention. 50 prompts per config (20–30 for long-context), warmup of 4. Cost
column uses T4 list price of $0.35/hr.

| config                         | system tok/s | TTFT p50 (ms) | $ / M tokens |
| ------------------------------ | -----------: | ------------: | -----------: |
| AWQ, batch=1, 256-tok context  |         23.4 |           547 |        $4.15 |
| AWQ, batch=4, 256-tok context  |         69.5 |         1,599 |        $1.40 |
| **AWQ, batch=16, 256-tok**     |    **133.2** |         3,722 |    **$0.73** |
| AWQ, batch=1, 2,048-tok        |         15.4 |         2,961 |        $6.32 |
| AWQ, batch=4, 2,048-tok        |         26.7 |         6,059 |        $3.64 |
| AWQ, batch=1, 8,192-tok        |          6.1 |        14,576 |       $16.00 |

### Three findings

1. **Batching is the dominant lever for cost/throughput on a T4.** Going from
   batch=1 to batch=16 gives **5.7× system throughput** and **5.7× cheaper per
   token**, at the cost of 7× worse TTFT. Workloads that tolerate ~3 s TTFT
   ship far more tokens per dollar than ones that don't.
2. **Long context at batch=1 is brutal.** $16/M tokens and 14.6 s TTFT for an
   8 K-token prefill on a single T4 — barely usable. This is where smarter
   serving (paged-attention tuning, prefix caching, larger GPUs) earns its
   keep.
3. **Context-induced TTFT cost grows sublinearly with batch size.** Short →
   medium context multiplies TTFT by **5.4× at bs=1** but only **3.8× at
   bs=4** — vLLM's batched prefill amortizes context cost across slots better
   than a naive linear-cost intuition suggests.

Full per-run records, including per-request percentiles, prompt fingerprints,
and GPU env snapshots: [`results/runs/`](results/runs/). Methodology and
limitations: [`methodology.md`](methodology.md).

## Quickstart

```bash
# CPU-only deps for the harness and tests
make install

# Run the unit tests (27 of them, all green in CI)
make test

# Smoke-test the pipeline end-to-end with a synthetic runner (no GPU needed)
make smoke
```

To actually run benchmarks you need an NVIDIA GPU. The intended path is the
[Kaggle notebook wrapper](notebooks/kaggle_runner.ipynb), which gives you 30
hours/week of free T4 time:

```bash
make install-gpu              # vllm, torch, autoawq — only on the GPU host
make sweep                    # runs every config in configs/sweep.yaml
make analyze                  # aggregates results/runs/* into results/summary
```

## What's in the box

```
src/llm_bench/
  config.py         RunConfig / SweepConfig pydantic schemas
  cli.py            run / sweep / analyze commands
  env.py            git sha, GPU, driver, CUDA, lib versions per run
  records.py        stable JSON schema for result records
  prompts/          deterministic ShareGPT sampling with synthetic fallback
  runners/
    base.py         Runner protocol
    mock_runner.py  synthetic runner — exercises the pipeline without a GPU
    vllm_runner.py  vLLM AsyncLLMEngine + worker-pool concurrency
  metrics/
    latency.py      TTFT / TPOT / request latency percentiles
    throughput.py   system tok/s vs per-request tok/s
    memory.py       peak GPU memory accounting
    cost.py         $/M tokens with per-hardware hourly pricing
  stats/
    analysis.py     bootstrap CIs, Mann-Whitney U, Welch's t, Cohen's d
  quality/
    judge.py        LLM-as-judge regression check (Anthropic-backed)
configs/
  smoke.yaml        single-config smoke test
  sweep.yaml        T4-feasible sweep across batch × context
notebooks/
  kaggle_runner.ipynb   thin wrapper for running on Kaggle T4
results/
  runs/             raw JSON per (config, repeat)
  summary/          aggregated tables for the writeup
```

## Methodology

See [methodology.md](methodology.md) for: warmup protocol, repeat counts and
seed handling, prompt sampling, statistical tests, GPU-memory headroom rules,
and an honest list of limitations of the current sweep.

## Reproducing the sweep

```bash
gh repo clone Akashchandru613/llm-inference-bench
```

Then run the Kaggle notebook ([`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb))
with GPU + Internet enabled. The notebook clones this repo, installs the GPU
dependency stack (including the
[hard-won `pyairports` stub workaround](#known-environment-quirks)), runs the
sweep, and emits `bench-results.zip`.

### Known environment quirks

Both encountered and worked around in the harness:

- `pyairports==2.1.1` is needed by vLLM's `outlines` dependency but the only
  version on PyPI is a broken `0.0.1` stub. A minimal in-tree stub satisfies
  the import without affecting inference.
- `transformers ≥ 4.45` rewrote `rope_scaling` to use `rope_type` keys; vLLM
  0.5.x still expects the old `type` + `factor` format. `requirements-gpu.txt`
  pins `transformers < 4.45`.
- T4 (sm_75) doesn't support FlashAttention-2; vLLM falls back to XFormers
  automatically. Expect ~10–20% lower throughput than Ampere+ hardware.

## Status

- [x] Config schema, metrics math, stats, prompt sampling, MockRunner, CLI
- [x] vLLM async runner: AsyncLLMEngine, worker-pool with bounded concurrency,
      streaming TTFT capture, peak-memory accounting
- [x] LLM-as-judge orchestrator with Anthropic backend + prompt caching
- [x] GitHub Actions CI: ruff + pytest + smoke run on 3.10/3.11/3.12
- [x] **First sweep on Kaggle T4** — 6 configs, Mistral-7B-AWQ
- [ ] Blog post with the headline finding
- [ ] Speculative decoding axis (needs vLLM 0.6+ for vocab-compatible drafts)
- [ ] FP16 baseline (needs L4/A10G — won't fit FP16 7B + KV cache on T4)
- [ ] LLM-as-judge quality regression check on the AWQ outputs

## License

MIT
