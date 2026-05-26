# llm-inference-bench

A reproducible benchmark suite for open-weight LLM inference, comparing the
optimizations that matter in production: **quantization × speculative decoding
× batch size × context length**, with rigorous methodology, statistical
significance testing, and quality regression checks.

## The question this benchmark answers

> For open-weight LLMs at the 7B–13B scale, what combination of optimizations
> maximizes useful output tokens per second per dollar — and where do the
> bottlenecks shift across batch size and context length?

People publish individual numbers (vLLM, AWQ paper, the speculative decoding
paper). Almost nobody publishes the *interactions* with proper methodology.
This repo does.

## Headline results

| config | sys tok/s | TTFT p50 (ms) | TPOT p50 (ms) | $/M tok | quality vs FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| _pending — fill in after running the sweep on Kaggle T4_ | | | | | |

## Quickstart

```bash
# CPU-only deps for the harness and tests
make install

# Run the unit tests
make test

# Smoke-test the pipeline end-to-end with a synthetic runner (no GPU needed)
make smoke
```

To actually run benchmarks, you need a GPU. The intended path is the
[Kaggle notebook wrapper](notebooks/kaggle_runner.ipynb), which gives you
30 hours/week of free T4 time:

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
  prompts/          deterministic ShareGPT sampling with fallback
  runners/
    base.py         Runner protocol
    mock_runner.py  synthetic runner — exercises the pipeline without a GPU
    vllm_runner.py  vLLM AsyncLLMEngine integration (model loading deferred)
  metrics/
    latency.py      TTFT / TPOT / request latency percentiles
    throughput.py   system tok/s vs per-request tok/s
    memory.py       peak GPU memory accounting
    cost.py         $/M tokens with per-hardware hourly pricing
  stats/
    analysis.py     bootstrap CIs, Mann-Whitney U, Welch's t, Cohen's d
  quality/
    judge.py        LLM-as-judge regression check (scaffold)
configs/
  smoke.yaml        single-config smoke test
  sweep.yaml        illustrative sweep across quant × spec × batch × context
notebooks/
  kaggle_runner.ipynb   thin wrapper for running on Kaggle T4
results/
  runs/             raw JSON per (config, repeat) — gitignored by default
  summary/          aggregated tables for the writeup
```

## Methodology

See [methodology.md](methodology.md) for: warmup protocol, repeat counts and
seed handling, prompt sampling, statistical tests, GPU-memory headroom rules,
and an honest list of limitations of the current sweep.

## Status

This repo currently ships the harness skeleton. The next milestones are:

- [x] Config schema, metrics math, stats, prompt sampling, MockRunner, CLI
- [ ] vLLM runner: model loading + async streaming (deferred until GPU host)
- [ ] First full sweep on Kaggle T4
- [ ] Writeup + headline chart
- [ ] LLM-as-judge quality check

## License

MIT
