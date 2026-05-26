# Methodology

This document records the choices behind the benchmark so a reader can decide
whether to trust the numbers. It is meant to be revised as the sweep is run.

## What we measure

Per request:

- **TTFT** — wall-clock seconds from request submission to the first generated
  token. Measured from the streaming output, not from `generate()` returning.
- **TPOT** — `(end_time − first_token_time) / (output_tokens − 1)`. Requests
  with fewer than 2 output tokens contribute to TTFT only and are excluded
  from TPOT.
- **Request latency** — `end_time − submit_time`.

Per run:

- **System throughput (tokens/sec)** — total output tokens divided by the wall
  time of the measured window. Reflects what a server actually delivers.
- **Per-request throughput** — average of `output_tokens / request_latency`
  across requests. Diverges from system throughput under batching; we report
  both.
- **Peak GPU memory** — `torch.cuda.max_memory_allocated/reserved` after
  warmup, before the measured window.
- **Cost per million output tokens** — `(wall_time_hours × $/hr) ÷
  (output_tokens / 1e6)`, using the hourly rates in `metrics/cost.py`. These
  are list/spot prices for the equivalent cloud GPU; the actual run can be on
  free-tier Kaggle and the comparison still means something.

## Reproducibility controls

- **Warmup**: `controls.num_warmup` requests are submitted with timings
  discarded before the measured window begins. The KV cache, CUDA graphs,
  and any JIT compilation are warm before any number we report.
- **Repeats**: Each config runs `controls.repeats` times, each producing its
  own result record. We report mean ± stdev across repeats in the summary.
- **Prompt determinism**: `prompts.sample_prompts(seed=...)` is deterministic.
  The SHA-16 fingerprint of the sampled prompt list is captured in every
  result record so two configs can be confirmed to have been measured against
  the identical distribution.
- **Environment capture**: every record records git SHA, dirty bit, Python /
  torch / vllm / transformers versions, GPU model, driver, and CUDA version.
- **Config fingerprint**: a hash over benchmark-relevant config fields. Runs
  with the same fingerprint are comparable replicates; the on-disk layout
  groups records by fingerprint.

## Statistical tests

We default to non-parametric tests because TTFT and TPOT distributions are
heavy-tailed:

- **Bootstrap CIs over the median** (2000 resamples) — the band a reader
  should expect a re-run to land inside.
- **Mann-Whitney U** for pairwise comparisons (e.g. spec decoding on vs off).
- **Cohen's d** as the effect size — a small p-value with d < 0.2 is not
  interesting.
- **Welch's t** is exposed for cases where Gaussian residuals are justified.

A finding is reported as "spec decoding helps / does not help" only when the
test rejects at p < 0.01 *and* |d| ≥ 0.5 *and* the bootstrap CI on the median
delta excludes zero. Otherwise it is reported as inconclusive.

## Sweep design

Axes (see `configs/sweep.yaml` for an illustrative slice):

- Model — Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (a third can be added).
- Quantization — `fp16`, `awq` (4-bit), and `gptq` if we choose to extend.
- Speculative decoding — off / on with Qwen2.5-0.5B-Instruct as the draft.
- Batch size — 1, 4, 16, 64.
- Context length — short (256), medium (2048), long (8192 input tokens).

The full Cartesian product is large; we expect to publish a *T4-feasible*
slice (AWQ at most batch sizes, FP16 only at low batch / short context) and
document which cells were skipped because they would OOM.

## Hardware

Primary target: Kaggle T4 (sm_75, 16 GB, 30 hours/week free). Cost numbers
use $0.35/hr — the GCP spot list price for `n1-standard` + T4. A future run
on A10G (Modal free credits) is planned.

## Quality regression check

Throughput numbers mean little if 4-bit quantization is silently destroying
output quality. For each candidate vs baseline pair on the *same* prompts,
an LLM judge pairwise-scores outputs. The candidate's win/tie/loss rate
versus the baseline is reported alongside latency and cost. Scaffold lives
in `src/llm_bench/quality/judge.py`; the judge client is the next
implementation milestone.

## Known limitations (honest list)

- The current sweep uses `max_new_tokens=128–256`. Real chat workloads have
  longer-tailed output lengths; a follow-up should sample output lengths
  from ShareGPT.
- We treat prompts independently. Real serving exhibits prompt similarity
  and prefix-cache hits; PagedAttention's prefix caching is enabled but we
  don't engineer for it.
- Quality is checked with an LLM judge, which is itself noisy. The judge
  win-rate is a sanity check, not a replacement for task-specific
  evaluations on math/code/reasoning.
- We run on a single GPU. Tensor / pipeline parallelism interactions are
  out of scope for v1.
- The vLLM async engine integration is the next implementation step;
  `MockRunner` currently exists only to exercise the harness pipeline.

## Things I would do differently next time

(Section reserved — filled in after the first sweep. The point of writing
this section is to look the failure modes in the face.)
