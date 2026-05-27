# Substack version

Paste body verbatim into the Substack editor. Title and subtitle go in their respective fields.

---

**Title:** On a free-tier T4, batching matters more than your quantization choice

**Subtitle:** I measured Mistral-7B-AWQ across batch sizes and context lengths on a single T4. Batching dropped cost-per-token by 5.7×. Quantization, the optimization everyone leads with, was just the prerequisite.

**Tags (top of editor):** `LLM`, `inference`, `vLLM`, `benchmarks`, `AI engineering`

**Header image suggestion:** Screenshot of the headline table rendered from the README (works as a thumbnail), or a simple matplotlib chart of throughput vs batch size.

---

## Body

A common piece of advice for "make LLM inference cheap" is *quantize first*. AWQ 4-bit drops weight memory ~4×, and the inference papers that introduced it lead with throughput numbers measured at batch size 1. Reasonable, except: in real serving you don't get to pick batch size 1.

So I ran the experiment that wasn't easy to find pre-existing numbers for: hold the model and quantization fixed, sweep batch size and context length, and look at **cost per million output tokens** as the bottom-line number. Free-tier hardware (Kaggle's T4, 16 GB), open-weight model (Mistral-7B-Instruct-v0.2 in AWQ), vLLM 0.5.5 as the serving stack.

The headline:

| config                          | system tok/s | TTFT p50 (ms) | $ / M tokens |
| ------------------------------- | -----------: | ------------: | -----------: |
| AWQ, batch=1, 256-tok context   |         23.4 |           547 |        $4.15 |
| AWQ, batch=4, 256-tok context   |         69.5 |         1,599 |        $1.40 |
| **AWQ, batch=16, 256-tok**      |    **133.2** |         3,722 |    **$0.73** |
| AWQ, batch=1, 2,048-tok context |         15.4 |         2,961 |        $6.32 |
| AWQ, batch=4, 2,048-tok context |         26.7 |         6,059 |        $3.64 |
| AWQ, batch=1, 8,192-tok context |          6.1 |        14,576 |       $16.00 |

Cost column uses T4 list price ($0.35/hr). Kaggle itself is free; the per-token figure is meaningful as a cloud-equivalent.

Three findings worth your time.

## 1. Batching is the dominant cost lever

Going from batch=1 to batch=16 at fixed short context:

- **System throughput**: 23.4 → 133.2 tok/s — 5.7× more
- **Cost per million tokens**: $4.15 → $0.73 — 5.7× cheaper
- **TTFT p50**: 547 → 3,722 ms — 6.8× worse

Throughput and cost move in lockstep because cost is just `wall_time × $/hr ÷ output_tokens`, which at fixed hardware collapses to `1 / throughput`. The interesting part is the *latency tax*: a 3.7-second TTFT is acceptable for batch-style summarization or document analysis, but uncomfortable for chat. The serving regime you're targeting determines whether this 5.7× win is real for you.

This is also why "production inference is mostly about batching" is repeated so often by people running real fleets. AWQ was already applied across the whole sweep, and yet most of the cost-per-token improvement on a T4 came from the batch knob. **AWQ is a prerequisite, not the lever.**

## 2. Long context at bs=1 falls off a cliff

The bs=1 row at 8,192-token context:

- 6.1 tok/s — under one-third of the bs=1 short-context throughput
- 14,576 ms TTFT — a full 15-second wait before the first token streams
- $16.00 per million output tokens — 22× more expensive than the cheapest cell

The cause is prefill: each request now spends ~12 seconds processing context before generating anything. At bs=1 there's nothing else for the GPU to do during that prefill, so the cost is absorbed entirely by one user's tokens. Single-stream long-context inference on a T4 is borderline unusable for real workloads.

This is the regime where smarter serving earns its name: paged-attention tuning, prefix caching for shared system prompts, longer batches to amortize prefill across users, or a hardware bump to A10G/L4 for FlashAttention-2. None of those are free, but $16-per-million is the floor anyone optimizing past should beat.

## 3. Context-induced TTFT growth is *sublinear* in batch size

The result I didn't expect:

| context        | TTFT @ bs=1 | TTFT @ bs=4 | ratio |
| -------------- | ----------: | ----------: | ----: |
| short (256)    |      547 ms |    1,599 ms |  2.9× |
| medium (2,048) |    2,961 ms |    6,059 ms |  2.0× |

Going from short to medium context multiplies TTFT by 5.4× at bs=1 but only 3.8× at bs=4. Naïve intuition says prefill cost is linear in context length and should add a constant per-token cost regardless of batch size. The data says otherwise: vLLM's batched prefill is doing real work to amortize FLOPs across the batch.

Translated to operational advice: **as your average context length grows, batching becomes more valuable, not less**. The instinct to drop batch size when contexts get long (to fit in memory) trades away exactly the lever that fights context-induced TTFT inflation.

## Methodology in 30 seconds

- Prompts sampled deterministically from ShareGPT, bucketed by approximate input length, hash-fingerprinted per result so two configs can be confirmed measured against the same distribution
- Warmup of 4 prompts (timings discarded) before the measured window
- 50 prompts per config (20-30 for long-context), single seed
- TTFT from request submission to first streamed token; TPOT over the decode window; throughput as total output tokens ÷ wall time
- Every record captures git SHA, GPU name, driver, CUDA version, library versions

Full harness, including the methodology doc and a Kaggle reproduction notebook: [github.com/Akashchandru613/llm-inference-bench](https://github.com/Akashchandru613/llm-inference-bench).

## What I'd do differently

1. **Add repeats.** Each config ran once. For the "batching helps 5.7×" finding the signal is huge, so a single run is fine — but cells near the edge would benefit from bootstrap CIs.
2. **Tokenize prompts during sampling.** My word-count heuristic underestimates tokens by 30-60% for chat text. Worked around it with a 2.5× `max_model_len` buffer; the right fix is to load the tokenizer and count for real.
3. **Add FP16 baseline.** Mistral-7B-FP16 doesn't fit on a T4 with usable KV cache. Needs A10G or L4 — the next sweep.
4. **Add speculative decoding.** Requires a vocab-compatible draft model (none official for Mistral) or ngram speculation (separate plumbing). v2.
5. **Run the LLM-as-judge quality check.** Harness has the module (Anthropic-backed, randomized A/B, prompt-cached rubric); I haven't run it on these outputs yet.

## A note on reproducibility on free-tier infra

For anyone tempted by the same exercise: Kaggle's free T4 is genuinely usable for benchmark work at the 7B scale, but **pin every version**. I lost an evening to a transitive dep (`pyairports`) being a squatter-stub on PyPI, and another hour to `transformers ≥ 4.45` changing the `rope_scaling` schema in a way vLLM 0.5.x rejects. Both documented in the repo. The "Known environment quirks" section of the README is worth reading before you reproduce.

---

*If you found something here that contradicts your own measurements, I'd genuinely like to know. The harness is set up to be re-runnable — bring data.*

---

*Akash Chandrasekharan is a Northeastern student working on inference systems. Code and methodology at [github.com/Akashchandru613/llm-inference-bench](https://github.com/Akashchandru613/llm-inference-bench).*
