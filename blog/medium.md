# Medium version

Medium's editor is slightly different: paste body verbatim, use the title/subtitle slots in the editor itself, add tags from the publish dialog.

---

**Title:** Batching beats quantization 5.7× for LLM inference on a free-tier T4

**Subtitle:** Most "make inference cheap" advice leads with quantization. I measured the actual cost-per-token across batch sizes and context lengths on Mistral-7B-AWQ. The batch knob is the dominant lever — quantization is just the prerequisite.

**Tags (5 max, Medium's limit):** `Machine Learning`, `LLM`, `Inference`, `Benchmarks`, `AI Engineering`

**Featured image suggestion:** Clean matplotlib chart of `system tokens/sec vs batch size` with two lines (short and medium context). The throughput-by-batch-size curve makes the headline finding visual.

---

## Body

A common piece of advice for "make LLM inference cheap" is *quantize first*. AWQ 4-bit drops weight memory by ~4×, and the inference papers that introduced it lead with throughput numbers measured at batch size 1. Reasonable, except: in real serving, you don't get to pick batch size 1.

So I ran the experiment that wasn't easy to find pre-existing numbers for. Hold the model and quantization fixed; sweep batch size and context length; look at **cost per million output tokens** as the bottom-line number. Free-tier hardware (Kaggle's T4, 16 GB). Open-weight model (Mistral-7B-Instruct-v0.2 in AWQ). vLLM 0.5.5 as the serving stack.

Here are the numbers:

![Headline results](headline-table.png)

(If the image doesn't load, the full table is in the GitHub README: [github.com/Akashchandru613/llm-inference-bench](https://github.com/Akashchandru613/llm-inference-bench))

Cost column uses T4 list price ($0.35/hr). Kaggle itself is free; the per-token figure is meaningful as a cloud-equivalent benchmark.

Three findings worth your time.

---

## Finding 1: Batching is the dominant cost lever

Going from batch=1 to batch=16 at fixed short context:

- System throughput: 23.4 → 133.2 tok/s — **5.7× more**
- Cost per million tokens: $4.15 → $0.73 — **5.7× cheaper**
- TTFT p50: 547 → 3,722 ms — 6.8× worse

> Throughput and cost move in lockstep because cost is just wall_time × $/hr ÷ output_tokens, which at fixed hardware collapses to 1 / throughput.

The interesting part is the *latency tax*. A 3.7-second TTFT is acceptable for batch-style summarization or document analysis. It's uncomfortable for chat. The serving regime you're targeting determines whether this 5.7× win is real for you.

This is also why "production inference is mostly about batching" is repeated so often by people running real fleets. AWQ was already applied across the whole sweep — and yet most of the cost-per-token improvement on a T4 came from the batch knob.

**AWQ is a prerequisite, not the lever.**

---

## Finding 2: Long context at batch=1 falls off a cliff

Look at the batch=1 row with 8,192-token context:

- 6.1 tok/s — under one-third of the bs=1 short-context throughput
- 14,576 ms TTFT — a full 15-second wait before the first token streams
- $16.00 per million output tokens — 22× more expensive than the cheapest cell

The cause is prefill. Each request now spends ~12 seconds processing context before generating anything. At batch=1 there's nothing else for the GPU to do during that prefill, so the cost is absorbed entirely by one user's tokens.

Single-stream long-context inference on a T4 is borderline unusable for real workloads. Full stop.

This is the regime where smarter serving earns its name: paged-attention tuning, prefix caching for shared system prompts, longer batches to amortize prefill across users, or a hardware bump to A10G/L4 to unlock FlashAttention-2. None of those are free, but $16-per-million is the floor anyone optimizing past should beat.

---

## Finding 3: Context-induced TTFT growth is *sublinear* in batch size

This was the result I didn't expect.

| context        | TTFT @ bs=1 | TTFT @ bs=4 | ratio |
| -------------- | ----------: | ----------: | ----: |
| short (256)    |      547 ms |    1,599 ms |  2.9× |
| medium (2,048) |    2,961 ms |    6,059 ms |  2.0× |

Going from short to medium context multiplies TTFT by 5.4× at batch=1 — but only 3.8× at batch=4.

Naïve intuition says prefill cost is linear in context length and should add a constant per-token cost regardless of batch size. The data says otherwise: **vLLM's batched prefill is doing real work to amortize FLOPs across the batch**.

Translated to operational advice: as your average context length grows, batching becomes *more* valuable, not less. The instinct to drop batch size when contexts get long (to fit in memory) trades away exactly the lever that fights context-induced TTFT inflation.

This is the kind of interaction effect that doesn't show up in any single published benchmark I could find, because most published numbers either fix batch size or fix context length. The interaction lives where most studies don't measure.

---

## Methodology in 30 seconds

- Prompts sampled deterministically from ShareGPT, bucketed by approximate input length, hash-fingerprinted per result so two configs can be confirmed measured against the same distribution
- Warmup of 4 prompts (timings discarded) before the measured window
- 50 prompts per config (20–30 for long-context to keep runtime reasonable), single seed per cell
- TTFT measured from request submission to first streamed token; TPOT computed across the decode-only window; throughput is total output tokens ÷ wall time
- Every result record captures git SHA, GPU name, driver version, CUDA version, library versions — so re-running on different hardware is a one-config-change exercise

Full harness, methodology doc, Kaggle reproduction notebook, and raw result JSONs: **[github.com/Akashchandru613/llm-inference-bench](https://github.com/Akashchandru613/llm-inference-bench)**.

---

## What I'd do differently in v2

1. **Add repeats.** Each config ran once. For the "batching helps 5.7×" finding the signal is huge, so a single run is fine. Cells closer to the noise floor (bs=4-medium vs bs=1-medium) deserve bootstrap CIs.
2. **Tokenize prompts during sampling.** My current word-count heuristic underestimates tokens by 30–60% for chat-style text. I worked around it with a 2.5× context buffer in `max_model_len`, but the right fix is to load the model's tokenizer and count for real.
3. **Add an FP16 baseline.** Mistral-7B-FP16 is 14 GB of weights — doesn't fit on a T4 with usable KV cache. The "AWQ vs FP16" comparison needs A10G or L4. That's the next sweep.
4. **Add speculative decoding.** vLLM 0.5.x speculative decoding requires a vocab-compatible draft model. No official tiny Mistral exists. Ngram speculative decoding needs separate plumbing.
5. **Run the LLM-as-judge quality check.** The harness has a judge module (Anthropic-backed, randomized A/B order, prompt-cached rubric), but I haven't run it on these outputs yet. Without it, the benchmark assumes AWQ doesn't tank quality on chat-style tasks — which is probably true but isn't measured here.

---

## A note on reproducibility on free-tier infra

For anyone tempted by the same exercise: Kaggle's free T4 is genuinely usable for benchmark work at the 7B parameter scale, but pin every version aggressively.

I lost an evening to a transitive dependency (`pyairports`) being a squatter-stub on PyPI, and another hour to `transformers ≥ 4.45` changing the `rope_scaling` schema in a way that vLLM 0.5.x rejects with an `AssertionError`. Both are documented in the repo as one-line workarounds.

The "Known environment quirks" section of the README is worth reading before you reproduce. The kind of friction that makes free-tier benchmarking hard isn't the compute; it's the dependency resolver.

---

*If you found a result here that contradicts something you've measured, I'd genuinely like to know. The harness is set up to be re-runnable. Bring data.*

---

*Code, methodology, and raw results: [github.com/Akashchandru613/llm-inference-bench](https://github.com/Akashchandru613/llm-inference-bench)*

*Reach out: [chandrasekharantha.a@northeastern.edu](mailto:chandrasekharantha.a@northeastern.edu)*
