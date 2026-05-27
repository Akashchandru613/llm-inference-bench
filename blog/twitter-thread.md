# Twitter / X thread

8 tweets. Post them one-by-one as replies under your own tweet (a "thread"). Don't bulk-post — type each in turn so each one threads correctly.

Character counts after each tweet (Twitter's hard limit is 280 for free users, 25k for Premium — these all fit in 280).

---

## Tweet 1 — the hook

> Common LLM-cost advice: "quantize first."
>
> I measured the actual cost-per-token across batch sizes and context lengths on Mistral-7B-AWQ + free-tier T4.
>
> Batching gives 5.7× cheaper tokens. Quantization is just the prerequisite.
>
> Thread + numbers ↓

*(258 chars)*

---

## Tweet 2 — the headline numbers

> The data, in one table:
>
> bs=1, 256-tok ctx → 23.4 tok/s, $4.15/M
> bs=4, 256-tok ctx → 69.5 tok/s, $1.40/M
> bs=16, 256-tok ctx → 133.2 tok/s, **$0.73/M**
> bs=1, 8192-tok ctx → 6.1 tok/s, $16.00/M
>
> Same model. Same quantization. Same hardware.

*(257 chars)*

---

## Tweet 3 — finding 1 (the batching lever)

> Finding 1: Batching is the dominant cost lever on a T4.
>
> bs=1 → bs=16 at fixed context:
> - throughput: 5.7× more
> - cost: 5.7× cheaper
> - TTFT: 6.8× worse
>
> If your workload tolerates ~3 sec TTFT, you ship far more tokens per dollar. If you're serving chat, you can't.

*(280 chars)*

---

## Tweet 4 — finding 2 (long context cliff)

> Finding 2: Long context at bs=1 falls off a cliff.
>
> 8K input @ bs=1 on T4:
> 6.1 tok/s
> 14.6 SEC TTFT
> $16/M tokens — 22× more expensive than the cheapest cell
>
> Prefill dominates. This is where paged-attention tuning, prefix caching, or A10G/L4 stop being optional.

*(279 chars)*

---

## Tweet 5 — finding 3 (the surprising one)

> Finding 3 (the one I didn't expect):
>
> Going short → medium context multiplies TTFT by **5.4× at bs=1** but only **3.8× at bs=4**.
>
> vLLM's batched prefill amortizes context cost across slots. As your context grows, batching gets MORE valuable, not less.
>
> Counter-intuitive.

*(279 chars)*

---

## Tweet 6 — methodology + repo

> Methodology: ShareGPT-sampled prompts (deterministic, hash-fingerprinted), warmup of 4 + 50 measured prompts/config, TTFT from first streamed token, every record carries git SHA + GPU + driver + lib versions.
>
> Harness, methodology doc, Kaggle notebook:
> github.com/Akashchandru613/llm-inference-bench

*(280 chars — adjust the URL if Twitter strips trailing chars)*

---

## Tweet 7 — what's missing (the honesty tweet)

> What I'd do differently:
> - Add repeats for noise estimates (single seed per cell)
> - Tokenize prompts during sampling, not heuristic
> - FP16 baseline (needs L4/A10G — won't fit on T4)
> - Speculative decoding (needs vLLM 0.6+ for vocab-compatible drafts)
> - Run the LLM-judge quality check

*(280 chars)*

---

## Tweet 8 — the close + ask

> If you've measured anything that contradicts this, I want to know. The harness is set up to be re-runnable — bring data.
>
> Full writeup: [link to Substack/Medium post once published]
>
> Tag the inference-eng folks you trust. @vllm_project @anyscalecompute @modal_labs

*(275 chars — replace last line with actual handles you want to tag)*

---

## Notes on posting

- **Best time**: 9–11am PT on a weekday (when inference engineers are caffeinated and scrolling)
- **First reply tip**: comment on your own thread with one extra detail or a chart link — that bumps the thread above newer posts in the algorithm
- **Don't tag too aggressively**: pick 2–3 handles you genuinely want to engage. Spammy tagging gets you muted
- **If a tweet gets traction**: pin the thread to your profile while the buzz is alive
- **Crosspost the thread to LinkedIn** (same text, but you can be slightly more formal): LinkedIn surfaces this kind of content well for the recruiter/eng-manager audience, which is exactly your target
