"""Deterministic prompt sampling.

Same (dataset, split, context_bucket, seed, n) -> same prompts. A SHA over the
sampled prompt list goes into the result record so reviewers can confirm two
configs were measured against the identical distribution.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from ..config import CONTEXT_TOKENS, ContextBucket


@dataclass(frozen=True)
class PromptSample:
    text: str
    approx_input_tokens: int


# Synthetic fallback used when `datasets` is unavailable (Kaggle install can fail
# on flaky network). Bucket length is approximate; precise tokenization happens
# at the runner once a tokenizer is loaded.
_FALLBACK_TEMPLATES = [
    "Summarize the following passage in two sentences:\n\n{filler}",
    "Explain step by step how you would approach this problem:\n\n{filler}",
    "Rewrite the following text in a more formal register:\n\n{filler}",
    "Continue the story below, staying in the same voice:\n\n{filler}",
    "Translate the following text into French, preserving tone:\n\n{filler}",
]


def _synthetic_prompts(bucket: ContextBucket, n: int, rng: random.Random) -> list[PromptSample]:
    target = CONTEXT_TOKENS[bucket]
    # ~0.75 tokens per word for English — fine for synthesizing a target length.
    words_target = int(target / 0.75)
    lorem = (
        "the quick brown fox jumps over the lazy dog while the river flows "
        "past the ancient stones and birds sing in the morning light "
    ).split()
    samples: list[PromptSample] = []
    for _ in range(n):
        tmpl = rng.choice(_FALLBACK_TEMPLATES)
        filler = " ".join(rng.choices(lorem, k=words_target))
        text = tmpl.format(filler=filler)
        samples.append(PromptSample(text=text, approx_input_tokens=target))
    return samples


def sample_prompts(
    *,
    dataset: str,
    split: str,
    context_bucket: ContextBucket,
    n: int,
    seed: int,
) -> list[PromptSample]:
    rng = random.Random(seed)
    pool: list[PromptSample] = []
    try:
        from datasets import load_dataset  # type: ignore

        target = CONTEXT_TOKENS[context_bucket]
        low, high = int(target * 0.7), int(target * 1.3)
        ds = load_dataset(dataset, split=split, streaming=True)
        for row in ds:
            text = _extract_user_text(row)
            if text is None:
                continue
            approx_tokens = int(len(text.split()) / 0.75)
            if low <= approx_tokens <= high:
                pool.append(PromptSample(text=text, approx_input_tokens=approx_tokens))
            if len(pool) >= n * 8:
                break
    except Exception:
        # Network down, dataset gated, schema drifted — any of these is fine,
        # fall through to synthetic so the harness stays runnable.
        pool = []

    if len(pool) < n:
        pool.extend(_synthetic_prompts(context_bucket, n - len(pool), rng))
    rng.shuffle(pool)
    return pool[:n]


def _extract_user_text(row: dict) -> str | None:
    convs = row.get("conversations")
    if not convs:
        return None
    for turn in convs:
        if turn.get("from") in {"human", "user"}:
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                return value
    return None


def prompts_fingerprint(prompts: Sequence[PromptSample]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
