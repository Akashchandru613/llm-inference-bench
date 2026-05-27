#!/usr/bin/env python3
"""Generate the two headline charts for the blog post and README.

Reads results/summary/summary.json (produced by `make analyze`) and writes:
  - docs/charts/throughput_vs_batch.png  (chart 1: batch sweep at short context)
  - docs/charts/ttft_vs_context.png      (chart 2: context sweep at bs=1 vs bs=4)

Run with: python scripts/make_charts.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "summary" / "summary.json"
OUT_DIR = REPO / "docs" / "charts"

CONTEXT_TOKENS = {"short": 256, "medium": 2048, "long": 8192}
NAME_RE = re.compile(r"bs(\d+)-(short|medium|long)")


def parse(name: str) -> tuple[int, str] | None:
    m = NAME_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def load_rows() -> list[dict]:
    data = json.loads(SUMMARY.read_text())
    rows = []
    for r in data:
        parsed = parse(r["name"])
        if not parsed:
            continue
        batch, bucket = parsed
        rows.append({
            "name": r["name"],
            "batch": batch,
            "bucket": bucket,
            "context_tokens": CONTEXT_TOKENS[bucket],
            "tps": r["system_tps_mean"],
            "ttft_ms": r["ttft_p50_ms_mean"],
            "cost": r.get("cost_per_M_tokens_usd"),
        })
    return rows


def chart_throughput_vs_batch(rows):
    short = sorted(
        [r for r in rows if r["bucket"] == "short"], key=lambda r: r["batch"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    batches = [r["batch"] for r in short]
    tps = [r["tps"] for r in short]

    ax.plot(batches, tps, marker="o", markersize=11, linewidth=2.5, color="#2E86AB")
    for b, t in zip(batches, tps):
        ax.annotate(
            f"{t:.1f} tok/s",
            xy=(b, t),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )

    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("System throughput (tokens/sec)", fontsize=12)
    ax.set_title(
        "Batching gives 5.7x the tokens/sec on a T4\n(Mistral-7B-AWQ, 256-token context, vLLM 0.5.5)",
        fontsize=13,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels(batches)
    ax.set_ylim(0, max(tps) * 1.2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "throughput_vs_batch.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_ttft_vs_context(rows):
    bs1 = sorted([r for r in rows if r["batch"] == 1], key=lambda r: r["context_tokens"])
    bs4 = sorted([r for r in rows if r["batch"] == 4], key=lambda r: r["context_tokens"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        [r["context_tokens"] for r in bs1],
        [r["ttft_ms"] for r in bs1],
        marker="o",
        markersize=11,
        linewidth=2.5,
        color="#A23B72",
        label="batch = 1",
    )
    ax.plot(
        [r["context_tokens"] for r in bs4],
        [r["ttft_ms"] for r in bs4],
        marker="s",
        markersize=11,
        linewidth=2.5,
        color="#F18F01",
        label="batch = 4",
    )

    for r in bs1:
        ax.annotate(
            f"{r['ttft_ms']:.0f} ms",
            xy=(r["context_tokens"], r["ttft_ms"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#A23B72",
        )
    for r in bs4:
        ax.annotate(
            f"{r['ttft_ms']:.0f} ms",
            xy=(r["context_tokens"], r["ttft_ms"]),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#F18F01",
        )

    ax.set_xlabel("Input context length (tokens)", fontsize=12)
    ax.set_ylabel("TTFT p50 (milliseconds, lower is better)", fontsize=12)
    ax.set_title(
        "Context-induced TTFT grows sublinearly in batch size\n(Mistral-7B-AWQ on T4)",
        fontsize=13,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 2048, 8192])
    ax.set_xticklabels(["256\n(short)", "2,048\n(medium)", "8,192\n(long)"])
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "ttft_vs_context.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    p1 = chart_throughput_vs_batch(rows)
    p2 = chart_ttft_vs_context(rows)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
