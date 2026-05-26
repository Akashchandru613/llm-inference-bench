"""Cost accounting.

Hourly prices are best-effort spot/list values from major clouds. They make
cost-per-token comparisons meaningful even when the actual run was on free-tier
hardware (Kaggle T4 = $0 wall-cost, but the cost-equivalent perspective is
what readers care about). Update as cloud prices change.
"""
from __future__ import annotations

HARDWARE_HOURLY_USD: dict[str, float] = {
    "T4": 0.35,
    "L4": 0.71,
    "A10G": 0.75,
    "A100-40GB": 1.50,
    "A100-80GB": 2.20,
    "H100-80GB": 3.50,
}


def cost_per_million_tokens(
    wall_time_s: float,
    total_output_tokens: int,
    hardware: str,
    *,
    overrides: dict[str, float] | None = None,
) -> float:
    table = {**HARDWARE_HOURLY_USD, **(overrides or {})}
    if hardware not in table:
        raise KeyError(f"unknown hardware '{hardware}'; pass via overrides")
    if total_output_tokens <= 0:
        return float("nan")
    hours = wall_time_s / 3600.0
    cost_usd = hours * table[hardware]
    return cost_usd / (total_output_tokens / 1_000_000)
