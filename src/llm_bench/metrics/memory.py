"""GPU memory accounting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySummary:
    peak_allocated_gib: float
    peak_reserved_gib: float
    device_total_gib: float

    @property
    def headroom_gib(self) -> float:
        return self.device_total_gib - self.peak_reserved_gib


def capture_memory() -> MemorySummary:
    # Filled in at runner time when torch is available; the harness imports lazily
    # so this module stays import-safe in a no-torch environment.
    import torch

    if not torch.cuda.is_available():
        return MemorySummary(0.0, 0.0, 0.0)
    gib = 1024 ** 3
    props = torch.cuda.get_device_properties(0)
    return MemorySummary(
        peak_allocated_gib=torch.cuda.max_memory_allocated() / gib,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / gib,
        device_total_gib=props.total_memory / gib,
    )
