"""Result record IO.

Each run writes exactly one JSON file under <output_dir>/<config_fingerprint>/
<timestamp>.json. The schema is stable so downstream analysis can union runs
across machines without bespoke parsing.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig
from .env import EnvSnapshot
from .metrics.latency import LatencySummary, RequestMeasurement
from .metrics.memory import MemorySummary
from .metrics.throughput import ThroughputSummary


@dataclass(frozen=True)
class RunRecord:
    config: RunConfig
    config_fingerprint: str
    env: EnvSnapshot
    prompts_fingerprint: str
    runner: str
    repeat_index: int
    latency: LatencySummary
    throughput: ThroughputSummary
    memory: MemorySummary | None
    cost_per_million_output_tokens_usd: float | None
    per_request: list[RequestMeasurement]
    status: str = "ok"
    error: str | None = None


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def write_record(record: RunRecord, output_dir: Path) -> Path:
    target_dir = output_dir / record.config_fingerprint
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = record.env.timestamp_utc.replace(":", "-")
    path = target_dir / f"{ts}__r{record.repeat_index}.json"
    payload = _to_jsonable(record)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
