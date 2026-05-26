"""Capture environment metadata so a result file is self-describing."""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class EnvSnapshot:
    timestamp_utc: str
    git_sha: str
    git_dirty: bool
    python_version: str
    platform: str
    gpu_name: str | None
    gpu_driver: str | None
    cuda_version: str | None
    torch_version: str | None
    vllm_version: str | None
    transformers_version: str | None
    env_vars: dict[str, str] = field(default_factory=dict)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _import_version(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except ImportError:
        return None


def _git_sha() -> tuple[str, bool]:
    sha = _run(["git", "rev-parse", "HEAD"]) or "unknown"
    status = _run(["git", "status", "--porcelain"])
    dirty = bool(status)
    return sha, dirty


def _gpu_info() -> tuple[str | None, str | None, str | None]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ])
    if not out:
        return None, None, _import_version("torch") and None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if len(parts) > 0 else None
    driver = parts[1] if len(parts) > 1 else None
    cuda = None
    try:
        import torch  # type: ignore

        cuda = getattr(torch.version, "cuda", None)
    except ImportError:
        pass
    return name, driver, cuda


def capture_env(extra_env_keys: list[str] | None = None) -> EnvSnapshot:
    sha, dirty = _git_sha()
    gpu_name, gpu_driver, cuda = _gpu_info()
    keys = extra_env_keys or []
    return EnvSnapshot(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=sha,
        git_dirty=dirty,
        python_version=platform.python_version(),
        platform=platform.platform(),
        gpu_name=gpu_name,
        gpu_driver=gpu_driver,
        cuda_version=cuda,
        torch_version=_import_version("torch"),
        vllm_version=_import_version("vllm"),
        transformers_version=_import_version("transformers"),
        env_vars={k: os.environ[k] for k in keys if k in os.environ},
    )
