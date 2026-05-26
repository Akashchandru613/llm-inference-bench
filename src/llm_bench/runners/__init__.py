from .base import Runner, RunOutput
from .mock_runner import MockRunner
from .vllm_runner import VLLMRunner

__all__ = ["Runner", "RunOutput", "MockRunner", "VLLMRunner", "get_runner"]


def get_runner(name: str) -> type[Runner]:
    table: dict[str, type[Runner]] = {"mock": MockRunner, "vllm": VLLMRunner}
    if name not in table:
        raise KeyError(f"unknown runner '{name}'; choose from {sorted(table)}")
    return table[name]
