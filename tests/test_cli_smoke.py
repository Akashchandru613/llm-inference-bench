"""End-to-end smoke: CLI runs MockRunner and writes a parseable result file."""
import json
from pathlib import Path

from typer.testing import CliRunner

from llm_bench.cli import app

REPO = Path(__file__).resolve().parents[1]


def test_cli_run_writes_record(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--config", str(REPO / "configs" / "smoke.yaml"),
            "--runner", "mock",
            "--output", str(tmp_path),
            "--hardware", "T4",
        ],
    )
    assert result.exit_code == 0, result.output
    written = list(tmp_path.rglob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["runner"] == "mock"
    assert payload["latency"]["n"] == 16
    assert payload["throughput"]["system_output_tps"] > 0
    assert payload["cost_per_million_output_tokens_usd"] is not None
