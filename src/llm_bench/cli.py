"""Command-line interface.

  llm-bench run     --config configs/smoke.yaml --runner mock --output results/runs
  llm-bench sweep   --config configs/sweep.yaml --runner vllm --output results/runs
  llm-bench analyze --input results/runs --output results/summary
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import RunConfig, SweepConfig, load_run_config
from .env import capture_env
from .metrics import cost_per_million_tokens, summarize_latency, summarize_throughput
from .metrics.memory import MemorySummary
from .prompts import sample_prompts
from .prompts.loader import prompts_fingerprint
from .records import RunRecord, read_record, write_record
from .runners import get_runner

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _execute(
    config: RunConfig,
    runner_name: str,
    output_dir: Path,
    *,
    hardware_override: str | None,
) -> list[Path]:
    runner_cls = get_runner(runner_name)
    written: list[Path] = []

    prompts_total = config.controls.num_prompts + config.controls.num_warmup
    prompts = sample_prompts(
        dataset=config.prompts.dataset,
        split=config.prompts.split,
        context_bucket=config.prompts.context_bucket,
        n=prompts_total,
        seed=config.controls.seed,
    )
    warmup_prompts = prompts[: config.controls.num_warmup]
    measure_prompts = prompts[config.controls.num_warmup :]
    fingerprint_prompts = prompts_fingerprint(measure_prompts)
    config_fp = config.fingerprint()

    for repeat in range(config.controls.repeats):
        env = capture_env()
        runner = runner_cls(config)
        memory: MemorySummary | None = None
        try:
            runner.warmup(warmup_prompts)
            output = runner.run(measure_prompts)
            try:
                from .metrics.memory import capture_memory

                memory = capture_memory()
            except ImportError:
                memory = None
        finally:
            runner.shutdown()

        latency = summarize_latency(output.measurements)
        throughput = summarize_throughput(output.measurements, output.wall_time_s)
        hardware = hardware_override or output.hardware

        cost: float | None = None
        try:
            cost = cost_per_million_tokens(
                wall_time_s=output.wall_time_s,
                total_output_tokens=throughput.total_output_tokens,
                hardware=hardware,
            )
        except KeyError:
            cost = None

        record = RunRecord(
            config=config,
            config_fingerprint=config_fp,
            env=env,
            prompts_fingerprint=fingerprint_prompts,
            runner=runner_name,
            repeat_index=repeat,
            latency=latency,
            throughput=throughput,
            memory=memory,
            cost_per_million_output_tokens_usd=cost,
            per_request=output.measurements,
        )
        path = write_record(record, output_dir)
        written.append(path)
        console.print(f"[green]wrote[/green] {path}")
    return written


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a single-run YAML config."),
    runner: str = typer.Option("mock", "--runner", "-r", help="mock | vllm"),
    output: Path = typer.Option(Path("results/runs"), "--output", "-o"),
    hardware: str | None = typer.Option(None, "--hardware", help="Override hardware label for cost accounting."),
) -> None:
    """Execute a single config and write a result record."""
    cfg = load_run_config(config)
    paths = _execute(cfg, runner, output, hardware_override=hardware)
    console.print(f"[bold]wrote {len(paths)} record(s)[/bold]")


@app.command()
def sweep(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a sweep YAML config."),
    runner: str = typer.Option("vllm", "--runner", "-r"),
    output: Path = typer.Option(Path("results/runs"), "--output", "-o"),
    hardware: str | None = typer.Option(None, "--hardware"),
    skip_failed: bool = typer.Option(True, "--skip-failed/--stop-on-fail"),
) -> None:
    """Execute every run in the sweep, one at a time."""
    sweep_cfg = SweepConfig.from_yaml(config)
    console.print(f"[bold]sweep:[/bold] {sweep_cfg.name} — {len(sweep_cfg.runs)} run(s)")
    failures: list[tuple[str, str]] = []
    for run_cfg in sweep_cfg.runs:
        console.rule(run_cfg.name)
        try:
            _execute(run_cfg, runner, output, hardware_override=hardware)
        except Exception as exc:
            failures.append((run_cfg.name, repr(exc)))
            console.print(f"[red]FAILED[/red] {run_cfg.name}: {exc}")
            if not skip_failed:
                raise
    if failures:
        console.print(f"[red]{len(failures)} run(s) failed[/red]")
        for name, err in failures:
            console.print(f"  - {name}: {err}")


@app.command()
def analyze(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(Path("results/summary"), "--output", "-o"),
) -> None:
    """Aggregate result records into a per-config summary table."""
    records: list[dict] = []
    for path in sorted(input.rglob("*.json")):
        records.append(read_record(path))
    if not records:
        console.print(f"[yellow]no records under {input}[/yellow]")
        raise typer.Exit(1)

    by_fp: dict[str, list[dict]] = {}
    for rec in records:
        by_fp.setdefault(rec["config_fingerprint"], []).append(rec)

    rows = []
    for fp, group in by_fp.items():
        sample = group[0]
        tps = [g["throughput"]["system_output_tps"] for g in group]
        ttft = [g["latency"]["ttft_p50_ms"] for g in group]
        rows.append({
            "name": sample["config"]["name"],
            "fingerprint": fp,
            "n_repeats": len(group),
            "system_tps_mean": sum(tps) / len(tps),
            "system_tps_stdev": _stdev(tps),
            "ttft_p50_ms_mean": sum(ttft) / len(ttft),
            "cost_per_M_tokens_usd": sample.get("cost_per_million_output_tokens_usd"),
        })

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True))

    table = Table(title="Benchmark summary")
    table.add_column("name")
    table.add_column("n", justify="right")
    table.add_column("sys tok/s", justify="right")
    table.add_column("±", justify="right")
    table.add_column("TTFT p50 (ms)", justify="right")
    table.add_column("$/M tok", justify="right")
    for r in rows:
        table.add_row(
            r["name"],
            str(r["n_repeats"]),
            f"{r['system_tps_mean']:.1f}",
            f"{r['system_tps_stdev']:.1f}",
            f"{r['ttft_p50_ms_mean']:.1f}",
            "—" if r["cost_per_M_tokens_usd"] is None else f"{r['cost_per_M_tokens_usd']:.3f}",
        )
    console.print(table)
    console.print(f"[green]wrote[/green] {summary_path}")


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


if __name__ == "__main__":
    app()
