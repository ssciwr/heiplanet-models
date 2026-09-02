"""Reusable command-line helpers for model pipeline runners."""

import argparse
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

T = TypeVar("T")


@dataclass
class PipelineRunResult:
    label: str
    status: str
    seconds: float
    output_path: str | None = None
    message: str | None = None
    traceback_text: str | None = None


YearRunResult = PipelineRunResult


class SkipYear(Exception):
    """Raise from a model pipeline when a year should be skipped."""


class SkipRun(Exception):
    """Raise from a model pipeline when the current labeled run should be skipped."""


class StageRunner:
    """Run named stages with timing and Rich status messages."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.stage_number = 0

    def __call__(self, stage_name: str, action: Callable[[], T]) -> T:
        self.stage_number += 1
        self.console.print(
            f"[yellow]Running Stage {self.stage_number}: {stage_name}...[/yellow]"
        )
        start = time.perf_counter()

        try:
            value = action()
        except Exception as exc:
            seconds = time.perf_counter() - start
            self.console.print(
                f"  [red][FAILED][/red] Stage {self.stage_number}: "
                f"{stage_name} ({seconds:.2f}s) - {exc}"
            )
            raise

        seconds = time.perf_counter() - start
        self.console.print(
            f"  [green][OK][/green] Stage {self.stage_number}: "
            f"{stage_name} ({seconds:.2f}s)"
        )
        return value


def stage_running(console: Console, idx: int, total: int, name: str) -> None:
    console.print(f"[yellow]Running Stage {idx}/{total}: {name}...[/yellow]")


def stage_success(
    console: Console, idx: int, total: int, name: str, seconds: float
) -> None:
    console.print(f"  [green][OK][/green] Stage {idx}/{total}: {name} ({seconds:.2f}s)")


def stage_failure(
    console: Console, idx: int, total: int, name: str, seconds: float, reason: str
) -> None:
    console.print(
        f"  [red][FAILED][/red] Stage {idx}/{total}: {name} ({seconds:.2f}s) - {reason}"
    )


def render_summary(console: Console, results: list) -> None:
    table = Table(title="Pipeline Run Summary")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Duration", style="white", justify="right")
    table.add_column("Message", style="white")
    for stage, status, seconds, message in results:
        color = {"ok": "green", "fail": "red"}.get(status, "white")
        table.add_row(
            stage,
            f"[{color}]{status}[/{color}]",
            f"{seconds:.2f}s",
            message or "-",
        )
    console.print(table)


def _build_parser(
    *,
    model_name: str,
    default_settings: str,
    default_initial_year: int,
    default_final_year: int,
    add_model_args: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {model_name}.")
    parser.add_argument("--settings", default=default_settings)
    parser.add_argument("--initial-year", type=int, default=default_initial_year)
    parser.add_argument("--final-year", type=int, default=default_final_year)
    add_model_args(parser)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _render_run_header(
    console: Console,
    model_name: str,
    args: argparse.Namespace,
    display_args: Callable[[argparse.Namespace], dict[str, object]] | None,
) -> None:
    table = Table(show_header=False)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    header_values = display_args(args) if display_args is not None else vars(args)
    for key, value in header_values.items():
        if key == "settings":
            value = str(Path(value).resolve())
        table.add_row(key.replace("_", " "), str(value))

    console.print(Panel(table, title=model_name, border_style="cyan"))


def _render_final_summary(
    console: Console,
    results: list[PipelineRunResult],
    total_seconds: float,
    verbose: bool,
    label_heading: str = "Run",
) -> None:
    table = Table(title="Run Summary")
    table.add_column(label_heading, justify="right")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Message")

    for result in results:
        status_color = {
            "completed": "green",
            "skipped": "yellow",
            "failed": "red",
        }[result.status]
        table.add_row(
            result.label,
            f"[{status_color}]{result.status}[/{status_color}]",
            f"{result.seconds:.2f}s",
            result.message or result.output_path or "-",
        )

    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)

    console.print(table)
    console.print(
        f"[cyan]Completed:[/cyan] {completed}  "
        f"[cyan]Skipped:[/cyan] {skipped}  "
        f"[cyan]Failed:[/cyan] {failed}  "
        f"[cyan]Total:[/cyan] {total_seconds:.2f}s"
    )

    if verbose:
        for result in results:
            if result.traceback_text:
                console.print(
                    Panel(
                        result.traceback_text,
                        title=f"Traceback for {label_heading.lower()} {result.label}",
                        border_style="red",
                    )
                )


def _run_one_label(
    *,
    label: str,
    args: argparse.Namespace,
    settings: dict,
    console: Console,
    run_one: Callable[[str, argparse.Namespace, dict, StageRunner], Path | str | None],
) -> PipelineRunResult:
    run_start = time.perf_counter()
    console.rule(label)
    stage = StageRunner(console)

    try:
        output_path = run_one(label, args, settings, stage)
    except (SkipRun, SkipYear) as exc:
        return PipelineRunResult(
            label=label,
            status="skipped",
            seconds=time.perf_counter() - run_start,
            message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return PipelineRunResult(
            label=label,
            status="failed",
            seconds=time.perf_counter() - run_start,
            message=f"Pipeline failed: {exc}",
            traceback_text=traceback.format_exc(),
        )

    resolved_output_path = str(Path(output_path).resolve()) if output_path else None
    if resolved_output_path:
        console.print(f"  [cyan]Output:[/cyan] {resolved_output_path}")

    return PipelineRunResult(
        label=label,
        status="completed",
        seconds=time.perf_counter() - run_start,
        output_path=resolved_output_path,
    )


def run_labeled_pipeline_cli(
    *,
    model_name: str,
    default_settings: str,
    add_model_args: Callable[[argparse.ArgumentParser], None],
    read_settings: Callable[[str], dict],
    run_labels: Callable[[argparse.Namespace, dict], list[str]],
    run_one: Callable[[str, argparse.Namespace, dict, StageRunner], Path | str | None],
    display_args: Callable[[argparse.Namespace], dict[str, object]] | None = None,
    validate_args: Callable[[argparse.Namespace], None] | None = None,
    label_heading: str = "Run",
) -> None:
    """Run any labeled pipeline with shared CLI parsing, Rich output, and summaries."""

    parser = argparse.ArgumentParser(description=f"Run {model_name}.")
    parser.add_argument("--settings", default=default_settings)
    add_model_args(parser)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    console = Console()
    run_start = time.perf_counter()

    try:
        settings = read_settings(args.settings)
    except Exception as exc:
        console.print(f"[red]Could not read settings:[/red] {exc}")
        if args.verbose:
            console.print(
                Panel(
                    traceback.format_exc(),
                    title="Settings traceback",
                    border_style="red",
                )
            )
        raise SystemExit(1) from exc

    try:
        if validate_args is not None:
            validate_args(args)
    except ValueError as exc:
        console.print(f"[red]Invalid command-line arguments:[/red] {exc}")
        raise SystemExit(2) from exc

    _render_run_header(console, model_name, args, display_args)

    results = [
        _run_one_label(
            label=label,
            args=args,
            settings=settings,
            console=console,
            run_one=run_one,
        )
        for label in run_labels(args, settings)
    ]

    _render_final_summary(
        console,
        results,
        total_seconds=time.perf_counter() - run_start,
        verbose=args.verbose,
        label_heading=label_heading,
    )

    if any(result.status == "failed" for result in results):
        raise SystemExit(1)


def run_pipeline_cli(
    *,
    model_name: str,
    default_settings: str,
    default_initial_year: int | None,
    default_final_year: int | None,
    add_model_args: Callable[[argparse.ArgumentParser], None],
    read_settings: Callable[[str], dict],
    run_year: Callable[[int, argparse.Namespace, dict, StageRunner], Path | str | None],
    display_args: Callable[[argparse.Namespace], dict[str, object]] | None = None,
    validate_args: Callable[[argparse.Namespace], None] | None = None,
) -> None:
    """Run a year-labeled model pipeline with shared CLI and Rich output."""

    def add_year_args(parser: argparse.ArgumentParser) -> None:
        if default_initial_year is not None:
            parser.add_argument(
                "--initial-year", type=int, default=default_initial_year
            )
        if default_final_year is not None:
            parser.add_argument("--final-year", type=int, default=default_final_year)
        add_model_args(parser)

    def year_labels(args: argparse.Namespace, settings: dict) -> list[str]:
        if hasattr(args, "initial_year") and hasattr(args, "final_year"):
            initial_year = args.initial_year
            final_year = args.final_year
        else:
            initial_year = settings["execution"]["initial_year"]
            final_year = settings["execution"]["final_year"]
        if final_year < initial_year:
            raise SystemExit(
                "--final-year must be greater than or equal to --initial-year"
            )
        return [str(year) for year in range(initial_year, final_year + 1)]

    def run_one_year(
        label: str,
        args: argparse.Namespace,
        settings: dict,
        stage: StageRunner,
    ) -> Path | str | None:
        return run_year(int(label), args, settings, stage)

    run_labeled_pipeline_cli(
        model_name=model_name,
        default_settings=default_settings,
        add_model_args=add_year_args,
        read_settings=read_settings,
        run_labels=year_labels,
        run_one=run_one_year,
        display_args=display_args,
        validate_args=validate_args,
        label_heading="Year",
    )
