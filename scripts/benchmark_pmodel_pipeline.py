"""Benchmark the main Pmodel pipeline stages with readable stage timing tables."""

import argparse
import logging
import resource
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from heiplanet_models.Pmodel.Pmodel_initial import (
    assemble_filepaths,
    check_all_paths_exist,
    load_all_data,
    load_all_data_daily,
    read_global_settings,
)
from heiplanet_models.Pmodel.Pmodel_ode import VALID_BACKENDS, solve_system
from heiplanet_models.Pmodel.Pmodel_output import (
    build_output_dataset,
    save_output_dataset,
)
from heiplanet_models.Pmodel.Pmodel_rates_birth import water_hatching
from heiplanet_models.Pmodel.Pmodel_rates_development import carrying_capacity

# -----------------------------------------------------------------------------
# Pmodel benchmark defaults
#
# To benchmark another model, change the imports above, these defaults, and the
# MODEL BENCHMARK PIPELINE section below. The timing/reporting helpers can stay.
# -----------------------------------------------------------------------------
DEFAULT_SETTINGS = "./src/heiplanet_models/Pmodel/global_settings.yaml"
DEFAULT_INITIAL_YEAR = 2024
DEFAULT_FINAL_YEAR = 2024
P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class StageMetric:
    year: int
    stage: str
    wall_seconds: float
    cpu_seconds: float
    rss_delta_kb: int
    io_read_bytes: int | None
    io_write_bytes: int | None
    status: str
    message: str


@dataclass
class YearResult:
    year: int
    status: str
    seconds: float
    output_path: str | None = None
    message: str | None = None


class BenchmarkStageError(Exception):
    def __init__(self, metric: StageMetric, original_error: Exception) -> None:
        super().__init__(str(original_error))
        self.metric = metric
        self.original_error = original_error


class BenchmarkStageRunner:
    def __init__(
        self,
        year: int,
        metrics: list[StageMetric],
        console: Console,
        show_progress: bool,
    ) -> None:
        self.year = year
        self.metrics = metrics
        self.console = console
        self.show_progress = show_progress
        self.stage_number = 0

    def __call__(
        self,
        stage_name: str,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        self.stage_number += 1
        if self.show_progress:
            self.console.print(
                f"[yellow]Running Stage {self.stage_number}: {stage_name}...[/yellow]"
            )

        metric, value, error = run_stage(self.year, stage_name, func, *args, **kwargs)
        self.metrics.append(metric)
        if error is not None:
            if self.show_progress:
                self.console.print(
                    f"  [red][FAILED][/red] Stage {self.stage_number}: "
                    f"{stage_name} ({metric.wall_seconds:.2f}s) - {error}"
                )
            raise BenchmarkStageError(metric, error)

        if self.show_progress:
            self.console.print(
                f"  [green][OK][/green] Stage {self.stage_number}: "
                f"{stage_name} ({metric.wall_seconds:.2f}s)"
            )
        return value


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark stage-level performance of the full Pmodel pipeline."
    )
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--initial-year", type=int, default=DEFAULT_INITIAL_YEAR)
    parser.add_argument("--final-year", type=int, default=DEFAULT_FINAL_YEAR)
    parser.add_argument(
        "--backend",
        choices=sorted(VALID_BACKENDS),
        default="legacy_optimized",
        help="ODE solver backend.",
    )
    parser.add_argument("--chunk-lon", type=int, default=288)
    parser.add_argument("--chunk-lat", type=int, default=144)
    parser.add_argument("--scipy-method", default="RK45")
    parser.add_argument("--scipy-rtol", type=float, default=1e-6)
    parser.add_argument("--scipy-atol", type=float, default=1e-9)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed year.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show traceback details for failures.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live stage progress messages.",
    )
    parser.add_argument(
        "--chunk-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Show scipy_chunked chunk progress. Defaults to enabled only when "
            "--backend scipy_chunked is selected."
        ),
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable live stage and chunk progress output.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Benchmark measurement helpers
#
# These helpers collect wall time, CPU time, memory, and process I/O. They are
# intentionally separate from the model pipeline so that developers can edit the
# ecological stages without touching benchmark mechanics.
# -----------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def kb_to_mb(kb: int) -> float:
    return kb / 1024.0


def bytes_to_mb(byte_count: int) -> float:
    return byte_count / (1024 * 1024)


def format_optional_bytes_as_mb(byte_count: int | None) -> str:
    if byte_count is None:
        return "-"
    return f"{bytes_to_mb(byte_count):.2f}MB"


def format_stage_status(status: str) -> str:
    color = "green" if status == "ok" else "red"
    return f"[{color}]{status}[/{color}]"


def format_year_status(status: str) -> str:
    color = {"completed": "green", "skipped": "yellow", "failed": "red"}[status]
    return f"[{color}]{status}[/{color}]"


def wall_percent(seconds: float, total_seconds: float) -> float:
    if total_seconds == 0:
        return 0.0
    return seconds / total_seconds * 100.0


def show_stage_progress(args: argparse.Namespace) -> bool:
    return bool(args.progress and not args.quiet_progress)


def show_chunk_progress(args: argparse.Namespace) -> bool:
    if args.quiet_progress:
        return False
    if args.chunk_progress is None:
        return args.backend == "scipy_chunked"
    return bool(args.chunk_progress)


def configure_progress_logging(console: Console, args: argparse.Namespace) -> None:
    """Route solver chunk progress logs to the benchmark console when requested."""

    logger = logging.getLogger("heiplanet_models.Pmodel.Pmodel_ode")
    for handler in logger.handlers:
        if getattr(handler, "_benchmark_progress_handler", False):
            logger.removeHandler(handler)

    if not show_chunk_progress(args):
        return

    handler = RichHandler(
        console=console,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.INFO)
    handler._benchmark_progress_handler = True  # type: ignore[attr-defined]

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def read_process_io_counters() -> tuple[int | None, int | None]:
    io_path = Path("/proc/self/io")
    if not io_path.exists():
        return None, None

    read_bytes = None
    write_bytes = None
    with io_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("read_bytes:"):
                read_bytes = int(line.split(":", 1)[1].strip())
            elif line.startswith("write_bytes:"):
                write_bytes = int(line.split(":", 1)[1].strip())
    return read_bytes, write_bytes


def build_stage_metric(
    *,
    year: int,
    stage: str,
    wall_start: float,
    cpu_start: float,
    rss_start: int,
    io_read_start: int | None,
    io_write_start: int | None,
    status: str,
    message: str,
) -> StageMetric:
    wall_end = time.perf_counter()
    cpu_end = time.process_time()
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    io_read_end, io_write_end = read_process_io_counters()

    return StageMetric(
        year=year,
        stage=stage,
        wall_seconds=wall_end - wall_start,
        cpu_seconds=cpu_end - cpu_start,
        rss_delta_kb=max(0, rss_end - rss_start),
        io_read_bytes=(
            None
            if io_read_start is None or io_read_end is None
            else io_read_end - io_read_start
        ),
        io_write_bytes=(
            None
            if io_write_start is None or io_write_end is None
            else io_write_end - io_write_start
        ),
        status=status,
        message=message,
    )


def run_stage(
    year: int,
    stage: str,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[StageMetric, T | None, Exception | None]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    io_read_start, io_write_start = read_process_io_counters()

    try:
        result = func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return (
            build_stage_metric(
                year=year,
                stage=stage,
                wall_start=wall_start,
                cpu_start=cpu_start,
                rss_start=rss_start,
                io_read_start=io_read_start,
                io_write_start=io_write_start,
                status="failed",
                message=str(exc),
            ),
            None,
            exc,
        )

    return (
        build_stage_metric(
            year=year,
            stage=stage,
            wall_start=wall_start,
            cpu_start=cpu_start,
            rss_start=rss_start,
            io_read_start=io_read_start,
            io_write_start=io_write_start,
            status="ok",
            message="-",
        ),
        result,
        None,
    )


# -----------------------------------------------------------------------------
# Rich output
# -----------------------------------------------------------------------------
def build_header_rows(args: argparse.Namespace) -> list[list[str]]:
    rows = [
        ["settings", str(Path(args.settings).resolve())],
        ["year range", f"{args.initial_year}..{args.final_year}"],
        ["backend", args.backend],
    ]
    if args.backend == "scipy_chunked":
        rows.extend(
            [
                ["chunk lon", str(args.chunk_lon)],
                ["chunk lat", str(args.chunk_lat)],
                ["scipy method", args.scipy_method],
                ["scipy rtol", f"{args.scipy_rtol:g}"],
                ["scipy atol", f"{args.scipy_atol:g}"],
            ]
        )
    else:
        rows.append(["solver options", "legacy backend; SciPy options not used"])
    rows.extend(
        [
            ["stage progress", "on" if show_stage_progress(args) else "off"],
            ["chunk progress", "on" if show_chunk_progress(args) else "off"],
        ]
    )
    return rows


def add_stage_table_columns(table: Table) -> None:
    table.add_column("Year", justify="right")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Wall", justify="right")
    table.add_column("Wall %", justify="right")
    table.add_column("CPU", justify="right")
    table.add_column("Peak RSS delta", justify="right")
    table.add_column("Read", justify="right")
    table.add_column("Write", justify="right")


def stage_table_row(metric: StageMetric, total_wall_seconds: float) -> list[str]:
    return [
        str(metric.year),
        metric.stage,
        format_stage_status(metric.status),
        format_seconds(metric.wall_seconds),
        f"{wall_percent(metric.wall_seconds, total_wall_seconds):.1f}%",
        format_seconds(metric.cpu_seconds),
        f"{kb_to_mb(metric.rss_delta_kb):.2f}MB",
        format_optional_bytes_as_mb(metric.io_read_bytes),
        format_optional_bytes_as_mb(metric.io_write_bytes),
    ]


def add_bottleneck_table_columns(table: Table) -> None:
    table.add_column("Stage")
    table.add_column("Wall total", justify="right")
    table.add_column("Wall %", justify="right")
    table.add_column("CPU total", justify="right")
    table.add_column("Peak RSS delta total", justify="right")
    table.add_column("Read total", justify="right")
    table.add_column("Write total", justify="right")


def default_aggregate_stage_metric() -> dict[str, float]:
    return {
        "wall": 0.0,
        "cpu": 0.0,
        "rss_mb": 0.0,
        "read_mb": 0.0,
        "write_mb": 0.0,
    }


def aggregate_stage_metric(
    aggregate: dict[str, dict[str, float]], metric: StageMetric
) -> None:
    values = aggregate.setdefault(metric.stage, default_aggregate_stage_metric())
    values["wall"] += metric.wall_seconds
    values["cpu"] += metric.cpu_seconds
    values["rss_mb"] += kb_to_mb(metric.rss_delta_kb)
    if metric.io_read_bytes is not None:
        values["read_mb"] += bytes_to_mb(metric.io_read_bytes)
    if metric.io_write_bytes is not None:
        values["write_mb"] += bytes_to_mb(metric.io_write_bytes)


def bottleneck_table_row(
    stage: str, values: dict[str, float], total_wall_seconds: float
) -> list[str]:
    return [
        stage,
        format_seconds(values["wall"]),
        f"{wall_percent(values['wall'], total_wall_seconds):.1f}%",
        format_seconds(values["cpu"]),
        f"{values['rss_mb']:.2f}MB",
        f"{values['read_mb']:.2f}MB",
        f"{values['write_mb']:.2f}MB",
    ]


def count_year_statuses(year_results: list[YearResult]) -> dict[str, int]:
    return {
        "completed": sum(result.status == "completed" for result in year_results),
        "skipped": sum(result.status == "skipped" for result in year_results),
        "failed": sum(result.status == "failed" for result in year_results),
    }


def add_year_summary_columns(table: Table) -> None:
    table.add_column("Year", justify="right")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Message")


def year_summary_row(result: YearResult) -> list[str]:
    return [
        str(result.year),
        format_year_status(result.status),
        format_seconds(result.seconds),
        result.message or "-",
    ]


def build_year_totals(
    year_results: list[YearResult], total_seconds: float
) -> dict[str, str]:
    counts = count_year_statuses(year_results)
    return {
        "years requested": str(len(year_results)),
        "completed": str(counts["completed"]),
        "skipped": str(counts["skipped"]),
        "failed": str(counts["failed"]),
        "total duration": format_seconds(total_seconds),
    }


def render_header(console: Console, args: argparse.Namespace) -> None:
    rows = build_header_rows(args)
    table = Table(show_header=False)
    table.add_column("setting", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in rows:
        table.add_row(key, value)
    console.print(
        Panel(table, title="Pmodel Pipeline Benchmark", border_style="magenta")
    )


def render_stage_table(console: Console, stage_metrics: list[StageMetric]) -> None:
    total_wall_seconds = sum(metric.wall_seconds for metric in stage_metrics)
    table = Table(title="Stage Measurements")
    add_stage_table_columns(table)

    for metric in stage_metrics:
        table.add_row(*stage_table_row(metric, total_wall_seconds))
    console.print(table)


def render_bottleneck_summary(
    console: Console, stage_metrics: list[StageMetric]
) -> None:
    aggregate: dict[str, dict[str, float]] = {}
    for metric in stage_metrics:
        aggregate_stage_metric(aggregate, metric)

    ranking = sorted(aggregate.items(), key=lambda item: item[1]["wall"], reverse=True)
    total_wall_seconds = sum(values["wall"] for values in aggregate.values())

    table = Table(title="Bottleneck Summary (aggregated by stage)")
    add_bottleneck_table_columns(table)

    for stage, values in ranking:
        table.add_row(*bottleneck_table_row(stage, values, total_wall_seconds))
    console.print(table)


def render_year_summary(
    console: Console, year_results: list[YearResult], total_seconds: float
) -> None:
    table = Table(title="Year Summary")
    add_year_summary_columns(table)

    for result in year_results:
        table.add_row(*year_summary_row(result))

    totals = Table(title="Benchmark Totals", show_header=False)
    totals.add_column("Metric", style="cyan")
    totals.add_column("Value", justify="right")
    for metric, value in build_year_totals(year_results, total_seconds).items():
        totals.add_row(metric, value)

    console.print(table)
    console.print(totals)


def build_failed_year_result(
    year: int, year_start: float, error: Exception
) -> YearResult:
    stage = "Pipeline"
    original_error = error
    if isinstance(error, BenchmarkStageError):
        stage = error.metric.stage
        original_error = error.original_error

    return YearResult(
        year=year,
        status="failed",
        seconds=time.perf_counter() - year_start,
        message=f"{stage} failed: {original_error}",
    )


def build_skipped_year_result(year: int, year_start: float) -> YearResult:
    return YearResult(
        year=year,
        status="skipped",
        seconds=time.perf_counter() - year_start,
        message="One or more input paths do not exist.",
    )


def build_completed_year_result(
    year: int, year_start: float, output_path: str | Path
) -> YearResult:
    resolved_output_path = Path(output_path).resolve()
    return YearResult(
        year=year,
        status="completed",
        seconds=time.perf_counter() - year_start,
        output_path=str(resolved_output_path),
        message=f"Output: {resolved_output_path}",
    )


def load_paths(year: int, etl_settings: dict[str, Any]) -> dict[str, Path]:
    return assemble_filepaths(year=year, **etl_settings)


def load_model_data(paths: dict[str, Path], etl_settings: dict[str, Any]) -> Any:
    return load_all_data(paths=paths, etl_settings=etl_settings)


def load_model_data_daily(paths: dict[str, Path], etl_settings: dict[str, Any]) -> Any:
    return load_all_data_daily(paths=paths, etl_settings=etl_settings)


def load_data_stage(
    stage: BenchmarkStageRunner,
    paths: dict[str, Path],
    etl_settings: dict[str, Any],
    backend: str,
) -> Any:
    loader = load_model_data_daily if backend == "scipy_chunked" else load_model_data
    return stage("Load all data", loader, paths, etl_settings)


def temperature_for_backend(model_data: Any, backend: str) -> Any:
    if backend == "scipy_chunked":
        return None
    return model_data.temperature


def calculate_carrying_capacity(model_data: Any) -> Any:
    return carrying_capacity(
        rainfall_data=model_data.rainfall,
        population_data=model_data.population_density,
    )


def calculate_egg_activity(model_data: Any) -> Any:
    return water_hatching(
        rainfall_data=model_data.rainfall,
        population_data=model_data.population_density,
    )


def solve_ode(
    model_data: Any,
    carry_capacity: Any,
    egg_active: Any,
    args: argparse.Namespace,
    etl_settings: dict[str, Any],
) -> Any:
    return solve_system(
        state=model_data.initial_conditions,
        temperature=temperature_for_backend(model_data, args.backend),
        temperature_mean=model_data.temperature_mean,
        latitudes=model_data.latitude,
        carrying_capacity=carry_capacity,
        egg_activate=egg_active,
        time_step=etl_settings["ode_system"]["time_step"],
        backend=args.backend,
        chunk_lon=args.chunk_lon,
        chunk_lat=args.chunk_lat,
        scipy_method=args.scipy_method,
        scipy_rtol=args.scipy_rtol,
        scipy_atol=args.scipy_atol,
    )


def create_output_dataset(
    ode_solution: Any, model_data: Any, etl_settings: dict[str, Any]
) -> Any:
    return build_output_dataset(
        state=ode_solution,
        model_data=model_data,
        compartments=etl_settings["ode_system"]["model_variables"],
    )


def write_output_dataset(
    output_dataset: Any, year: int, etl_settings: dict[str, Any]
) -> Path:
    return save_output_dataset(
        dataset=output_dataset,
        year=year,
        **etl_settings["serving"],
    )


def run_model_benchmark_stages(
    year: int,
    args: argparse.Namespace,
    etl_settings: dict[str, Any],
    stage: BenchmarkStageRunner,
) -> str | Path | None:
    paths = stage("Assemble filepaths", load_paths, year, etl_settings)
    paths_exist = stage("Check all paths exist", check_all_paths_exist, paths)
    if paths_exist is False:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model_data = load_data_stage(stage, paths, etl_settings, args.backend)
        capacity = stage(
            "Calculate carrying capacity",
            calculate_carrying_capacity,
            model_data,
        )
        egg_active = stage(
            "Calculate water hatching",
            calculate_egg_activity,
            model_data,
        )
        ode_solution = stage(
            "Solve ODE system",
            solve_ode,
            model_data,
            capacity,
            egg_active,
            args,
            etl_settings,
        )
        output_dataset = stage(
            "Build output dataset",
            create_output_dataset,
            ode_solution,
            model_data,
            etl_settings,
        )
        return stage(
            "Save output dataset",
            write_output_dataset,
            output_dataset,
            year,
            etl_settings,
        )


# -----------------------------------------------------------------------------
# MODEL BENCHMARK PIPELINE: edit this section for another model
#
# Keep the stage order explicit. Each stage("Name", lambda: ...) call is one
# measured step. To benchmark another model, replace these calls with that
# model's loading, equations, solver, and output-writing functions.
# -----------------------------------------------------------------------------
def benchmark_year(
    year: int,
    args: argparse.Namespace,
    etl_settings: dict[str, Any],
    console: Console,
    all_stage_metrics: list[StageMetric],
    verbose: bool,
) -> tuple[YearResult, bool]:
    year_start = time.perf_counter()

    if not args.quiet_progress:
        console.rule(f"Year {year}")
    stage = BenchmarkStageRunner(
        year,
        all_stage_metrics,
        console,
        show_stage_progress(args),
    )

    try:
        output_path = run_model_benchmark_stages(year, args, etl_settings, stage)
    except Exception as error:  # noqa: BLE001
        if verbose:
            render_exception_traceback(console, f"Traceback for year {year}", error)
        return build_failed_year_result(year, year_start, error), False

    if output_path is None:
        return build_skipped_year_result(year, year_start), True

    return build_completed_year_result(year, year_start, output_path), True


def validate_year_range(args: argparse.Namespace) -> None:
    if args.final_year < args.initial_year:
        raise SystemExit("--final-year must be greater than or equal to --initial-year")


def render_setup_progress_start(console: Console, enabled: bool) -> None:
    if enabled:
        console.print("[yellow]Running setup: read_global_settings...[/yellow]")


def render_setup_progress_result(
    console: Console,
    enabled: bool,
    metric: StageMetric,
    error: Exception | None,
) -> None:
    if not enabled:
        return

    if error is None:
        console.print(
            f"  [green][OK][/green] setup: read_global_settings "
            f"({metric.wall_seconds:.2f}s)"
        )
        return

    console.print(
        f"  [red][FAILED][/red] setup: read_global_settings "
        f"({metric.wall_seconds:.2f}s) - {error}"
    )


def read_settings_stage(
    args: argparse.Namespace, console: Console, setup_progress: bool
) -> tuple[StageMetric, dict[str, Any] | None, Exception | None]:
    render_setup_progress_start(console, setup_progress)
    metric, etl_settings, error = run_stage(
        args.initial_year,
        "read_global_settings",
        read_global_settings,
        filepath_configuration_file=args.settings,
    )
    render_setup_progress_result(console, setup_progress, metric, error)
    return metric, etl_settings, error


def render_exception_traceback(
    console: Console, title: str, error: Exception | None
) -> None:
    traceback_text = "".join(traceback.format_exception(error))
    console.print(Panel(traceback_text, title=title, border_style="red"))


def handle_settings_error(
    console: Console,
    args: argparse.Namespace,
    stage_metrics: list[StageMetric],
    error: Exception | None,
) -> None:
    if error is None:
        return

    if args.verbose:
        render_exception_traceback(console, "read_global_settings traceback", error)
    render_stage_table(console, stage_metrics)
    raise SystemExit(1) from error


def should_stop_after_year(
    result: YearResult, should_continue: bool, args: argparse.Namespace
) -> bool:
    return args.fail_fast and (result.status == "failed" or not should_continue)


def run_year_range(
    args: argparse.Namespace,
    etl_settings: dict[str, Any],
    console: Console,
    stage_metrics: list[StageMetric],
) -> list[YearResult]:
    year_results: list[YearResult] = []
    for year in range(args.initial_year, args.final_year + 1):
        result, should_continue = benchmark_year(
            year=year,
            args=args,
            etl_settings=etl_settings,
            console=console,
            all_stage_metrics=stage_metrics,
            verbose=args.verbose,
        )
        year_results.append(result)
        if should_stop_after_year(result, should_continue, args):
            break
    return year_results


def render_all_reports(
    console: Console,
    stage_metrics: list[StageMetric],
    year_results: list[YearResult],
    total_seconds: float,
) -> None:
    render_stage_table(console, stage_metrics)
    render_bottleneck_summary(console, stage_metrics)
    render_year_summary(console, year_results, total_seconds)


def exit_if_failures(year_results: list[YearResult]) -> None:
    if any(result.status == "failed" for result in year_results):
        raise SystemExit(1)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    console = Console()
    overall_start = time.perf_counter()
    setup_progress = show_stage_progress(args)
    configure_progress_logging(console, args)

    validate_year_range(args)

    if not args.quiet_progress:
        render_header(console, args)

    settings_metric, etl_settings, settings_error = read_settings_stage(
        args, console, setup_progress
    )
    stage_metrics: list[StageMetric] = [settings_metric]
    handle_settings_error(console, args, stage_metrics, settings_error)

    year_results = run_year_range(args, etl_settings, console, stage_metrics)
    total_seconds = time.perf_counter() - overall_start
    render_all_reports(console, stage_metrics, year_results, total_seconds)
    exit_if_failures(year_results)


if __name__ == "__main__":
    main()
