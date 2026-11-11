"""Run the Pmodel pipeline."""

import argparse
import contextlib
import copy
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from rich.logging import RichHandler

from heiplanet_models.Pmodel.Pmodel_initial import (
    assemble_filepaths,
    check_all_paths_exist,
    load_all_data,
    load_all_data_daily,
    read_global_settings,
)
from heiplanet_models.Pmodel.Pmodel_input import PmodelInput
from heiplanet_models.Pmodel.Pmodel_ode import (
    SCIPY_METHODS,
    VALID_BACKENDS,
    solve_system,
)
from heiplanet_models.Pmodel.Pmodel_output import (
    TIME_AGGREGATIONS,
    FinalStateCollector,
    IncrementalNetCDFWriter,
    LastStepWriter,
    MonthlyMeanWriter,
    TeeWriter,
    aggregate_dataset_monthly,
    assemble_output_filepath,
    build_output_dataset,
    final_state_filepath,
    monthly_groups,
    output_is_daily,
    save_output_dataset,
)
from heiplanet_models.Pmodel.Pmodel_rates_birth import water_hatching
from heiplanet_models.Pmodel.Pmodel_rates_development import carrying_capacity

try:
    from scripts.pipeline_utils import SkipYear, StageRunner, run_pipeline_cli
except ModuleNotFoundError:
    scripts_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scripts_dir))
    from pipeline_utils import SkipYear, StageRunner, run_pipeline_cli


MODEL_NAME = "Pmodel Pipeline"
DEFAULT_SETTINGS = "./src/heiplanet_models/Pmodel/global_settings.yaml"
DEFAULT_CHUNK_LON = 288
DEFAULT_CHUNK_LAT = 144
DEFAULT_SCIPY_METHOD = "RK45"
DEFAULT_SCIPY_RTOL = 1e-5
DEFAULT_SCIPY_ATOL = 1e-5

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def configure_cli_logging() -> None:
    """Show Pmodel warnings in the command-line runner."""

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(show_path=False)],
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=sorted(VALID_BACKENDS),
        default="scipy_chunked",
        help="ODE solver backend.",
    )
    parser.add_argument(
        "--chunk-lon",
        type=int,
        default=DEFAULT_CHUNK_LON,
        help="Longitude chunk size for the scipy_chunked backend.",
    )
    parser.add_argument(
        "--chunk-lat",
        type=int,
        default=DEFAULT_CHUNK_LAT,
        help="Latitude chunk size for the scipy_chunked backend.",
    )
    parser.add_argument(
        "--scipy-method",
        default=DEFAULT_SCIPY_METHOD,
        help="solve_ivp method for the scipy_chunked backend.",
    )
    parser.add_argument(
        "--scipy-rtol",
        type=float,
        default=DEFAULT_SCIPY_RTOL,
        help="Relative tolerance for the scipy_chunked backend.",
    )
    parser.add_argument(
        "--scipy-atol",
        type=float,
        default=DEFAULT_SCIPY_ATOL,
        help="Absolute tolerance for the scipy_chunked backend.",
    )


def validate_model_args(args: argparse.Namespace) -> None:
    """Validate backend-specific command-line arguments before a pipeline run."""

    scipy_options_are_custom = (
        args.chunk_lon != DEFAULT_CHUNK_LON
        or args.chunk_lat != DEFAULT_CHUNK_LAT
        or args.scipy_method != DEFAULT_SCIPY_METHOD
        or args.scipy_rtol != DEFAULT_SCIPY_RTOL
        or args.scipy_atol != DEFAULT_SCIPY_ATOL
    )
    if args.backend != "scipy_chunked":
        if scipy_options_are_custom:
            logger.warning("SciPy options are ignored by the %s backend.", args.backend)
        return

    if args.chunk_lon <= 0 or args.chunk_lat <= 0:
        raise ValueError("--chunk-lon and --chunk-lat must be positive.")
    if args.scipy_rtol <= 0.0 or args.scipy_atol <= 0.0:
        raise ValueError("--scipy-rtol and --scipy-atol must be positive.")
    if args.scipy_method not in SCIPY_METHODS:
        methods = ", ".join(sorted(SCIPY_METHODS))
        raise ValueError(f"--scipy-method must be one of: {methods}.")


def display_args(args: argparse.Namespace) -> dict[str, object]:
    header_values: dict[str, object] = {
        "settings": args.settings,
        "backend": args.backend,
    }

    if args.backend == "scipy_chunked":
        header_values.update(
            {
                "chunk_lon": args.chunk_lon,
                "chunk_lat": args.chunk_lat,
                "scipy_method": args.scipy_method,
                "scipy_rtol": args.scipy_rtol,
                "scipy_atol": args.scipy_atol,
            }
        )
    else:
        header_values["solver_options"] = "legacy backend; SciPy options not used"

    if args.verbose:
        header_values["verbose"] = True

    return header_values


def load_data_for_backend(
    paths: dict, etl_settings: dict, backend: str, year: int | None = None
):
    if backend == "scipy_chunked":
        return load_all_data_daily(paths=paths, etl_settings=etl_settings, year=year)
    return load_all_data(paths=paths, etl_settings=etl_settings, year=year)


def rechunk_for_regions(
    model_data: PmodelInput, chunk_lon: int, chunk_lat: int
) -> PmodelInput:
    """Align lazy input chunks with the solver's spatial regions.

    Each solver region then maps onto exactly one dask block, so the derived
    inputs (carrying capacity, hatching, rates) are computed once per region
    without materialising the global arrays. Eager (NumPy) inputs are unchanged.
    """

    spec = {"longitude": chunk_lon, "latitude": chunk_lat, "time": -1}

    def rechunk(data_array):
        if data_array is None or data_array.chunks is None:
            return data_array
        return data_array.chunk({d: spec[d] for d in data_array.dims if d in spec})

    return dataclasses.replace(
        model_data,
        population_density=rechunk(model_data.population_density),
        rainfall=rechunk(model_data.rainfall),
        temperature_mean=rechunk(model_data.temperature_mean),
    )


def resolve_initial_conditions(year: int | None, etl_settings: dict) -> Path | None:
    """Pick the file whose last time step is used as this year's initial state.

    With ``ingestion.initial_conditions.chain_previous_year`` enabled, the
    previous year's final state file is used if it exists, else the previous
    year's output if it holds daily values. Otherwise the configured
    ``file_path_initial_conditions`` is used if it exists. ``None`` means the
    model starts from its default initial conditions.
    """

    settings = etl_settings["ingestion"].get("initial_conditions") or {}
    if year is not None and settings.get("chain_previous_year", False):
        serving = etl_settings["serving"]
        previous_state = final_state_filepath(year=year - 1, **serving)
        if previous_state.exists():
            return previous_state
        previous_output = assemble_output_filepath(year=year - 1, **serving)
        if previous_output.exists():
            if output_is_daily(previous_output):
                return previous_output
            logger.warning(
                "Cannot chain from %s: it holds monthly means and its final "
                "state file %s is missing.",
                previous_output,
                previous_state,
            )

    configured = settings.get("file_path_initial_conditions")
    if configured and Path(configured).exists():
        return Path(configured)
    return None


def settings_with_initial_conditions(
    etl_settings: dict, initial_conditions: Path | None
) -> dict:
    """Return a copy of the settings pointing at ``initial_conditions``."""

    year_settings = copy.deepcopy(etl_settings)
    year_settings["ingestion"].setdefault("initial_conditions", {})
    year_settings["ingestion"]["initial_conditions"]["file_path_initial_conditions"] = (
        str(initial_conditions) if initial_conditions is not None else None
    )
    return year_settings


def run_solver(
    args: argparse.Namespace,
    etl_settings: dict,
    model_data: PmodelInput,
    carry_capacity,
    egg_active,
    output_writer=None,
):
    """Call ``solve_system`` with the pipeline's inputs and solver options."""

    return solve_system(
        state=model_data.initial_conditions,
        temperature=temperature_for_backend(model_data, args.backend),
        temperature_mean=model_data.temperature_mean,
        latitudes=model_data.latitude,
        carrying_capacity=carry_capacity,
        egg_activate=egg_active,
        time_step=etl_settings["ode_system"]["time_step"],
        backend=args.backend,
        scipy_method=args.scipy_method,
        scipy_rtol=args.scipy_rtol,
        scipy_atol=args.scipy_atol,
        chunk_lon=args.chunk_lon,
        chunk_lat=args.chunk_lat,
        output_writer=output_writer,
    )


def spin_up_initial_conditions(
    n_years: int,
    args: argparse.Namespace,
    etl_settings: dict,
    model_data: PmodelInput,
    carry_capacity,
    egg_active,
) -> PmodelInput:
    """Run the year ``n_years`` times and start from the resulting final state.

    Only the final state is kept; the spin-up output itself is not written.
    """

    initial = model_data.initial_conditions
    for _ in range(n_years):
        if args.backend == "scipy_chunked":
            collector = FinalStateCollector(*initial.shape)
            run_solver(
                args,
                etl_settings,
                model_data,
                carry_capacity,
                egg_active,
                output_writer=collector,
            )
            final_state = collector.state
        else:
            final_state = run_solver(
                args, etl_settings, model_data, carry_capacity, egg_active
            )[..., -1]
        initial = initial.copy(data=final_state)
        model_data = dataclasses.replace(model_data, initial_conditions=initial)
    return model_data


def time_aggregation(etl_settings: dict) -> str:
    """Return the validated ``serving.time_aggregation`` setting."""

    aggregation = etl_settings["serving"].get("time_aggregation") or "daily"
    if aggregation not in TIME_AGGREGATIONS:
        options = ", ".join(TIME_AGGREGATIONS)
        raise ValueError(
            f"serving.time_aggregation must be one of: {options}; got {aggregation!r}."
        )
    return aggregation


def open_final_state_writer(
    path: Path,
    model_data: PmodelInput,
    compartments: list[str],
    chunk_lon: int,
    chunk_lat: int,
) -> IncrementalNetCDFWriter:
    """Writer for the last daily state (float64, used to chain the next year)."""

    temperature_mean = model_data.temperature_mean
    return IncrementalNetCDFWriter(
        output_path=path,
        longitude=temperature_mean["longitude"],
        latitude=temperature_mean["latitude"],
        time=temperature_mean["time"].isel(time=[-1]),
        compartments=compartments,
        chunk_lon=chunk_lon,
        chunk_lat=chunk_lat,
        attrs={"description": "Pmodel state on the last simulated day"},
        compression=True,
        dtype="f8",
    )


def solve_and_stream_output(
    year: int,
    args: argparse.Namespace,
    etl_settings: dict,
    model_data: PmodelInput,
    carry_capacity,
    egg_active,
) -> Path:
    """Solve region by region and write each region straight to NetCDF.

    Writes the model output (daily or monthly means, ``serving.time_aggregation``)
    and the final daily state used to chain the next year. Both are written to
    ``<name>.part`` and renamed only on success, so an interrupted run never
    leaves files that look complete.
    """

    serving = etl_settings["serving"]
    aggregation = time_aggregation(etl_settings)
    compartments = etl_settings["ode_system"]["model_variables"]
    daily_time = model_data.temperature_mean["time"]

    output_path = assemble_output_filepath(year=year, **serving)
    state_path = final_state_filepath(year=year, **serving)
    partial_paths = {
        path: path.with_name(path.name + ".part") for path in (output_path, state_path)
    }

    if aggregation == "monthly_mean":
        _, _, month_starts = monthly_groups(daily_time)
        output_time = xr.DataArray(month_starts, dims="time")
        variable_attrs = {"cell_methods": "time: mean"}
    else:
        output_time = daily_time
        variable_attrs = None

    try:
        with contextlib.ExitStack() as stack:
            output_writer = stack.enter_context(
                IncrementalNetCDFWriter(
                    output_path=partial_paths[output_path],
                    longitude=model_data.temperature_mean["longitude"],
                    latitude=model_data.temperature_mean["latitude"],
                    time=output_time,
                    compartments=compartments,
                    chunk_lon=args.chunk_lon,
                    chunk_lat=args.chunk_lat,
                    attrs={"time_aggregation": aggregation},
                    compression=serving.get("compression", True),
                    dtype=serving.get("dtype", "f4"),
                    variable_attrs=variable_attrs,
                )
            )
            state_writer = stack.enter_context(
                open_final_state_writer(
                    partial_paths[state_path],
                    model_data,
                    compartments,
                    args.chunk_lon,
                    args.chunk_lat,
                )
            )
            if aggregation == "monthly_mean":
                output_writer = MonthlyMeanWriter(output_writer, daily_time)
            run_solver(
                args,
                etl_settings,
                model_data,
                carry_capacity,
                egg_active,
                output_writer=TeeWriter(output_writer, LastStepWriter(state_writer)),
            )
    except BaseException:
        for partial_path in partial_paths.values():
            partial_path.unlink(missing_ok=True)
        raise

    for path, partial_path in partial_paths.items():
        partial_path.replace(path)
    return output_path


def save_output_and_state(
    year: int | None,
    args: argparse.Namespace,
    etl_settings: dict,
    model_data: PmodelInput,
    ode_solution,
) -> Path:
    """Save in-memory solver output (non-streaming backends) and final state."""

    compartments = etl_settings["ode_system"]["model_variables"]
    output_dataset = build_output_dataset(
        state=ode_solution, model_data=model_data, compartments=compartments
    )
    aggregation = time_aggregation(etl_settings)
    if aggregation == "monthly_mean":
        output_dataset = aggregate_dataset_monthly(output_dataset)
    else:
        output_dataset.attrs["time_aggregation"] = aggregation

    output_path = save_output_dataset(
        dataset=output_dataset, year=year, **etl_settings["serving"]
    )

    n_lon, n_lat = ode_solution.shape[:2]
    with open_final_state_writer(
        final_state_filepath(year=year, **etl_settings["serving"]),
        model_data,
        compartments,
        n_lon,
        n_lat,
    ) as state_writer:
        LastStepWriter(state_writer).write_chunk(
            slice(None), slice(None), np.asarray(ode_solution)
        )
    return output_path


def temperature_for_backend(model_data, backend: str):
    if backend == "scipy_chunked":
        return None
    return model_data.temperature


# -----------------------------------------------------------------------------
# MODEL PIPELINE
#
# To create another model runner, keep the CLI helper above and replace this
# function with calls to that model's ecological equations and output functions.
# -----------------------------------------------------------------------------
def run_year(
    year: int,
    args: argparse.Namespace,
    etl_settings: dict,
    stage: StageRunner,
) -> Path:
    paths = stage(
        "Assemble filepaths",
        lambda: assemble_filepaths(**etl_settings),
    )

    paths_exist = stage(
        "Check all paths exist",
        lambda: check_all_paths_exist(path_dict=paths),
    )
    if not paths_exist:
        raise SkipYear("Missing input files.")

    initial_conditions = resolve_initial_conditions(year, etl_settings)
    stage.console.print(
        "  Initial conditions: "
        + (
            f"last day of {initial_conditions}"
            if initial_conditions is not None
            else "model defaults"
        )
    )
    etl_settings = settings_with_initial_conditions(etl_settings, initial_conditions)

    model_data = stage(
        "Load all data",
        lambda: load_data_for_backend(paths, etl_settings, args.backend, year),
    )
    streaming = args.backend == "scipy_chunked"
    if streaming:
        model_data = rechunk_for_regions(model_data, args.chunk_lon, args.chunk_lat)

    carry_capacity = stage(
        "Calculate carrying capacity",
        lambda: carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        ),
    )

    egg_active = stage(
        "Calculate water hatching",
        lambda: water_hatching(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        ),
    )

    spinup_years = int(
        (etl_settings["ingestion"].get("initial_conditions") or {}).get(
            "spinup_years", 0
        )
        or 0
    )
    if initial_conditions is None and spinup_years > 0:
        model_data = stage(
            f"Spin up initial conditions ({spinup_years} x {year} forcing)",
            lambda: spin_up_initial_conditions(
                spinup_years,
                args,
                etl_settings,
                model_data,
                carry_capacity,
                egg_active,
            ),
        )

    if streaming:
        # Inputs above are lazy; they are computed region by region while solving.
        return stage(
            "Solve ODE system and write output",
            lambda: solve_and_stream_output(
                year, args, etl_settings, model_data, carry_capacity, egg_active
            ),
        )

    ode_solution = stage(
        "Solve ODE system",
        lambda: run_solver(args, etl_settings, model_data, carry_capacity, egg_active),
    )

    return stage(
        "Save output dataset and final state",
        lambda: save_output_and_state(
            year, args, etl_settings, model_data, ode_solution
        ),
    )


if __name__ == "__main__":
    configure_cli_logging()
    run_pipeline_cli(
        model_name=MODEL_NAME,
        default_settings=DEFAULT_SETTINGS,
        default_initial_year=None,
        default_final_year=None,
        add_model_args=add_model_args,
        read_settings=read_global_settings,
        run_year=run_year,
        display_args=display_args,
        validate_args=validate_model_args,
    )
