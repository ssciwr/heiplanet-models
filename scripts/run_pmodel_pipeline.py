"""Run the Pmodel pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from heiplanet_models.Pmodel.Pmodel_initial import (
    assemble_filepaths,
    check_all_paths_exist,
    load_all_data,
    load_all_data_daily,
    read_global_settings,
)
from heiplanet_models.Pmodel.Pmodel_ode import (
    SCIPY_METHODS,
    VALID_BACKENDS,
    solve_system,
)
from heiplanet_models.Pmodel.Pmodel_output import (
    build_output_dataset,
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


def load_data_for_backend(paths: dict, etl_settings: dict, backend: str):
    if backend == "scipy_chunked":
        return load_all_data_daily(paths=paths, etl_settings=etl_settings)
    return load_all_data(paths=paths, etl_settings=etl_settings)


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
        lambda: assemble_filepaths(year=year, **etl_settings),
    )

    paths_exist = stage(
        "Check all paths exist",
        lambda: check_all_paths_exist(path_dict=paths),
    )
    if not paths_exist:
        raise SkipYear("Missing input files.")

    model_data = stage(
        "Load all data",
        lambda: load_data_for_backend(paths, etl_settings, args.backend),
    )

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

    ode_solution = stage(
        "Solve ODE system",
        lambda: solve_system(
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
        ),
    )

    output_dataset = stage(
        "Build output dataset",
        lambda: build_output_dataset(
            state=ode_solution,
            model_data=model_data,
            compartments=etl_settings["ode_system"]["model_variables"],
        ),
    )

    return stage(
        "Save output dataset",
        lambda: save_output_dataset(
            dataset=output_dataset,
            year=year,
            **etl_settings["serving"],
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
