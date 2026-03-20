from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_initial import (
    read_global_settings,
    assemble_filepaths,
    check_all_paths_exist,
    load_all_data,
)
from heiplanet_models.Pmodel.Pmodel_rates_birth import water_hatching
from heiplanet_models.Pmodel.Pmodel_rates_development import carrying_capacity
from heiplanet_models.Pmodel.Pmodel_ode import solve_system
from heiplanet_models.Pmodel.Pmodel_output import build_output_dataset


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def integration_etl_settings(tmp_path: Path, repo_root: Path) -> dict:
    """Load ETL settings and redirect output to a temporary directory."""
    settings_path = repo_root / "test" / "test_resources" / "global_settings_dummy.yaml"
    etl_settings = read_global_settings(str(settings_path))

    # Make ingestion path absolute to keep tests independent from CWD.
    etl_settings["ingestion"]["path_root_datasets"] = str(
        repo_root / "test" / "test_resources"
    )

    # Write outputs into pytest temp directory.
    etl_settings["serving"]["path_output_datasets"] = str(tmp_path)
    etl_settings["serving"]["filename_components"]["prefix"] = "integration_output_"

    return etl_settings


@pytest.fixture
def run_pipeline():
    """Execute the same pipeline stages as scripts/run_model.py and return outputs."""

    def _run(etl_settings: dict) -> tuple[xr.Dataset, xr.DataArray]:
        paths = assemble_filepaths(year=None, **etl_settings)
        assert check_all_paths_exist(paths)

        model_data = load_all_data(paths=paths, etl_settings=etl_settings)

        capacity = carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        eggs_active = water_hatching(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )

        ode_solution = solve_system(
            state=model_data.initial_conditions,
            temperature=model_data.temperature,
            temperature_mean=model_data.temperature_mean,
            latitudes=model_data.latitude,
            carrying_capacity=capacity,
            egg_activate=eggs_active,
            time_step=etl_settings["ode_system"]["time_step"],
        )

        compartments = etl_settings["ode_system"]["model_variables"]
        output_dataset = build_output_dataset(
            state=ode_solution,
            model_data=model_data,
            compartments=compartments,
        )

        return output_dataset, model_data.temperature_mean

    return _run


@pytest.fixture
def assert_allclose_arrays():
    def _to_numeric_if_datetime(values: np.ndarray) -> np.ndarray:
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]").astype(np.int64)
        return values

    def _assert(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> None:
        assert a.shape == b.shape
        np.testing.assert_allclose(
            _to_numeric_if_datetime(np.asarray(a)),
            _to_numeric_if_datetime(np.asarray(b)),
            rtol=rtol,
            atol=atol,
        )

    return _assert


@pytest.fixture
def transpose_to_match_dims():
    def _transpose(reference: xr.DataArray, candidate: xr.DataArray) -> xr.DataArray:
        """Golden data stores dimensions in a different order; transpose to match output."""
        return candidate.transpose(*reference.dims)

    return _transpose
