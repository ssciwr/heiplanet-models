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
from heiplanet_models.Pmodel.Pmodel_output import (
    build_output_dataset,
    save_output_dataset,
)


def _run_pipeline(etl_settings: dict) -> tuple[xr.Dataset, xr.DataArray]:
    """Execute the same pipeline stages as scripts/run_model.py and return outputs."""
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


def _to_numeric_if_datetime(values: np.ndarray) -> np.ndarray:
    """Convert datetime-like arrays to integer nanoseconds for tolerant compare."""
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    return values


def _assert_allclose(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> None:
    """Assert two arrays are shape-compatible and numerically close."""
    assert a.shape == b.shape
    np.testing.assert_allclose(
        _to_numeric_if_datetime(np.asarray(a)),
        _to_numeric_if_datetime(np.asarray(b)),
        rtol=rtol,
        atol=atol,
    )


def _transpose_to_match_dims(
    reference: xr.DataArray, candidate: xr.DataArray
) -> xr.DataArray:
    """Golden data stores dimensions in a different order; transpose to match output."""
    return candidate.transpose(*reference.dims)


@pytest.fixture
def integration_etl_settings(tmp_path: Path) -> dict:
    """Load ETL settings and redirect output to a temporary directory."""
    repo_root = Path(__file__).resolve().parents[3]
    settings_path = repo_root / "test" / "test_resources" / "global_settings_dummy.yaml"
    etl_settings = read_global_settings(str(settings_path))

    # Make ingestion path absolute to keep the test independent from CWD.
    etl_settings["ingestion"]["path_root_datasets"] = str(
        repo_root / "test" / "test_resources"
    )

    # Write outputs into pytest temp directory.
    etl_settings["serving"]["path_output_datasets"] = str(tmp_path)
    etl_settings["serving"]["filename_components"]["prefix"] = "integration_output_"

    return etl_settings


def test_run_model_pipeline_end_to_end(integration_etl_settings: dict) -> None:
    """Run the same pipeline stages as scripts/run_model.py on dummy resources."""
    output_dataset, temperature_mean = _run_pipeline(integration_etl_settings)

    compartments = integration_etl_settings["ode_system"]["model_variables"]

    assert isinstance(output_dataset, xr.Dataset)
    assert set(output_dataset.data_vars) == set(compartments)

    n_lon = temperature_mean.sizes["longitude"]
    n_lat = temperature_mean.sizes["latitude"]
    n_time = temperature_mean.sizes["time"]

    for variable in compartments:
        data_array = output_dataset[variable]
        assert data_array.dims == ("longitude", "latitude", "time")
        assert data_array.shape == (n_lon, n_lat, n_time)
        assert np.isfinite(data_array.values).all()

    saved_path = save_output_dataset(
        dataset=output_dataset,
        year=2024,
        **integration_etl_settings["serving"],
    )

    assert saved_path.exists()

    with xr.open_dataset(saved_path) as reloaded:
        assert set(reloaded.data_vars) == set(compartments)
        assert reloaded.sizes["longitude"] == n_lon
        assert reloaded.sizes["latitude"] == n_lat
        assert reloaded.sizes["time"] == n_time


def test_run_model_pipeline_matches_golden_dataset(
    integration_etl_settings: dict,
) -> None:
    """Compare generated output to golden reference data for stable regression checks."""
    repo_root = Path(__file__).resolve().parents[3]
    golden_path = repo_root / "test" / "test_resources" / "output_dataset_dummy.nc"

    output_dataset, _ = _run_pipeline(integration_etl_settings)

    assert golden_path.exists()
    with xr.open_dataset(golden_path) as golden_dataset:
        # Golden data currently stores a single reference variable.
        assert "adults" in golden_dataset.data_vars
        assert "adults" in output_dataset.data_vars

        _assert_allclose(
            output_dataset["longitude"].values,
            golden_dataset["longitude"].values,
            rtol=0.0,
            atol=0.0,
        )
        _assert_allclose(
            output_dataset["latitude"].values,
            golden_dataset["latitude"].values,
            rtol=0.0,
            atol=0.0,
        )
        _assert_allclose(
            output_dataset["time"].values,
            golden_dataset["time"].values,
            rtol=0.0,
            atol=0.0,
        )
        golden_adults = _transpose_to_match_dims(
            reference=output_dataset["adults"],
            candidate=golden_dataset["adults"],
        )
        _assert_allclose(
            output_dataset["adults"].values,
            golden_adults.values,
            rtol=1e-6,
            atol=1e-8,
        )
