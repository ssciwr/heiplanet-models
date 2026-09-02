"""Unit tests for heiplanet_models.Pmodel.Pmodel_output.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_output import (
    PmodelOutput,
    assemble_output_filepath,
    build_output_dataset,
    create_incremental_netcdf_writer,
    save_output_dataset,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def compartments(test_etl_settings):
    return test_etl_settings["ode_system"]["model_variables"]


@pytest.fixture
def synthetic_state(model_input_dummy_datasets, compartments):
    n_lon = model_input_dummy_datasets.temperature_mean.sizes["longitude"]
    n_lat = model_input_dummy_datasets.temperature_mean.sizes["latitude"]
    n_time = model_input_dummy_datasets.temperature_mean.sizes["time"]
    n_vars = len(compartments)
    return np.arange(n_lon * n_lat * n_vars * n_time, dtype=np.float64).reshape(
        n_lon, n_lat, n_vars, n_time
    )


# =============================================================================
# TESTS: PmodelOutput
# =============================================================================


def test_pmodel_output_repr_contains_shape():
    array = np.zeros((2, 3, 4, 5), dtype=np.float64)
    output = PmodelOutput(model_output=array)

    text = repr(output)
    assert "PmodelOutput" in text
    assert "model_output" in text
    assert "shape=(2, 3, 4, 5)" in text


# =============================================================================
# TESTS: build_output_dataset
# =============================================================================


def test_build_output_dataset_structure(
    model_input_dummy_datasets,
    synthetic_state,
    compartments,
):
    output_dataset = build_output_dataset(
        state=synthetic_state,
        model_data=model_input_dummy_datasets,
        compartments=compartments,
    )

    assert isinstance(output_dataset, xr.Dataset)
    assert set(output_dataset.data_vars) == set(compartments)

    expected_dims = ("longitude", "latitude", "time")
    expected_shape = (
        model_input_dummy_datasets.temperature_mean.sizes["longitude"],
        model_input_dummy_datasets.temperature_mean.sizes["latitude"],
        model_input_dummy_datasets.temperature_mean.sizes["time"],
    )

    for name in compartments:
        assert output_dataset[name].dims == expected_dims
        assert output_dataset[name].shape == expected_shape


# =============================================================================
# TESTS: output file utilities
# =============================================================================


def test_assemble_output_filepath_with_year(tmp_path):
    path = assemble_output_filepath(
        year=2024,
        path_output_datasets=str(tmp_path),
        filename_components={
            "prefix": "result_",
            "suffix": "_dummy",
            "extension": ".nc",
        },
    )

    assert path == tmp_path / "result_2024_dummy.nc"


def test_assemble_output_filepath_without_year_and_default_extension(tmp_path):
    path = assemble_output_filepath(
        path_output_datasets=str(tmp_path),
        filename_components={
            "prefix": "result_",
            "suffix": None,
            "extension": None,
        },
    )

    assert path == tmp_path / "result_.nc"


def test_save_output_dataset_roundtrip(
    tmp_path,
    model_input_dummy_datasets,
    synthetic_state,
    compartments,
):
    dataset = build_output_dataset(
        state=synthetic_state,
        model_data=model_input_dummy_datasets,
        compartments=compartments,
    )

    output_path = save_output_dataset(
        dataset=dataset,
        year=2024,
        path_output_datasets=str(tmp_path),
        filename_components={
            "prefix": "result_",
            "suffix": "",
            "extension": ".nc",
        },
    )

    assert output_path.exists()

    with xr.open_dataset(output_path) as reloaded:
        assert set(reloaded.data_vars) == set(compartments)
        assert reloaded.sizes["longitude"] == dataset.sizes["longitude"]
        assert reloaded.sizes["latitude"] == dataset.sizes["latitude"]
        assert reloaded.sizes["time"] == dataset.sizes["time"]


def test_incremental_netcdf_writer_roundtrip(
    tmp_path,
    model_input_dummy_datasets,
    synthetic_state,
    compartments,
):
    output_path = tmp_path / "incremental.nc"

    with create_incremental_netcdf_writer(
        output_path=output_path,
        model_data=model_input_dummy_datasets,
        compartments=compartments,
        chunk_lon=1,
        chunk_lat=1,
        attrs={"solver_backend": "scipy_chunked"},
    ) as writer:
        for lon_start in range(synthetic_state.shape[0]):
            for lat_start in range(synthetic_state.shape[1]):
                lon_slice = slice(lon_start, lon_start + 1)
                lat_slice = slice(lat_start, lat_start + 1)
                writer.write_chunk(
                    lon_slice,
                    lat_slice,
                    synthetic_state[lon_slice, lat_slice],
                )

    assert output_path.exists()
    with xr.open_dataset(output_path) as reloaded:
        assert reloaded.attrs["solver_backend"] == "scipy_chunked"
        assert set(reloaded.data_vars) == set(compartments)
        assert reloaded[compartments[0]].encoding.get("zlib") is False
        for index, name in enumerate(compartments):
            np.testing.assert_allclose(
                reloaded[name].values,
                synthetic_state[:, :, index, :],
            )


def test_incremental_netcdf_writer_preserves_numeric_time_values(
    tmp_path,
    model_input_dummy_datasets,
    synthetic_state,
    compartments,
):
    """Test numeric time coordinates are written without datetime metadata."""
    numeric_time = np.arange(
        model_input_dummy_datasets.temperature_mean.sizes["time"], dtype=np.float64
    )
    model_input_dummy_datasets.temperature_mean = (
        model_input_dummy_datasets.temperature_mean.assign_coords(time=numeric_time)
    )
    output_path = tmp_path / "numeric-time.nc"

    with create_incremental_netcdf_writer(
        output_path=output_path,
        model_data=model_input_dummy_datasets,
        compartments=compartments,
        chunk_lon=1,
        chunk_lat=1,
    ) as writer:
        writer.write_chunk(
            slice(0, synthetic_state.shape[0]),
            slice(0, synthetic_state.shape[1]),
            synthetic_state,
        )

    with xr.open_dataset(output_path, decode_times=False) as reloaded:
        np.testing.assert_allclose(reloaded.time.values, numeric_time)
        assert "units" not in reloaded.time.attrs
