"""Unit tests for heiplanet_models.Pmodel.Pmodel_output.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from heiplanet_models.Pmodel import Pmodel_output
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


def test_incremental_netcdf_writer_float32_compressed(
    tmp_path,
    model_input_dummy_datasets,
    synthetic_state,
    compartments,
):
    output_path = tmp_path / "incremental_f4.nc"

    with create_incremental_netcdf_writer(
        output_path=output_path,
        model_data=model_input_dummy_datasets,
        compartments=compartments,
        chunk_lon=2,
        chunk_lat=2,
        compression=True,
        dtype="f4",
    ) as writer:
        writer.write_chunk(slice(None), slice(None), synthetic_state)

    with xr.open_dataset(output_path) as reloaded:
        variable = reloaded[compartments[0]]
        assert variable.dtype == np.float32
        assert variable.encoding.get("zlib") is True
        for index, name in enumerate(compartments):
            np.testing.assert_allclose(
                reloaded[name].values, synthetic_state[:, :, index, :], rtol=1e-6
            )


def test_monthly_groups_across_months_and_years():
    time = xr.DataArray(
        pd.date_range("2015-12-30", "2016-02-02", freq="D"), dims="time"
    )
    starts, counts, month_starts = Pmodel_output.monthly_groups(time)
    np.testing.assert_array_equal(starts, [0, 2, 33])
    np.testing.assert_array_equal(counts, [2, 31, 2])
    np.testing.assert_array_equal(
        month_starts,
        np.array(["2015-12-01", "2016-01-01", "2016-02-01"], dtype="datetime64[ns]"),
    )


def test_monthly_groups_rejects_numeric_time():
    with pytest.raises(ValueError, match="datetime64"):
        Pmodel_output.monthly_groups(xr.DataArray(np.arange(3.0), dims="time"))


def test_monthly_mean_and_last_step_writers_match_xarray():
    time = pd.date_range("2016-01-25", "2016-03-03", freq="D")
    rng = np.random.default_rng(1)
    chunk = rng.random((2, 3, 5, time.size))

    class Sink:
        def write_chunk(self, lon_slice, lat_slice, output_chunk):
            self.chunk = output_chunk

    monthly_sink, last_sink = Sink(), Sink()
    writer = Pmodel_output.TeeWriter(
        Pmodel_output.MonthlyMeanWriter(monthly_sink, xr.DataArray(time, dims="time")),
        Pmodel_output.LastStepWriter(last_sink),
    )
    writer.write_chunk(slice(0, 2), slice(0, 3), chunk)

    expected = (
        xr.DataArray(chunk, dims=("lon", "lat", "var", "time"), coords={"time": time})
        .resample(time="1MS")
        .mean()
    )
    np.testing.assert_allclose(monthly_sink.chunk, expected.values, rtol=1e-12)
    np.testing.assert_array_equal(last_sink.chunk, chunk[..., -1:])


def test_aggregate_dataset_monthly_matches_resample():
    time = pd.date_range("2016-02-20", "2016-04-02", freq="D")
    rng = np.random.default_rng(2)
    dataset = xr.Dataset(
        {"adults": (("longitude", "latitude", "time"), rng.random((2, 2, time.size)))},
        coords={"longitude": [0.0, 1.0], "latitude": [0.0, 1.0], "time": time},
    )
    result = Pmodel_output.aggregate_dataset_monthly(dataset)
    expected = dataset.resample(time="1MS").mean()
    xr.testing.assert_allclose(
        result["adults"], expected["adults"].transpose(*dataset["adults"].dims)
    )
    assert result.attrs["time_aggregation"] == "monthly_mean"
    assert result["adults"].attrs["cell_methods"] == "time: mean"


def test_final_state_filepath_next_to_output(tmp_path):
    serving = {
        "path_output_datasets": str(tmp_path),
        "filename_components": {"prefix": "Mosq_", "suffix": "", "extension": ".nc"},
    }
    assert Pmodel_output.final_state_filepath(year=2014, **serving) == (
        tmp_path / "Mosq_2014_final_state.nc"
    )
