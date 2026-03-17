import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_output import (
    PmodelOutput,
    build_output_dataset,
    assemble_output_filepath,
    save_output_dataset,
)


# ---- Pytest Fixtures


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


# ---- Tests for PmodelOutput


def test_pmodel_output_repr_contains_shape():
    array = np.zeros((2, 3, 4, 5), dtype=np.float64)
    output = PmodelOutput(model_output=array)

    text = repr(output)
    assert "PmodelOutput" in text
    assert "model_output" in text
    assert "shape=(2, 3, 4, 5)" in text


# ---- Tests for build_output_dataset


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


# ---- Tests for output file utilities


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
