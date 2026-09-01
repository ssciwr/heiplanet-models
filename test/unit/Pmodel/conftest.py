"""Pytest configuration and shared fixtures for Pmodel unit tests."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_initial import (
    load_all_data,
    load_population_dataset,
    load_rainfall_dataset,
    load_temperature_dataset,
)


@pytest.fixture
def test_resources_path():
    """Return the path to the test/test_resources directory."""
    return Path(__file__).parent.parent.parent / "test_resources"


@pytest.fixture
def rng():
    """Seeded random generator shared by Pmodel unit tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def temp_dummy_data(test_resources_path):
    """Load the dummy temperature DataArray from test resources."""
    return xr.open_dataarray(test_resources_path / "temperature_dummy.nc")


@pytest.fixture
def dummy_file_rainfall_dataset(test_resources_path):
    """Load the dummy rainfall DataArray from test resources."""
    return xr.open_dataset(test_resources_path / "dataset_rainfall_dummy.nc").rainfall


@pytest.fixture
def dummy_file_population_dataset(test_resources_path):
    """Load the dummy population DataArray from test resources."""
    return xr.open_dataset(
        test_resources_path / "dataset_population_dummy.nc"
    ).population


@pytest.fixture
def test_etl_settings(test_resources_path):
    """
    Create a configuration dictionary based on global_settings_dummy.yaml
    but pointing to test/resources directory.
    """
    return {
        "ingestion": {
            "path_root_datasets": str(test_resources_path),
            "xarray_load_settings": {"chunks": "auto"},
            "filename_components": {
                "temperature_dataset": {
                    "prefix": "dataset_temperature_dummy",
                    "suffix": "",
                    "extension": ".nc",
                },
                "rainfall_dataset": {
                    "prefix": "dataset_rainfall_dummy",
                    "suffix": "",
                    "extension": ".nc",
                },
                "human_population_dataset": {
                    "prefix": "dataset_population_dummy",
                    "suffix": "",
                    "extension": ".nc",
                },
            },
            "initial_conditions": {"file_path_initial_conditions": None},
        },
        "transformation": {
            "temperature_dataset": {
                "data_variable": "temperature",
                "preprocessing": {
                    "names_dimensions": {"time": "time"},
                    "dimension_order": ["longitude", "latitude", "time"],
                },
            },
            "rainfall_dataset": {
                "data_variable": "rainfall",
                "preprocessing": {"dimension_order": ["longitude", "latitude", "time"]},
            },
            "human_population_dataset": {
                "data_variable": "population",
                "preprocessing": {
                    "names_dimensions": {
                        "longitude": "longitude",
                        "latitude": "latitude",
                    },
                    "dimension_order": ["longitude", "latitude", "time"],
                },
                "postprocessing": {"align_dataset": True},
            },
        },
        "serving": {
            "path_output_datasets": str(test_resources_path),
            "filename_components": {
                "prefix": "Mosquito_abundance_Global_",
                "suffix": "dummy",
                "extension": ".nc",
            },
        },
        "ode_system": {
            "time_step": 10,
            "model_variables": ["eggs", "ed", "juv", "imm", "adults"],
        },
    }


@pytest.fixture
def test_dataset_paths(test_resources_path):
    """Return paths to the test datasets."""
    return {
        "temperature_dataset": test_resources_path / "dataset_temperature_dummy.nc",
        "rainfall_dataset": test_resources_path / "dataset_rainfall_dummy.nc",
        "human_population_dataset": test_resources_path / "dataset_population_dummy.nc",
    }


@pytest.fixture
def loaded_temperature_dataset(test_dataset_paths, test_etl_settings):
    """Load the temperature dataset from test/resources using Pmodel_initial functions."""
    return load_temperature_dataset(
        path_dataset=test_dataset_paths["temperature_dataset"],
        **test_etl_settings,
    )


@pytest.fixture
def loaded_rainfall_dataset(test_dataset_paths, test_etl_settings):
    """Load the rainfall dataset from test/resources using Pmodel_initial functions."""
    return load_rainfall_dataset(
        path_dataset=test_dataset_paths["rainfall_dataset"],
        **test_etl_settings,
    )


@pytest.fixture
def loaded_population_dataset(test_dataset_paths, test_etl_settings):
    """Load the population dataset from test/resources using Pmodel_initial functions."""
    return load_population_dataset(
        path_dataset=test_dataset_paths["human_population_dataset"],
        **test_etl_settings,
    )


@pytest.fixture
def model_input_dummy_datasets(test_dataset_paths, test_etl_settings):
    """
    Load all datasets from test/resources using the load_all_data function.

    Returns:
        PmodelInput: An object containing all loaded and processed model input arrays.
    """
    return load_all_data(paths=test_dataset_paths, etl_settings=test_etl_settings)
