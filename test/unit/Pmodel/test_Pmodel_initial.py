"""Pytest unit tests for Pmodel_initial.py module functions.

Tests use dummy NetCDF datasets located in test/resources:
- dense_dummy.nc: human population density dataset
- pr_dummy.nc: rainfall dataset
- temperature_dummy.nc: temperature dataset
"""

import yaml
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd

from heiplanet_models.Pmodel import Pmodel_initial


# ===================================
# ===       Pytest Fixtures       ===
# ===================================
@pytest.fixture
def valid_yaml_file(tmp_path):
    """Fixture that creates a valid YAML file and returns its path."""
    data = {
        "ingestion": {
            "path_root_datasets": "/data",
            "xarray_load_settings": {"engine": "netcdf4"},
            "filename_components": {
                "temperature_dataset": {
                    "prefix": "temp_",
                    "suffix": "",
                    "extension": ".nc",
                }
            },
            "initial_conditions": {"file_path_initial_conditions": "/data/init.nc"},
        },
        "transformation": {},
        "ode_system": {},
    }
    file_path = tmp_path / "valid_settings.yaml"
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f)
    return file_path, data


# Additional fixtures for other test cases
@pytest.fixture
def invalid_yaml_file(tmp_path):
    file_path = tmp_path / "invalid.yaml"
    with open(file_path, "w") as f:
        f.write("ingestion: [unclosed_list\n")
    return file_path


@pytest.fixture
def empty_yaml_file(tmp_path):
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    return file_path


@pytest.fixture
def comments_only_yaml_file(tmp_path):
    file_path = tmp_path / "comments_only.yaml"
    file_path.write_text("# This is a comment\n# Another comment\n")
    return file_path


@pytest.fixture
def unicode_yaml_file(tmp_path):
    data = {"greeting": "こんにちは", "emoji": "😀"}
    file_path = tmp_path / "unicode.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return file_path, data


@pytest.fixture
def various_types_yaml_file(tmp_path):
    data = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "bool": True,
        "none": None,
    }
    file_path = tmp_path / "various_types.yaml"
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f)
    return file_path, data


@pytest.fixture
def existing_paths(tmp_path):
    """Fixture to create a dictionary of existing paths (a directory and a file)."""
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    file_path = tmp_path / "test_file.txt"
    file_path.touch()
    return {"directory": str(dir_path), "file": str(file_path)}


@pytest.fixture
def mixed_paths(tmp_path):
    """Fixture to create a dictionary of existing and non-existing paths."""
    existing_file = tmp_path / "existing_file.txt"
    existing_file.touch()
    return {
        "existing": str(existing_file),
        "non_existing": str(tmp_path / "non_existing_file.txt"),
    }


@pytest.fixture
def non_existing_paths(tmp_path):
    """Fixture to create a dictionary of non-existing paths."""
    return {
        "non_existing_1": str(tmp_path / "non_existing_file_1.txt"),
        "non_existing_2": str(tmp_path / "non_existing_dir" / "file.txt"),
    }


@pytest.fixture
def standard_etl_settings(tmp_path):
    """Fixture for standard ETL settings for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return {
        "ingestion": {
            "path_root_datasets": str(data_dir),
            "filename_components": {
                "temperature": {
                    "prefix": "temp_",
                    "suffix": "_daily",
                    "extension": ".nc",
                },
                "rainfall": {"prefix": "rain_", "suffix": None, "extension": ".nc"},
            },
        }
    }


@pytest.fixture
def valid_netcdf_file(tmp_path):
    """Fixture to create a valid NetCDF file and return its path and original dataset."""
    file_path = tmp_path / "test_dataset.nc"
    rng = np.random.default_rng(43)
    original_ds = xr.Dataset(
        {"temperature": (("lat", "lon"), rng.random((2, 3)))},
        coords={"lat": [10, 20], "lon": [100, 110, 120]},
    )
    original_ds.to_netcdf(file_path)
    return str(file_path), original_ds


@pytest.fixture
def invalid_format_file(tmp_path):
    """Fixture to create a simple text file, which is an invalid format."""
    file_path = tmp_path / "invalid_format.txt"
    file_path.write_text("this is not a netcdf file")
    return str(file_path)


@pytest.fixture
def corrupted_netcdf_file(tmp_path):
    """Fixture to create a corrupted NetCDF file."""
    file_path = tmp_path / "corrupted.nc"
    with open(file_path, "wb") as f:
        f.write(b"this is not a valid netcdf file header")
    return str(file_path)


@pytest.fixture
def empty_file(tmp_path):
    """Fixture to create an empty file."""
    file_path = tmp_path / "empty.nc"
    file_path.touch()
    return str(file_path)


@pytest.fixture
def sample_dataset():
    """Fixture to create a sample xarray.Dataset for preprocessing tests."""
    rng = np.random.default_rng(43)
    return xr.Dataset(
        {"data_var": (("x", "y"), rng.random((2, 3)))},
        coords={"x": [1, 2], "y": [10, 20, 30]},
    )


@pytest.fixture
def alignment_datasets():
    """Fixture to create a dataset to be aligned and a reference dataset."""
    # Dataset with a coarse grid
    misaligned_ds = xr.Dataset(
        {"data": (("latitude", "longitude"), np.arange(9).reshape(3, 3))},
        coords={"latitude": [0, 5, 10], "longitude": [0, 5, 10]},
    )
    # Dataset with a finer grid to align to
    reference_ds = xr.Dataset(
        {"data": (("latitude", "longitude"), np.arange(9).reshape(3, 3))},
        coords={"latitude": [5, 10, 15], "longitude": [5, 10, 15]},
    )
    return misaligned_ds, reference_ds


@pytest.fixture
def multidim_alignment_datasets():
    """Fixture for datasets with an extra 'time' dimension."""
    misaligned_ds = xr.Dataset(
        {"data": (("time", "latitude", "longitude"), np.arange(8).reshape(2, 2, 2))},
        coords={"time": [1, 2], "latitude": [0, 10], "longitude": [0, 10]},
    )
    fixed_ds = xr.Dataset(coords={"latitude": [0, 5, 10], "longitude": [0, 5, 10]})
    return misaligned_ds, fixed_ds


@pytest.fixture
def temperature_etl_settings():
    """Fixture for temperature-specific ETL settings."""
    return {
        "ingestion": {"xarray_load_settings": {"engine": "netcdf4"}},
        "transformation": {
            "temperature_dataset": {
                "preprocessing": {
                    "names_dimensions": {"x": "latitude", "y": "longitude"},
                    "dimension_order": ("latitude", "longitude"),
                }
            }
        },
    }


@pytest.fixture
def rainfall_etl_settings():
    """Fixture for rainfall-specific ETL settings with preprocessing."""
    return {
        "ingestion": {"xarray_load_settings": {"engine": "netcdf4"}},
        "transformation": {
            "rainfall_dataset": {
                "preprocessing": {
                    "names_dimensions": {"lat": "latitude", "lon": "longitude"},
                }
            }
        },
    }


@pytest.fixture
def population_etl_settings():
    """Fixture for population-specific ETL settings with preprocessing."""
    return {
        "ingestion": {"xarray_load_settings": {"engine": "netcdf4"}},
        "transformation": {
            "human_population_dataset": {
                "preprocessing": {
                    "names_dimensions": {"lat": "latitude", "lon": "longitude"},
                }
            }
        },
    }


@pytest.fixture
def temperature_daily_etl_settings():
    """Pytest fixture for ETL settings for create_temperature_daily function."""
    return {
        "ode_system": {"time_step": 3},
        "transformation": {"temperature_dataset": {"data_variable": "t2m"}},
    }


@pytest.fixture
def sample_temperature_dataset():
    """Pytest fixture for a sample temperature xarray Dataset."""
    rng = np.random.default_rng(43)
    return xr.Dataset(
        {
            "t2m": (
                ("time", "latitude", "longitude"),
                rng.random((2, 3, 4)),
            )
        },
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "latitude": [10, 20, 30],
            "longitude": [-10, 0, 10, 20],
        },
    )


@pytest.fixture
def mock_etl_settings():
    """Fixture that provides a mock ETL settings dictionary."""
    return {"ode_system": {"model_variables": ["S", "E", "I", "R"]}}


@pytest.fixture
def initial_conditions_netcdf_file(tmp_path):
    """Fixture that creates a valid NetCDF file for initial conditions."""
    file_path = tmp_path / "initial_conditions.nc"

    # Define dimensions and create sample data
    lon = np.array([10.0, 20.0])
    lat = np.array([30.0, 40.0])
    time = pd.to_datetime(["2023-01-01", "2023-01-02"])

    # Create a dataset with variables matching model_variables
    rng = np.random.default_rng(43)
    ds = xr.Dataset(
        {
            "S": (("time", "longitude", "latitude"), rng.random((2, 2, 2))),
            "E": (("time", "longitude", "latitude"), np.full((2, 2, 2), 0.5)),
            "I": (("time", "longitude", "latitude"), np.ones((2, 2, 2))),
            "R": (("time", "longitude", "latitude"), np.zeros((2, 2, 2))),
        },
        coords={"time": time, "longitude": lon, "latitude": lat},
    )

    ds.to_netcdf(file_path)

    # Return the file path and the expected data (last time step)
    expected_data = {var: ds[var].isel(time=-1).values for var in ["S", "E", "I", "R"]}
    return file_path, expected_data


@pytest.fixture
def missing_variable_netcdf_file(tmp_path):
    """Fixture that creates a NetCDF file missing a required variable."""
    file_path = tmp_path / "missing_variable.nc"
    rng = np.random.default_rng(43)
    ds = xr.Dataset(
        {
            "S": (("time", "longitude", "latitude"), rng.random((2, 2, 2))),
            # "E" is intentionally missing
            "I": (("time", "longitude", "latitude"), np.ones((2, 2, 2))),
            "R": (("time", "longitude", "latitude"), np.zeros((2, 2, 2))),
        },
        coords={
            "time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "longitude": [10.0, 20.0],
            "latitude": [30.0, 40.0],
        },
    )
    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def extra_variable_netcdf_file(tmp_path):
    """Fixture that creates a NetCDF file with an extra variable."""
    file_path = tmp_path / "extra_variable.nc"
    rng = np.random.default_rng(43)
    ds = xr.Dataset(
        {
            "S": (("time", "longitude", "latitude"), rng.random((2, 2, 2))),
            "E": (("time", "longitude", "latitude"), np.full((2, 2, 2), 0.5)),
            "I": (("time", "longitude", "latitude"), np.ones((2, 2, 2))),
            "R": (("time", "longitude", "latitude"), np.zeros((2, 2, 2))),
            "Z_extra": (
                ("time", "longitude", "latitude"),
                np.ones((2, 2, 2)),
            ),  # Extra variable
        },
        coords={
            "time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "longitude": [10.0, 20.0],
            "latitude": [30.0, 40.0],
        },
    )
    ds.to_netcdf(file_path)
    expected_data = {var: ds[var].isel(time=-1).values for var in ["S", "E", "I", "R"]}
    return file_path, expected_data


@pytest.fixture
def full_etl_settings():
    """Provides a complete ETL settings dictionary for load_all_data tests."""
    return {
        "transformation": {
            "human_population_dataset": {
                "postprocessing": {"align_dataset": True},
                "data_variable": "population_density",
            },
            "temperature_dataset": {"data_variable": "t2m"},
            "rainfall_dataset": {"data_variable": "tp"},
        },
        "ode_system": {"time_step": 365, "model_variables": ["S", "E", "I", "R"]},
        "ingestion": {
            "initial_conditions": {"file_path_initial_conditions": "/fake/path/init.nc"}
        },
    }


@pytest.fixture
def mock_data_paths():
    """Provides a dictionary of mock file paths."""
    return {
        "temperature_dataset": "/fake/path/temp.nc",
        "rainfall_dataset": "/fake/path/rain.nc",
        "human_population_dataset": "/fake/path/pop.nc",
    }


@pytest.fixture
def mock_model_inputs():
    """Provides a dictionary of mock xarray Datasets and numpy arrays for model inputs."""
    coords = {
        "longitude": [10, 20],
        "latitude": [30, 40],
        "time": [pd.to_datetime("2024-01-01")],
    }

    rng = np.random.default_rng(43)
    temperature_ds = xr.Dataset(
        {"t2m": (("time", "longitude", "latitude"), rng.random((1, 2, 2)))},
        coords=coords,
    )
    rainfall_ds = xr.Dataset(
        {"tp": (("longitude", "latitude"), rng.random((2, 2)))}, coords=coords
    )
    population_ds = xr.Dataset(
        {"population_density": (("longitude", "latitude"), rng.random((2, 2)))},
        coords=coords,
    )

    # Mock data returned by processing functions
    da_temp_daily = xr.DataArray(rng.random((730, 2, 2)), name="temperature_daily")
    da_temp_mean = temperature_ds["t2m"]
    initial_conditions_arr = np.zeros((2, 2, 4))

    return {
        "temperature": temperature_ds,
        "rainfall": rainfall_ds,
        "population": population_ds,
        "processed_population": population_ds.copy(),  # Simulate post-processing
        "da_temp_daily": da_temp_daily,
        "da_temp_mean": da_temp_mean,
        "initial_conditions": initial_conditions_arr,
    }


@pytest.fixture
def dummy_file_initial_conditions_path():
    """Loads the initial conditions dummy file dataset from test resources."""
    resources_dir = Path(__file__).parent.parent.parent / "resources"
    dataset_path = resources_dir / "initial_conditions_dummy.nc"
    return dataset_path


@pytest.fixture
def dummy_file_etl_settings_yaml():
    """Loads the initial conditions dummy file dataset from test resources."""
    resources_dir = Path(__file__).parent.parent.parent / "resources"
    dataset_path = resources_dir / "global_settings_dummy.yaml"
    return Pmodel_initial.read_global_settings(dataset_path)


@pytest.fixture
def dummy_filepath_datasets(dummy_file_etl_settings_yaml):
    paths = Pmodel_initial.assemble_filepaths(**dummy_file_etl_settings_yaml)

    return paths


@pytest.fixture
def dummy_input_data(dummy_filepath_datasets, dummy_file_etl_settings_yaml):
    input_data = Pmodel_initial.load_all_data(
        paths=dummy_filepath_datasets, etl_settings=dummy_file_etl_settings_yaml
    )

    return input_data


# ===================================
# ===         Unit tests          ===
# ===================================
# ---- Unit Tests for read_global_settings
def test_read_global_settings_valid_yaml(valid_yaml_file):
    """Test that read_global_settings correctly loads a valid YAML file."""
    file_path, expected_data = valid_yaml_file
    result = Pmodel_initial.read_global_settings(str(file_path))
    assert result == expected_data


def test_read_global_settings_file_not_found():
    """Test that FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError) as excinfo:
        Pmodel_initial.read_global_settings("/nonexistent/path/to/file.yaml")
    assert (
        "No such file or directory" in str(excinfo.value)
        or "not found" in str(excinfo.value).lower()
    )


def test_read_global_settings_invalid_yaml(invalid_yaml_file):
    """Test that yaml.YAMLError is raised for invalid YAML syntax."""
    import yaml as _yaml

    with pytest.raises(_yaml.YAMLError):
        Pmodel_initial.read_global_settings(str(invalid_yaml_file))


def test_read_global_settings_empty_yaml(empty_yaml_file):
    """Test that an empty YAML file returns None."""
    result = Pmodel_initial.read_global_settings(str(empty_yaml_file))
    assert result is None


def test_read_global_settings_comments_only_yaml(comments_only_yaml_file):
    """Test that a YAML file with only comments returns None."""
    result = Pmodel_initial.read_global_settings(str(comments_only_yaml_file))
    assert result is None


def test_read_global_settings_unicode_yaml(unicode_yaml_file):
    """Test that a YAML file with Unicode content loads correctly."""
    file_path, expected_data = unicode_yaml_file
    result = Pmodel_initial.read_global_settings(str(file_path))
    assert result == expected_data


def test_read_global_settings_various_types_yaml(various_types_yaml_file):
    """Test that a YAML file with various data types loads correctly."""
    file_path, expected_data = various_types_yaml_file
    result = Pmodel_initial.read_global_settings(str(file_path))
    assert result == expected_data


# ----  Unit tests for check_all_paths_exist
def test_check_all_paths_exist_all_paths_exist(existing_paths):
    """Test check_all_paths_exist returns True when all paths are valid."""
    assert Pmodel_initial.check_all_paths_exist(existing_paths) is True


def test_check_all_paths_exist_one_non_existing(mixed_paths):
    """Test check_all_paths_exist returns False if one path is invalid."""
    assert Pmodel_initial.check_all_paths_exist(mixed_paths) is False


def test_check_all_paths_exist_all_non_existing(non_existing_paths):
    """Test check_all_paths_exist returns False when all paths are invalid."""
    assert Pmodel_initial.check_all_paths_exist(non_existing_paths) is False


def test_check_all_paths_exist_empty_dict():
    """Test check_all_paths_exist returns True for an empty dictionary."""
    assert Pmodel_initial.check_all_paths_exist({}) is False


def test_check_all_paths_exist_logs_warning_for_non_existing(
    non_existing_paths, caplog
):
    """Test that check_all_paths_exist logs WARNING for non-existing paths."""
    with caplog.at_level("WARNING"):
        Pmodel_initial.check_all_paths_exist(non_existing_paths)
        for key, path in non_existing_paths.items():
            assert f"Path for '{key}': {path} ... Not Found" in caplog.text


# ----  Unit tests for assemble_filepaths
def test_assemble_filepaths_standard_input(standard_etl_settings):
    """Test assemble_filepaths with a standard year and configuration."""
    year = 2024
    result = Pmodel_initial.assemble_filepaths(year, **standard_etl_settings)

    expected_root = Path(standard_etl_settings["ingestion"]["path_root_datasets"])
    expected_paths = {
        "temperature": expected_root / "temp_2024_daily.nc",
        "rainfall": expected_root / "rain_2024.nc",
    }

    assert result == expected_paths
    assert isinstance(result["temperature"], Path)
    assert isinstance(result["rainfall"], Path)


def test_assemble_filepaths_different_year(standard_etl_settings):
    """Test that the year is correctly inserted into filenames."""
    year = 2025
    result = Pmodel_initial.assemble_filepaths(year, **standard_etl_settings)
    expected_root = Path(standard_etl_settings["ingestion"]["path_root_datasets"])
    expected_paths = {
        "temperature": expected_root / "temp_2025_daily.nc",
        "rainfall": expected_root / "rain_2025.nc",
    }
    assert result == expected_paths


def test_assemble_filepaths_empty_filename_components(standard_etl_settings):
    """Test that an empty dictionary is returned for empty filename_components."""
    standard_etl_settings["ingestion"]["filename_components"] = {}
    result = Pmodel_initial.assemble_filepaths(2024, **standard_etl_settings)
    assert result == {}


def test_assemble_filepaths_raises_key_error_for_missing_ingestion():
    """Test that a KeyError is raised if 'ingestion' key is missing."""
    with pytest.raises(KeyError) as excinfo:
        Pmodel_initial.assemble_filepaths(2024, **{})
    assert "ingestion" in str(excinfo.value)


def test_assemble_filepaths_raises_key_error_for_missing_path_root():
    """Test that a KeyError is raised if 'path_root_datasets' is missing."""
    etl_settings = {"ingestion": {"filename_components": {}}}
    with pytest.raises(KeyError) as excinfo:
        Pmodel_initial.assemble_filepaths(2024, **etl_settings)
    assert "path_root_datasets" in str(excinfo.value)


def test_assemble_filepaths_raises_key_error_for_malformed_components(
    standard_etl_settings,
):
    """Test that a KeyError is raised if a component is missing a required key."""
    # Missing 'prefix' key
    standard_etl_settings["ingestion"]["filename_components"]["temperature"].pop(
        "prefix"
    )
    with pytest.raises(KeyError) as excinfo:
        Pmodel_initial.assemble_filepaths(2024, **standard_etl_settings)
    assert "prefix" in str(excinfo.value)


@pytest.mark.parametrize("invalid_year", ["2024", 2024.0])
def test_assemble_filepaths_raises_type_error_for_non_integer_year(
    standard_etl_settings, invalid_year
):
    """Test that a TypeError is raised for a non-integer year."""
    with pytest.raises(TypeError) as excinfo:
        Pmodel_initial.assemble_filepaths(invalid_year, **standard_etl_settings)
    assert "not an integer" in str(excinfo.value) or excinfo.type is TypeError


# ---- Unit tests for load_dataset
def test_load_dataset_valid_netcdf_string_path(valid_netcdf_file):
    """Test load_dataset with a valid NetCDF file path as a string."""
    file_path, original_ds = valid_netcdf_file
    loaded_ds = Pmodel_initial.load_dataset(file_path)
    assert isinstance(loaded_ds, xr.Dataset)
    xr.testing.assert_equal(loaded_ds, original_ds)


def test_load_dataset_valid_netcdf_pathlib_path(valid_netcdf_file):
    """Test load_dataset with a valid NetCDF file path as a pathlib.Path object."""
    file_path_str, original_ds = valid_netcdf_file
    file_path_obj = Path(file_path_str)
    loaded_ds = Pmodel_initial.load_dataset(file_path_obj)
    assert isinstance(loaded_ds, xr.Dataset)
    xr.testing.assert_equal(loaded_ds, original_ds)


def test_load_dataset_file_not_found():
    """Test that FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError) as excinfo:
        Pmodel_initial.load_dataset("/nonexistent/path/to/file.nc")
    assert (
        "No such file or directory" in str(excinfo.value)
        or "not found" in str(excinfo.value).lower()
    )


def test_load_dataset_invalid_file_format(invalid_format_file):
    """Test that an error is raised for a file with an invalid format."""
    with pytest.raises(ValueError) as excinfo:
        Pmodel_initial.load_dataset(invalid_format_file)
    assert "cannot be parsed" in str(excinfo.value) or excinfo.type is ValueError


def test_load_dataset_corrupted_file(corrupted_netcdf_file):
    """Test that an error is raised when loading a corrupted file."""
    with pytest.raises((OSError, ValueError)) as excinfo:
        Pmodel_initial.load_dataset(corrupted_netcdf_file)
    assert excinfo.type in (OSError, ValueError)


def test_load_dataset_with_kwargs(valid_netcdf_file):
    """Test that kwargs are correctly passed to xarray.open_dataset."""
    file_path, _ = valid_netcdf_file
    loaded_ds = Pmodel_initial.load_dataset(file_path, chunks={"lat": 1})
    assert loaded_ds.temperature.chunks is not None
    import dask.array

    assert isinstance(loaded_ds.temperature.data, dask.array.Array)


def test_load_dataset_empty_file(empty_file):
    """Test that an error is raised when loading an empty file."""
    with pytest.raises((ValueError, OSError)) as excinfo:
        Pmodel_initial.load_dataset(empty_file)
    assert excinfo.type in (OSError, ValueError)


# ---- Unit tests for preprocess_dataset
def test_preprocess_dataset_no_args(sample_dataset):
    """Test that the dataset is returned unchanged with no kwargs."""
    original_ds = sample_dataset
    processed_ds = Pmodel_initial.preprocess_dataset(original_ds)
    xr.testing.assert_identical(processed_ds, original_ds)


def test_preprocess_dataset_rename_only(sample_dataset):
    """Test that dimensions are correctly renamed."""
    rename_map = {"x": "latitude", "y": "longitude"}
    processed_ds = Pmodel_initial.preprocess_dataset(
        sample_dataset, names_dimensions=rename_map
    )
    assert "latitude" in processed_ds.dims
    assert "longitude" in processed_ds.dims
    assert "x" not in processed_ds.dims
    assert "y" not in processed_ds.dims


def test_preprocess_dataset_transpose_only(sample_dataset):
    """Test that dimensions are correctly transposed."""
    order = ("y", "x")
    processed_ds = Pmodel_initial.preprocess_dataset(
        sample_dataset, dimension_order=order
    )
    assert processed_ds.data_var.dims == order


def test_preprocess_dataset_rename_and_transpose(sample_dataset):
    """Test both renaming and transposing in one call."""
    rename_map = {"x": "latitude", "y": "longitude"}
    order = ("longitude", "latitude")
    processed_ds = Pmodel_initial.preprocess_dataset(
        sample_dataset, names_dimensions=rename_map, dimension_order=order
    )
    assert processed_ds.data_var.dims == order


def test_preprocess_dataset_rename_raises_key_error(sample_dataset):
    """Test that a ValueError is raised for a non-existent dimension in rename map."""
    rename_map = {"z": "time"}
    with pytest.raises(Exception) as excinfo:
        Pmodel_initial.preprocess_dataset(sample_dataset, names_dimensions=rename_map)
    assert "z" in str(excinfo.value)


def test_preprocess_dataset_transpose_raises_error_non_existent_dim(sample_dataset):
    """Test that an error is raised if a dimension in the order does not exist."""
    order = ("x", "y", "z")
    with pytest.raises(Exception) as excinfo:
        Pmodel_initial.preprocess_dataset(sample_dataset, dimension_order=order)
    assert "z" in str(excinfo.value)


def test_preprocess_dataset_transpose_raises_error_incomplete_order(sample_dataset):
    """Test that a ValueError is raised for an incomplete dimension order."""
    order = ("x",)
    with pytest.raises(Exception) as excinfo:
        Pmodel_initial.preprocess_dataset(sample_dataset, dimension_order=order)
    assert "order" in str(excinfo.value) or excinfo.type is Exception


def test_preprocess_dataset_with_empty_dataset():
    """Test that the function handles an empty dataset gracefully."""
    empty_ds = xr.Dataset()
    processed_ds = Pmodel_initial.preprocess_dataset(empty_ds)
    xr.testing.assert_identical(processed_ds, empty_ds)


# ---- Unit tests for postprocess_dataset
def test_postprocess_dataset_no_args(sample_dataset):
    """Test that the dataset is returned unchanged with no kwargs."""
    original_ds = sample_dataset
    processed_ds = Pmodel_initial.postprocess_dataset(original_ds)
    xr.testing.assert_identical(processed_ds, original_ds)


def test_postprocess_dataset_align_false(alignment_datasets):
    """Test that no alignment occurs if align_dataset is False."""
    misaligned_ds, reference_ds = alignment_datasets
    processed_ds = Pmodel_initial.postprocess_dataset(
        misaligned_ds, reference_dataset=reference_ds, align_dataset=False
    )
    xr.testing.assert_identical(processed_ds, misaligned_ds)


def test_postprocess_dataset_no_reference_dataset(sample_dataset):
    """Test that no alignment occurs if reference_dataset is None."""
    processed_ds = Pmodel_initial.postprocess_dataset(
        sample_dataset, reference_dataset=None, align_dataset=True
    )
    xr.testing.assert_identical(processed_ds, sample_dataset)


def test_postprocess_dataset_successful_alignment(alignment_datasets):
    """Test that the dataset is correctly aligned to the reference grid."""
    misaligned_ds, reference_ds = alignment_datasets
    processed_ds = Pmodel_initial.postprocess_dataset(
        misaligned_ds, reference_dataset=reference_ds, align_dataset=True
    )
    # Check that the coordinates now match the reference dataset
    xr.testing.assert_equal(processed_ds.coords, reference_ds.coords)
    # Check that the data variable has been interpolated to the new shape
    assert processed_ds["data"].shape == (3, 3)


def test_postprocess_dataset_alignment_with_identical_grids(alignment_datasets):
    """Test alignment when grids are already identical."""
    _, reference_ds = alignment_datasets
    # Create a dataset that already has the same grid as the reference
    aligned_ds = reference_ds.copy()
    rng = np.random.default_rng(43)
    aligned_ds["data"] = (("latitude", "longitude"), rng.random((3, 3)))

    processed_ds = Pmodel_initial.postprocess_dataset(
        aligned_ds, reference_dataset=reference_ds, align_dataset=True
    )
    xr.testing.assert_allclose(processed_ds, aligned_ds)


def test_postprocess_dataset_error_during_alignment(alignment_datasets, monkeypatch):
    """Test that an exception during alignment is propagated."""
    misaligned_ds, reference_ds = alignment_datasets

    # Mock the alignment function to raise an error
    def mock_align_raises_error(*args, **kwargs):
        raise ValueError("Alignment failed")

    monkeypatch.setattr(
        Pmodel_initial, "align_xarray_datasets", mock_align_raises_error
    )

    with pytest.raises(ValueError, match="Alignment failed") as excinfo:
        Pmodel_initial.postprocess_dataset(
            misaligned_ds, reference_dataset=reference_ds, align_dataset=True
        )
    assert "Alignment failed" in str(excinfo.value)


def test_postprocess_dataset_with_empty_dataset(alignment_datasets):
    """Test that an empty dataset is handled gracefully during alignment."""
    _, reference_ds = alignment_datasets
    empty_ds = xr.Dataset()
    processed_ds = Pmodel_initial.postprocess_dataset(
        empty_ds, reference_dataset=reference_ds, align_dataset=True
    )
    # The result should be an empty dataset with the reference coordinates
    assert not processed_ds.data_vars
    xr.testing.assert_equal(processed_ds.coords, reference_ds.coords)


# ---- Unit tests for align_xarray_datasets
def test_align_xarray_datasets_successful_alignment(alignment_datasets):
    """Test that a dataset is correctly aligned to a reference grid."""
    misaligned_ds, fixed_ds = alignment_datasets
    aligned_ds = Pmodel_initial.align_xarray_datasets(misaligned_ds, fixed_ds)

    # Check that coordinates match the fixed dataset
    xr.testing.assert_equal(aligned_ds.coords, fixed_ds.coords)
    # Check that the data variable has been interpolated to the new shape
    assert aligned_ds["data"].shape == (3, 3)
    # Check a specific interpolated value
    # The center point (5, 5) should be the average of the four original points
    expected_center_value = misaligned_ds["data"].values.mean()
    actual_center_value = aligned_ds["data"].sel(latitude=5, longitude=5).item()
    assert actual_center_value == expected_center_value


def test_align_xarray_datasets_empty_misaligned(alignment_datasets):
    """Test alignment with an empty misaligned dataset."""
    _, fixed_ds = alignment_datasets
    empty_ds = xr.Dataset()
    aligned_ds = Pmodel_initial.align_xarray_datasets(empty_ds, fixed_ds)

    assert not aligned_ds.data_vars
    xr.testing.assert_equal(aligned_ds.coords, fixed_ds.coords)


def test_align_xarray_datasets_identical_grids(alignment_datasets):
    """Test alignment when grids are already identical."""
    misaligned_ds, _ = alignment_datasets
    # The misaligned_ds is used as both, since it has a valid grid
    aligned_ds = Pmodel_initial.align_xarray_datasets(misaligned_ds, misaligned_ds)
    xr.testing.assert_identical(aligned_ds, misaligned_ds)


def test_align_xarray_datasets_with_extra_dims(multidim_alignment_datasets):
    """Test that non-spatial dimensions are preserved during alignment."""
    misaligned_ds, fixed_ds = multidim_alignment_datasets
    aligned_ds = Pmodel_initial.align_xarray_datasets(misaligned_ds, fixed_ds)

    # Check that spatial dimensions are aligned
    assert "latitude" in aligned_ds.dims
    assert "longitude" in aligned_ds.dims
    assert aligned_ds["latitude"].size == 3
    assert aligned_ds["longitude"].size == 3

    # Check that the 'time' dimension is preserved
    assert "time" in aligned_ds.dims
    xr.testing.assert_equal(aligned_ds["time"], misaligned_ds["time"])
    assert aligned_ds["data"].shape == (2, 3, 3)


def test_align_xarray_datasets_missing_coords_in_fixed(alignment_datasets):
    """Test for AttributeError when fixed_dataset is missing coordinates."""
    misaligned_ds, _ = alignment_datasets
    # Create a fixed_dataset with no coordinates
    invalid_fixed_ds = xr.Dataset()
    with pytest.raises(AttributeError) as excinfo:
        Pmodel_initial.align_xarray_datasets(misaligned_ds, invalid_fixed_ds)
    assert excinfo.type is AttributeError


def test_align_xarray_datasets_wrong_dims_in_misaligned(alignment_datasets):
    """Test for ValueError when misaligned_dataset has wrong dimension names."""
    _, fixed_ds = alignment_datasets
    # Create a dataset with 'x' and 'y' instead of 'latitude' and 'longitude'
    rng = np.random.default_rng(43)
    wrong_dims_ds = xr.Dataset(
        {"data": (("x", "y"), rng.random((2, 2)))}, coords={"x": [0, 1], "y": [0, 1]}
    )
    with pytest.raises(ValueError) as excinfo:
        Pmodel_initial.align_xarray_datasets(wrong_dims_ds, fixed_ds)
    assert excinfo.type is ValueError


# ---- Unit tests for load_temperature_dataset
def test_load_temperature_dataset_happy_path(
    temperature_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test successful loading and preprocessing of the temperature dataset."""
    file_path, _ = valid_netcdf_file

    # Define a dummy preprocessing function
    def mock_preprocess(dataset, **kwargs):
        # Simulate renaming dimensions as in the original test
        return dataset.rename({"lat": "latitude", "lon": "longitude"})

    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    # The function under test calls the real load_dataset and the mocked preprocess
    result_ds = Pmodel_initial.load_temperature_dataset(
        file_path, **temperature_etl_settings
    )

    # Verify the result has the new dimensions
    assert "latitude" in result_ds.dims
    assert "longitude" in result_ds.dims
    assert (
        "temp" not in result_ds.dims
    )  # Assuming 'temperature' is the var name in valid_netcdf_file
    assert "temperature" in result_ds.data_vars


def test_load_temperature_dataset_load_error(temperature_etl_settings, monkeypatch):
    """Test that an error from load_dataset is propagated."""
    mock_path = "/fake/path/temp.nc"

    def mock_load_raises_error(*args, **kwargs):
        raise FileNotFoundError("File not found")

    monkeypatch.setattr(Pmodel_initial, "load_dataset", mock_load_raises_error)

    with pytest.raises(FileNotFoundError, match="File not found"):
        Pmodel_initial.load_temperature_dataset(mock_path, **temperature_etl_settings)


def test_load_temperature_dataset_preprocess_error(
    temperature_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test that an error from preprocess_dataset is propagated."""
    file_path, _ = valid_netcdf_file

    def mock_preprocess_raises_error(*args, **kwargs):
        raise KeyError("Bad config")

    monkeypatch.setattr(
        Pmodel_initial, "preprocess_dataset", mock_preprocess_raises_error
    )

    with pytest.raises(KeyError, match="Bad config"):
        Pmodel_initial.load_temperature_dataset(file_path, **temperature_etl_settings)


def test_load_temperature_dataset_missing_load_settings(
    temperature_etl_settings, monkeypatch
):
    """Test that a KeyError is raised for missing xarray_load_settings."""
    mock_path = "/fake/path/temp.nc"
    # Mock load_dataset to prevent FileNotFoundError
    monkeypatch.setattr(Pmodel_initial, "load_dataset", lambda *args, **kwargs: None)
    del temperature_etl_settings["ingestion"]["xarray_load_settings"]
    with pytest.raises(KeyError):
        Pmodel_initial.load_temperature_dataset(mock_path, **temperature_etl_settings)


def test_load_temperature_dataset_no_preprocessing(
    temperature_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test that preprocessing is skipped if config is missing."""
    file_path, original_ds = valid_netcdf_file
    preprocess_called = False

    def mock_preprocess(*args, **kwargs):
        nonlocal preprocess_called
        preprocess_called = True

    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    # Remove the preprocessing key to test the conditional logic
    del temperature_etl_settings["transformation"]["temperature_dataset"][
        "preprocessing"
    ]

    result_ds = Pmodel_initial.load_temperature_dataset(
        file_path, **temperature_etl_settings
    )

    # Assert that preprocessing was NOT called
    assert not preprocess_called
    # Assert that the original, unmodified dataset is returned
    xr.testing.assert_identical(result_ds, original_ds)


def test_load_temperature_dataset_missing_transform_config(
    temperature_etl_settings, monkeypatch
):
    """Test KeyError for missing transformation:temperature_dataset."""
    mock_path = "/fake/path/temp.nc"
    monkeypatch.setattr(
        Pmodel_initial, "load_dataset", lambda *args, **kwargs: xr.Dataset()
    )

    del temperature_etl_settings["transformation"]["temperature_dataset"]

    with pytest.raises(KeyError):
        Pmodel_initial.load_temperature_dataset(mock_path, **temperature_etl_settings)


# ---- Unit tests for load_rainfall_dataset
def test_load_rainfall_dataset_happy_path(
    rainfall_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test successful loading and preprocessing of the rainfall dataset."""
    file_path, original_ds = valid_netcdf_file
    preprocess_called = False

    def mock_preprocess(dataset, **kwargs):
        """Mock preprocess_dataset to verify it's called with correct args."""
        nonlocal preprocess_called
        preprocess_called = True
        # Ensure the correct preprocessing params are passed
        assert (
            kwargs
            == rainfall_etl_settings["transformation"]["rainfall_dataset"][
                "preprocessing"
            ]
        )
        # Return a modified dataset to confirm it was processed
        return dataset.rename(kwargs.get("names_dimensions", {}))

    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    # Call the function under test
    result_ds = Pmodel_initial.load_rainfall_dataset(file_path, **rainfall_etl_settings)

    # Assert that preprocessing was called
    assert preprocess_called

    # Assert that the returned dataset is the one from the (mocked) preprocessor
    assert "latitude" in result_ds.dims
    assert "longitude" in result_ds.dims
    assert "lat" not in result_ds.dims
    assert "lon" not in result_ds.dims
    # Check that the data variable from the original file is still there
    assert list(original_ds.data_vars)[0] in result_ds.data_vars


def test_load_rainfall_dataset_no_preprocessing(
    rainfall_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test that preprocessing is skipped if config is missing."""
    file_path, original_ds = valid_netcdf_file
    preprocess_called = False

    def mock_preprocess(*args, **kwargs):
        nonlocal preprocess_called
        preprocess_called = True

    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    # Remove the preprocessing key to test the conditional logic
    del rainfall_etl_settings["transformation"]["rainfall_dataset"]["preprocessing"]

    result_ds = Pmodel_initial.load_rainfall_dataset(file_path, **rainfall_etl_settings)

    # Assert that preprocessing was NOT called
    assert not preprocess_called
    # Assert that the original, unmodified dataset is returned
    xr.testing.assert_identical(result_ds, original_ds)


def test_load_rainfall_dataset_load_error(rainfall_etl_settings, monkeypatch):
    """Test that an error from load_dataset is propagated."""
    mock_path = "/fake/path/rain.nc"
    preprocess_called = False

    def mock_load_raises_error(*args, **kwargs):
        raise FileNotFoundError("File not found")

    def mock_preprocess(*args, **kwargs):
        nonlocal preprocess_called
        preprocess_called = True

    monkeypatch.setattr(Pmodel_initial, "load_dataset", mock_load_raises_error)
    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    with pytest.raises(FileNotFoundError, match="File not found"):
        Pmodel_initial.load_rainfall_dataset(mock_path, **rainfall_etl_settings)

    assert not preprocess_called


def test_load_rainfall_dataset_preprocess_error(
    rainfall_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test that an error from preprocess_dataset is propagated."""
    file_path, _ = valid_netcdf_file

    def mock_preprocess_raises_error(*args, **kwargs):
        raise ValueError("Preprocessing failed")

    monkeypatch.setattr(
        Pmodel_initial, "preprocess_dataset", mock_preprocess_raises_error
    )

    with pytest.raises(ValueError, match="Preprocessing failed"):
        Pmodel_initial.load_rainfall_dataset(file_path, **rainfall_etl_settings)


def test_load_rainfall_dataset_missing_load_config(rainfall_etl_settings, monkeypatch):
    """Test KeyError for missing ingestion:xarray_load_settings."""
    mock_path = "/fake/path/rain.nc"
    # Mock load_dataset to prevent it from running and causing a different error
    monkeypatch.setattr(Pmodel_initial, "load_dataset", lambda *args, **kwargs: None)

    del rainfall_etl_settings["ingestion"]["xarray_load_settings"]

    with pytest.raises(KeyError):
        Pmodel_initial.load_rainfall_dataset(mock_path, **rainfall_etl_settings)


def test_load_rainfall_dataset_missing_transform_config(
    rainfall_etl_settings, monkeypatch
):
    """Test KeyError for missing transformation:rainfall_dataset."""
    mock_path = "/fake/path/rain.nc"
    monkeypatch.setattr(
        Pmodel_initial, "load_dataset", lambda *args, **kwargs: xr.Dataset()
    )

    del rainfall_etl_settings["transformation"]["rainfall_dataset"]

    with pytest.raises(KeyError):
        Pmodel_initial.load_rainfall_dataset(mock_path, **rainfall_etl_settings)


# ---- Unit tests for load_population_dataset
def test_load_population_dataset_happy_path(
    population_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test successful loading and preprocessing of the population dataset."""
    file_path, _ = valid_netcdf_file
    preprocess_called = False

    def mock_preprocess(dataset, **kwargs):
        nonlocal preprocess_called
        preprocess_called = True
        assert (
            kwargs
            == population_etl_settings["transformation"]["human_population_dataset"][
                "preprocessing"
            ]
        )
        return dataset.rename(kwargs.get("names_dimensions", {}))

    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    result_ds = Pmodel_initial.load_population_dataset(
        file_path, **population_etl_settings
    )

    assert preprocess_called
    assert "latitude" in result_ds.dims
    assert "longitude" in result_ds.dims


def test_load_population_dataset_no_preprocessing(
    population_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test that preprocessing is skipped if config is missing."""
    file_path, original_ds = valid_netcdf_file
    preprocess_called = False

    def mock_preprocess(*args, **kwargs):
        nonlocal preprocess_called
        preprocess_called = True

    monkeypatch.setattr(Pmodel_initial, "preprocess_dataset", mock_preprocess)

    del population_etl_settings["transformation"]["human_population_dataset"][
        "preprocessing"
    ]

    result_ds = Pmodel_initial.load_population_dataset(
        file_path, **population_etl_settings
    )

    assert not preprocess_called
    xr.testing.assert_identical(result_ds, original_ds)


def test_load_population_dataset_load_error(population_etl_settings, monkeypatch):
    """Test that an error from load_dataset is propagated."""
    mock_path = "/fake/path/pop.nc"

    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("File not found")

    monkeypatch.setattr(Pmodel_initial, "load_dataset", raise_file_not_found)

    with pytest.raises(FileNotFoundError, match="File not found"):
        Pmodel_initial.load_population_dataset(mock_path, **population_etl_settings)


def test_load_population_dataset_preprocess_error(
    population_etl_settings, monkeypatch, valid_netcdf_file
):
    """Test that an error from preprocess_dataset is propagated."""
    file_path, _ = valid_netcdf_file

    def raise_preprocessing_failed(*args, **kwargs):
        raise ValueError("Preprocessing failed")

    monkeypatch.setattr(
        Pmodel_initial, "preprocess_dataset", raise_preprocessing_failed
    )

    with pytest.raises(ValueError, match="Preprocessing failed"):
        Pmodel_initial.load_population_dataset(file_path, **population_etl_settings)


def test_load_population_dataset_missing_load_config(
    population_etl_settings, monkeypatch
):
    """Test KeyError for missing ingestion:xarray_load_settings."""
    mock_path = "/fake/path/pop.nc"
    monkeypatch.setattr(Pmodel_initial, "load_dataset", lambda *args, **kwargs: None)
    del population_etl_settings["ingestion"]["xarray_load_settings"]

    with pytest.raises(KeyError):
        Pmodel_initial.load_population_dataset(mock_path, **population_etl_settings)


def test_load_population_dataset_missing_transform_config(
    population_etl_settings, monkeypatch
):
    """Test KeyError for missing transformation:human_population_dataset."""
    mock_path = "/fake/path/pop.nc"
    monkeypatch.setattr(
        Pmodel_initial, "load_dataset", lambda *args, **kwargs: xr.Dataset()
    )
    del population_etl_settings["transformation"]["human_population_dataset"]

    with pytest.raises(KeyError):
        Pmodel_initial.load_population_dataset(mock_path, **population_etl_settings)


# ---- Unit tests for create_temperature_daily
def test_create_temperature_daily_happy_path(
    sample_temperature_dataset, temperature_daily_etl_settings
):
    """Test that create_temperature_daily correctly expands temperature data."""
    time_step = temperature_daily_etl_settings["ode_system"]["time_step"]
    data_variable = temperature_daily_etl_settings["transformation"][
        "temperature_dataset"
    ]["data_variable"]
    original_data_array = sample_temperature_dataset[data_variable]

    temp_daily, temp_mean = Pmodel_initial.create_temperature_daily(
        temperature_dataset=sample_temperature_dataset, **temperature_daily_etl_settings
    )

    # Verify the shape of the output daily temperature array
    expected_shape = (
        original_data_array.shape[0] * time_step,
        original_data_array.shape[1],
        original_data_array.shape[2],
    )
    assert temp_daily.shape == expected_shape

    # Verify that the data values are correctly repeated
    np.testing.assert_array_equal(
        temp_daily.values[0:time_step],
        np.repeat(original_data_array.values[0:1], time_step, axis=0),
    )

    # Verify that the returned mean temperature is identical to the original
    xr.testing.assert_identical(temp_mean, original_data_array)

    # Verify that coordinates are preserved (excluding the 'time' coordinate)
    assert "longitude" in temp_daily.coords
    assert "latitude" in temp_daily.coords
    xr.testing.assert_equal(temp_daily["longitude"], original_data_array["longitude"])
    xr.testing.assert_equal(temp_daily["latitude"], original_data_array["latitude"])


def test_create_temperature_daily_time_step_one(
    sample_temperature_dataset, temperature_daily_etl_settings
):
    """Test create_temperature_daily with a time_step of 1."""

    temperature_daily_etl_settings["ode_system"]["time_step"] = 1
    data_variable = temperature_daily_etl_settings["transformation"][
        "temperature_dataset"
    ]["data_variable"]
    original_data_array = sample_temperature_dataset[data_variable]

    temp_daily, temp_mean = Pmodel_initial.create_temperature_daily(
        temperature_dataset=sample_temperature_dataset, **temperature_daily_etl_settings
    )

    # With time_step=1, the output data should be equal to the input
    assert temp_daily.shape == original_data_array.shape
    xr.testing.assert_identical(temp_mean, original_data_array)


def test_create_temperature_daily_time_step_zero(
    sample_temperature_dataset, temperature_daily_etl_settings
):
    """Test create_temperature_daily with a time_step of 0."""
    temperature_daily_etl_settings["ode_system"]["time_step"] = 0
    data_variable = temperature_daily_etl_settings["transformation"][
        "temperature_dataset"
    ]["data_variable"]
    original_data_array = sample_temperature_dataset[data_variable]

    temp_daily, _ = Pmodel_initial.create_temperature_daily(
        temperature_dataset=sample_temperature_dataset, **temperature_daily_etl_settings
    )

    # With time_step=0, the time dimension should be 0
    assert temp_daily.shape[0] == 0
    assert temp_daily.shape[1:] == original_data_array.shape[1:]


def test_create_temperature_daily_missing_time_step_config(
    sample_temperature_dataset, temperature_daily_etl_settings
):
    """Test for KeyError when 'time_step' is missing from settings."""
    del temperature_daily_etl_settings["ode_system"]["time_step"]
    with pytest.raises(KeyError):
        Pmodel_initial.create_temperature_daily(
            temperature_dataset=sample_temperature_dataset,
            **temperature_daily_etl_settings,
        )


def test_create_temperature_daily_missing_data_variable_config(
    sample_temperature_dataset, temperature_daily_etl_settings
):
    """Test for KeyError when 'data_variable' is missing from settings."""
    del temperature_daily_etl_settings["transformation"]["temperature_dataset"][
        "data_variable"
    ]
    with pytest.raises(KeyError):
        Pmodel_initial.create_temperature_daily(
            temperature_dataset=sample_temperature_dataset,
            **temperature_daily_etl_settings,
        )


def test_create_temperature_daily_dataset_missing_time_dimension(
    temperature_daily_etl_settings,
):
    """Test for an error when the input dataset is missing the 'time' dimension."""
    # Create a dataset without a 'time' dimension
    rng = np.random.default_rng(43)
    dataset_no_time = xr.Dataset(
        {"t2m": (("latitude", "longitude"), rng.random((3, 4)))},
        coords={"latitude": [10, 20, 30], "longitude": [-10, 0, 10, 20]},
    )
    with pytest.raises(ValueError):
        Pmodel_initial.create_temperature_daily(
            temperature_dataset=dataset_no_time, **temperature_daily_etl_settings
        )


def test_create_temperature_daily_multidimensional_data(temperature_daily_etl_settings):
    """Test that the function handles more than 3 dimensions correctly."""
    # Create a 4D dataset
    rng = np.random.default_rng(43)
    dataset_4d = xr.Dataset(
        {
            "t2m": (
                ("time", "level", "latitude", "longitude"),
                rng.random((2, 2, 3, 4)),
            )
        },
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "level": [1000, 850],
            "latitude": [10, 20, 30],
            "longitude": [-10, 0, 10, 20],
        },
    )
    time_step = temperature_daily_etl_settings["ode_system"]["time_step"]
    original_data_array = dataset_4d["t2m"]

    temp_daily, _ = Pmodel_initial.create_temperature_daily(
        temperature_dataset=dataset_4d, **temperature_daily_etl_settings
    )

    # Verify that only the time dimension was expanded
    expected_shape = (
        original_data_array.shape[0] * time_step,
        original_data_array.shape[1],
        original_data_array.shape[2],
        original_data_array.shape[3],
    )
    assert temp_daily.shape == expected_shape
    assert "level" in temp_daily.dims


# ---- Unit tests for load_initial_conditions
def test_load_initial_conditions_from_file(
    initial_conditions_netcdf_file, mock_etl_settings
):
    """
    Tests that load_initial_conditions correctly loads data from a valid NetCDF file.
    """
    file_path, expected_data = initial_conditions_netcdf_file
    sizes = (2, 2)  # (n_longitude, n_latitude)

    # Load initial conditions using the function
    result_v0 = Pmodel_initial.load_initial_conditions(
        filepath=file_path, sizes=sizes, **mock_etl_settings
    )

    # Verify the type and dims of the output
    n_vars = len(mock_etl_settings["ode_system"]["model_variables"])
    assert isinstance(result_v0, xr.DataArray)
    assert result_v0.shape == (sizes[0], sizes[1], n_vars)
    assert set(result_v0.dims) == {"longitude", "latitude", "variable"}
    # Check variable coordinate matches model_variables
    expected_variables = mock_etl_settings["ode_system"]["model_variables"]
    assert list(result_v0.coords["variable"].values) == expected_variables
    # Verify the content of the array
    for i, var in enumerate(expected_variables):
        np.testing.assert_allclose(
            result_v0.isel(variable=i).values, expected_data[var], rtol=1e-6
        )


def test_load_initial_conditions_default_initialization(mock_etl_settings):
    """
    Tests that default values are used when the filepath is None or does not exist.
    """
    sizes = (2, 3)
    n_vars = len(mock_etl_settings["ode_system"]["model_variables"])

    # Test with filepath=None
    result_v0_none = Pmodel_initial.load_initial_conditions(
        filepath=None, sizes=sizes, **mock_etl_settings
    )

    # Test with a non-existent file path
    result_v0_nonexistent = Pmodel_initial.load_initial_conditions(
        filepath="/non/existent/file.nc", sizes=sizes, **mock_etl_settings
    )

    # Expected default array
    expected_v0 = np.zeros((sizes[0], sizes[1], n_vars), dtype=np.float64)
    const_k1 = Pmodel_initial.CONSTANTS_INITIAL_CONDITIONS["CONST_K1"]
    const_k2 = Pmodel_initial.CONSTANTS_INITIAL_CONDITIONS["CONST_K2"]
    expected_v0[:, :, 1] = const_k1 * const_k2

    # Check type and dims
    for result_v0 in [result_v0_none, result_v0_nonexistent]:
        assert isinstance(result_v0, xr.DataArray)
        assert result_v0.shape == (sizes[0], sizes[1], n_vars)
        assert set(result_v0.dims) == {"longitude", "latitude", "variable"}
        # Check values for variable index 1
        np.testing.assert_allclose(
            result_v0.isel(variable=1).values, expected_v0[:, :, 1]
        )
        # Check all zeros for other variables (except index 1)
        for i in range(n_vars):
            if i != 1:
                np.testing.assert_allclose(result_v0.isel(variable=i).values, 0.0)


def test_load_initial_conditions_corrupt_file(corrupted_netcdf_file, mock_etl_settings):
    """
    Tests that an error is raised when the input file is corrupt.
    """
    with pytest.raises(Exception):
        Pmodel_initial.load_initial_conditions(
            filepath=corrupted_netcdf_file, sizes=(2, 2), **mock_etl_settings
        )


def test_load_initial_conditions_missing_variable(
    missing_variable_netcdf_file, mock_etl_settings
):
    """
    Tests that a KeyError is raised if a required variable is missing from the NetCDF file.
    """
    with pytest.raises(KeyError):
        Pmodel_initial.load_initial_conditions(
            filepath=missing_variable_netcdf_file, sizes=(2, 2), **mock_etl_settings
        )


def test_load_initial_conditions_ignores_extra_variables(
    extra_variable_netcdf_file, mock_etl_settings
):
    """
    Tests that the function ignores extra variables in the input file.
    """
    file_path, expected_data = extra_variable_netcdf_file
    sizes = (2, 2)

    result_v0 = Pmodel_initial.load_initial_conditions(
        filepath=file_path, sizes=sizes, **mock_etl_settings
    )

    n_vars = len(mock_etl_settings["ode_system"]["model_variables"])
    assert isinstance(result_v0, xr.DataArray)
    assert result_v0.shape == (sizes[0], sizes[1], n_vars)
    assert set(result_v0.dims) == {"longitude", "latitude", "variable"}
    expected_variables = mock_etl_settings["ode_system"]["model_variables"]
    assert list(result_v0.coords["variable"].values) == expected_variables
    for i, var in enumerate(expected_variables):
        np.testing.assert_allclose(
            result_v0.isel(variable=i).values, expected_data[var], rtol=1e-6
        )


def test_load_initial_conditions_nofile_regression(dummy_file_etl_settings_yaml):

    etl_settings = dummy_file_etl_settings_yaml

    number_longitudes = 3
    number_latitudes = 2
    sizes = (number_longitudes, number_latitudes)

    result = Pmodel_initial.load_initial_conditions(
        filepath=None, sizes=sizes, **etl_settings
    )

    expected_values = np.zeros((3, 2, 5), dtype=np.float64)
    expected_values[:, :, 1] = 62500.0
    expected_values = xr.DataArray(
        expected_values, dims=result.dims, coords=result.coords
    )

    # Compare dimensions
    expected_dim = (3, 2, 5)
    assert result.shape == expected_dim

    # Assert values
    xr.testing.assert_allclose(result, expected_values, rtol=1e-4, atol=1e-4)


def test_load_initial_conditions_regression(
    dummy_file_etl_settings_yaml, dummy_file_initial_conditions_path
):
    etl_settings = dummy_file_etl_settings_yaml

    number_longitudes = 3
    number_latitudes = 2

    sizes = (number_longitudes, number_latitudes)

    result = Pmodel_initial.load_initial_conditions(
        filepath=dummy_file_initial_conditions_path, sizes=sizes, **etl_settings
    )

    expected_values = np.array(
        [
            # longitude 0
            [
                [5.9, 11.8, 17.7, 23.6, 35.4],
                [6.05, 12.1, 18.15, 24.2, 36.3],
            ],
            # longitude 1
            [
                [5.95, 11.9, 17.85, 23.8, 35.7],
                [6.1, 12.2, 18.3, 24.4, 36.6],
            ],
            # longitude 2
            [
                [6.0, 12.0, 18.0, 24.0, 36.0],
                [6.15, 12.3, 18.45, 24.6, 36.9],
            ],
        ]
    )
    expected_values = xr.DataArray(
        expected_values, dims=result.dims, coords=result.coords
    )

    # Compare dimensions
    expected_dim = (3, 2, 5)
    assert result.shape == expected_dim

    # Assert values
    xr.testing.assert_allclose(result, expected_values, rtol=1e-4, atol=1e-4)


# ---- Unit tests for load_all_data
def test_load_all_data_happy_path(
    monkeypatch, full_etl_settings, mock_data_paths, mock_model_inputs
):
    """
    Tests the successful execution of load_all_data (happy path) by mocking all
    underlying functions.
    """
    # --- Mock all external function calls ---
    mock_load_temp = lambda *args, **kwargs: mock_model_inputs["temperature"]
    mock_load_rain = lambda *args, **kwargs: mock_model_inputs["rainfall"]
    mock_load_pop = lambda *args, **kwargs: mock_model_inputs["population"]

    mock_postprocess = lambda dataset, **kwargs: mock_model_inputs[
        "processed_population"
    ]

    mock_create_temp = lambda *args, **kwargs: (
        mock_model_inputs["da_temp_daily"],
        mock_model_inputs["da_temp_mean"],
    )

    mock_load_initial = lambda *args, **kwargs: mock_model_inputs["initial_conditions"]

    monkeypatch.setattr(Pmodel_initial, "load_temperature_dataset", mock_load_temp)
    monkeypatch.setattr(Pmodel_initial, "load_rainfall_dataset", mock_load_rain)
    monkeypatch.setattr(Pmodel_initial, "load_population_dataset", mock_load_pop)
    monkeypatch.setattr(Pmodel_initial, "postprocess_dataset", mock_postprocess)
    monkeypatch.setattr(Pmodel_initial, "create_temperature_daily", mock_create_temp)
    monkeypatch.setattr(Pmodel_initial, "load_initial_conditions", mock_load_initial)

    # --- Call the function under test ---
    result = Pmodel_initial.load_all_data(
        paths=mock_data_paths, etl_settings=full_etl_settings
    )

    # --- Assertions ---
    # Verify the type of the output
    assert isinstance(result, Pmodel_initial.PmodelInput)

    # Verify that the data in the PmodelInput object matches the mock data
    np.testing.assert_array_equal(
        result.initial_conditions, mock_model_inputs["initial_conditions"]
    )
    xr.testing.assert_identical(result.temperature, mock_model_inputs["da_temp_daily"])
    xr.testing.assert_identical(result.rainfall, mock_model_inputs["rainfall"]["tp"])
    xr.testing.assert_identical(
        result.population_density,
        mock_model_inputs["processed_population"]["population_density"],
    )
    xr.testing.assert_identical(
        result.temperature_mean, mock_model_inputs["da_temp_mean"]
    )
    xr.testing.assert_identical(
        result.latitude, mock_model_inputs["da_temp_mean"]["latitude"]
    )


def test_load_all_data_missing_path(full_etl_settings, mock_data_paths):
    """
    Tests that a KeyError is raised if the paths dictionary is missing a required key.
    """
    # Remove a required path from the dictionary
    del mock_data_paths["temperature_dataset"]

    with pytest.raises(KeyError, match="temperature_dataset"):
        Pmodel_initial.load_all_data(
            paths=mock_data_paths, etl_settings=full_etl_settings
        )


def test_load_all_data_loading_error(monkeypatch, full_etl_settings, mock_data_paths):
    """
    Tests that an error from a dataset loading function is propagated.
    """

    # Mock one of the loading functions to raise an error
    def mock_load_raises_error(*args, **kwargs):
        raise FileNotFoundError("Dataset file not found")

    monkeypatch.setattr(
        Pmodel_initial, "load_temperature_dataset", mock_load_raises_error
    )

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        Pmodel_initial.load_all_data(
            paths=mock_data_paths, etl_settings=full_etl_settings
        )


def test_load_all_data_processing_error(
    monkeypatch, full_etl_settings, mock_data_paths, mock_model_inputs
):
    """
    Tests that an error from a data processing function is propagated.
    """
    # Mock loading functions to return valid data
    monkeypatch.setattr(
        Pmodel_initial,
        "load_temperature_dataset",
        lambda *args, **kwargs: mock_model_inputs["temperature"],
    )
    monkeypatch.setattr(
        Pmodel_initial,
        "load_rainfall_dataset",
        lambda *args, **kwargs: mock_model_inputs["rainfall"],
    )
    monkeypatch.setattr(
        Pmodel_initial,
        "load_population_dataset",
        lambda *args, **kwargs: mock_model_inputs["population"],
    )
    monkeypatch.setattr(
        Pmodel_initial,
        "postprocess_dataset",
        lambda dataset, **kwargs: mock_model_inputs["processed_population"],
    )

    # Mock a processing function to raise an error
    def mock_create_temp_raises_error(*args, **kwargs):
        raise ValueError("Processing failed")

    monkeypatch.setattr(
        Pmodel_initial, "create_temperature_daily", mock_create_temp_raises_error
    )

    with pytest.raises(ValueError, match="Processing failed"):
        Pmodel_initial.load_all_data(
            paths=mock_data_paths, etl_settings=full_etl_settings
        )


def test_load_all_data_missing_etl_config(
    monkeypatch, full_etl_settings, mock_data_paths, mock_model_inputs
):
    """
    Tests that a KeyError is raised if a critical ETL setting is missing.
    """
    # Mock loading functions to avoid errors before the target check
    monkeypatch.setattr(
        Pmodel_initial,
        "load_temperature_dataset",
        lambda *args, **kwargs: mock_model_inputs["temperature"],
    )
    monkeypatch.setattr(
        Pmodel_initial,
        "load_rainfall_dataset",
        lambda *args, **kwargs: mock_model_inputs["rainfall"],
    )
    monkeypatch.setattr(
        Pmodel_initial,
        "load_population_dataset",
        lambda *args, **kwargs: mock_model_inputs["population"],
    )

    # Remove a critical key from the settings
    del full_etl_settings["transformation"]["rainfall_dataset"]["data_variable"]

    with pytest.raises(KeyError, match="data_variable"):
        Pmodel_initial.load_all_data(
            paths=mock_data_paths, etl_settings=full_etl_settings
        )


# ---- Unit tests for load_all_data create_model
def test_create_model_output_regression(dummy_input_data, dummy_file_etl_settings_yaml):

    dataset_base = dummy_input_data.temperature
    initial_conditions_shape = dummy_input_data.initial_conditions.shape

    time_step = dummy_file_etl_settings_yaml["ode_system"]["time_step"]

    result = Pmodel_initial.create_model_output(
        dataset_base_shape=dataset_base.shape,
        initial_conditions_shape=initial_conditions_shape,
        time_step=time_step,
    )

    assert result.shape == (3, 2, 5, 4)
    assert result.dtype == np.float64
    assert np.all(result == 0)
