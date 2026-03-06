from pathlib import Path

import numpy as np
import pytest
import xarray as xr


from heiplanet_models.utils import (
    load_module,
    load_name_from_module,
    validate_spatial_alignment,
)


# ---- Fixtures
def make_dataarray(lat, lon, name="var"):
    data = np.zeros((len(lat), len(lon)))
    return xr.DataArray(
        data,
        coords={"latitude": lat, "longitude": lon},
        dims=["latitude", "longitude"],
        name=name,
    )


# ---- load_module()
def test_load_module():
    module_name = "cm"
    file_path = Path("./test/computation_module.py").resolve().absolute()
    module = load_module(module_name, file_path)
    assert module is not None, "Module should be loaded successfully"
    assert hasattr(module, "add"), "Module should have an 'add' function"

    with pytest.raises(RuntimeError, match="Error in loading module"):
        load_module("non_existent_module", "./non_existent_path.py")


def test_load_name_from_module():
    file_path = Path("./test/computation_module.py").resolve().absolute()
    function_name = "add"
    module_name = "cm"
    function = load_name_from_module(module_name, file_path, function_name)
    assert callable(function), "Loaded function should be callable"
    assert function(2, 3) == 5, "Function should return correct result"


# ---- validate_spatial_alignment
def test_validate_spatial_alignment_success():
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    arr1 = make_dataarray(lat, lon)
    arr2 = make_dataarray(lat, lon)
    # Should not raise
    validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_latitude_mismatch():
    lat1 = np.linspace(-90, 90, 5)
    lat2 = np.linspace(-80, 80, 5)
    lon = np.linspace(-180, 180, 5)
    arr1 = make_dataarray(lat1, lon)
    arr2 = make_dataarray(lat2, lon)
    with pytest.raises(ValueError, match="must be aligned"):
        validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_longitude_mismatch():
    lat = np.linspace(-90, 90, 5)
    lon1 = np.linspace(-180, 180, 5)
    lon2 = np.linspace(-170, 170, 5)
    arr1 = make_dataarray(lat, lon1)
    arr2 = make_dataarray(lat, lon2)
    with pytest.raises(ValueError, match="must be aligned"):
        validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_missing_latitude():
    lon = np.linspace(-180, 180, 5)
    arr1 = xr.DataArray(np.zeros((5,)), coords={"longitude": lon}, dims=["longitude"])
    arr2 = make_dataarray(np.linspace(-90, 90, 5), lon)
    with pytest.raises(
        ValueError, match="Input DataArrays must have a 'latitude' coordinate."
    ):
        validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_missing_longitude():
    lat = np.linspace(-90, 90, 5)
    arr1 = xr.DataArray(np.zeros((5,)), coords={"latitude": lat}, dims=["latitude"])
    arr2 = make_dataarray(lat, np.linspace(-180, 180, 5))
    with pytest.raises(
        ValueError, match="Input DataArrays must have a 'longitude' coordinate."
    ):
        validate_spatial_alignment(arr1, arr2)
