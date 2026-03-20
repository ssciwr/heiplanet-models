"""Unit tests for heiplanet_models.Pmodel.Pmodel_rates_mortality.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_rates_mortality import (
    mosq_mort_e,
    mosq_mort_j,
    mosq_mort_a,
    mosq_surv_ed,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def typical_temperature_array():
    arr = np.array([10, 20, 25, 30], dtype=float)
    return xr.DataArray(arr, dims=["time"])


@pytest.fixture
def edge_case_temperature_array():
    arr = np.array([0, -273.15, 1e6], dtype=float)
    return xr.DataArray(arr, dims=["time"])


@pytest.fixture
def negative_temperature_array():
    arr = np.array([-10, -20, -30], dtype=float)
    return xr.DataArray(arr, dims=["time"])


@pytest.fixture
def large_temperature_array():
    arr = np.linspace(-100, 100, 1000)
    return xr.DataArray(arr, dims=["time"])


@pytest.fixture
def empty_temperature_array():
    arr = np.array([], dtype=float)
    return xr.DataArray(arr, dims=["time"])


@pytest.fixture
def multidimensional_temperature_array():
    arr = np.array([[10, 20], [30, 40]], dtype=float)
    return xr.DataArray(arr, dims=["x", "y"])


@pytest.fixture
def typical_3d_temperature_array(rng):
    arr = rng.uniform(10, 30, size=(2, 2, 4))
    return xr.DataArray(arr, dims=["x", "y", "t"])


@pytest.fixture
def edge_case_3d_temperature_array():
    arr = np.array([[[0, -273.15], [1e6, 0]], [[-1e6, 1e6], [0, 0]]], dtype=float)
    return xr.DataArray(arr, dims=["x", "y", "t"])


@pytest.fixture
def negative_3d_temperature_array(rng):
    arr = -np.abs(rng.uniform(0, 100, size=(2, 2, 4)))
    return xr.DataArray(arr, dims=["x", "y", "t"])


@pytest.fixture
def large_3d_temperature_array(rng):
    arr = rng.uniform(-100, 100, size=(4, 4, 10))
    return xr.DataArray(arr, dims=["x", "y", "t"])


@pytest.fixture
def empty_3d_temperature_array():
    arr = np.empty((0, 0, 0), dtype=float)
    return xr.DataArray(arr, dims=["x", "y", "t"])


@pytest.fixture
def temperature_array_4x3x2():
    arr = np.arange(24, dtype=float).reshape((4, 3, 2))
    return xr.DataArray(arr, dims=["x", "y", "t"])


@pytest.fixture
def chunked_temperature_array(rng):
    arr = rng.random((2, 2, 2))
    da = xr.DataArray(arr, dims=["x", "y", "t"])
    # Use xarray's chunk method to create a chunked DataArray (requires dask)
    # If dask is not used, simulate chunking by adding a 'chunks' attribute
    da = da.chunk({"t": 1})
    return da


# ---- Unit Tests for mosq_mort_e
def test_mosq_mort_e_typical_temperatures(typical_temperature_array):
    result = mosq_mort_e(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result.data))
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_e_scalar_input():
    temp = xr.DataArray(25.0)
    result = mosq_mort_e(temp)
    assert result.shape == ()


def test_mosq_mort_e_edge_case_temperatures(edge_case_temperature_array):
    result = mosq_mort_e(edge_case_temperature_array)
    assert result.shape == edge_case_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_e_negative_temperatures(negative_temperature_array):
    result = mosq_mort_e(negative_temperature_array)
    assert result.shape == negative_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_e_large_temperatures(large_temperature_array):
    result = mosq_mort_e(large_temperature_array)
    assert result.shape == large_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_e_empty_array(empty_temperature_array):
    result = mosq_mort_e(empty_temperature_array)
    assert result.shape == empty_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_e_non_numeric_input():
    with pytest.raises(TypeError):
        mosq_mort_e(xr.DataArray(["a", "b"], dims=["time"]))


def test_mosq_mort_e_multidimensional_input(multidimensional_temperature_array):
    result = mosq_mort_e(multidimensional_temperature_array)
    assert result.shape == multidimensional_temperature_array.shape
    assert isinstance(result, xr.DataArray)


# ---- Unit Tests for mosq_mort_j


def test_mosq_mort_j_typical_temperatures(typical_temperature_array):
    result = mosq_mort_j(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result.data))
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_j_scalar_input():
    temp = xr.DataArray(25.0)
    result = mosq_mort_j(temp)
    assert result.shape == ()


def test_mosq_mort_j_edge_case_temperatures(edge_case_temperature_array):
    result = mosq_mort_j(edge_case_temperature_array)
    assert result.shape == edge_case_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_j_negative_temperatures(negative_temperature_array):
    result = mosq_mort_j(negative_temperature_array)
    assert result.shape == negative_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_j_large_temperatures(large_temperature_array):
    result = mosq_mort_j(large_temperature_array)
    assert result.shape == large_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_j_empty_array(empty_temperature_array):
    result = mosq_mort_j(empty_temperature_array)
    assert result.shape == empty_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_j_non_numeric_input():
    with pytest.raises(TypeError):
        mosq_mort_j(xr.DataArray(["a", "b"], dims=["time"]))


def test_mosq_mort_j_multidimensional_input(multidimensional_temperature_array):
    result = mosq_mort_j(multidimensional_temperature_array)
    assert result.shape == multidimensional_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_j_numerical_stability_near_zero():
    arr = np.array([-1e6, 1e6], dtype=float)
    temp = xr.DataArray(arr, dims=["time"])
    result = mosq_mort_j(temp)
    assert np.all(np.isfinite(result.data) | np.isinf(result.data))


# ---- Unit Tests for mosq_mort_a


def test_mosq_mort_a_typical_temperatures(typical_temperature_array):
    result = mosq_mort_a(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result.data))
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_a_scalar_input():
    temp = xr.DataArray(25.0)
    result = mosq_mort_a(temp)
    assert result.shape == ()


def test_mosq_mort_a_multidimensional_input(multidimensional_temperature_array):
    result = mosq_mort_a(multidimensional_temperature_array)
    assert result.shape == multidimensional_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_a_edge_case_temperatures(edge_case_temperature_array):
    result = mosq_mort_a(edge_case_temperature_array)
    assert result.shape == edge_case_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_a_negative_temperatures(negative_temperature_array):
    result = mosq_mort_a(negative_temperature_array)
    assert result.shape == negative_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_a_large_temperatures(large_temperature_array):
    result = mosq_mort_a(large_temperature_array)
    assert result.shape == large_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_a_empty_array(empty_temperature_array):
    result = mosq_mort_a(empty_temperature_array)
    assert result.shape == empty_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_mort_a_non_numeric_input():
    with pytest.raises(TypeError):
        mosq_mort_a(xr.DataArray(["a", "b"], dims=["time"]))


def test_mosq_mort_a_numerical_stability_near_zero():
    arr = np.array([-1e6, 1e6], dtype=float)
    temp = xr.DataArray(arr, dims=["time"])
    result = mosq_mort_a(temp)
    assert np.all(np.isfinite(result.data) | np.isinf(result.data))


# ---- Unit Tests for mosq_surv_ed


def test_mosq_surv_ed_typical_temperatures(typical_3d_temperature_array):
    result = mosq_surv_ed(typical_3d_temperature_array)
    assert result.shape == typical_3d_temperature_array.shape
    assert np.all(np.isfinite(result.data))
    assert isinstance(result, xr.DataArray)
    assert np.all((result.data >= 0) & (result.data <= 1))


def test_mosq_surv_ed_scalar_input():
    with pytest.raises(ValueError):
        mosq_surv_ed(xr.DataArray(25.0))
    with pytest.raises(ValueError):
        mosq_surv_ed(xr.DataArray(25))
    with pytest.raises(ValueError):
        mosq_surv_ed(xr.DataArray([25.0]))
    with pytest.raises(ValueError):
        mosq_surv_ed(xr.DataArray((25.0,)))


def test_mosq_surv_ed_edge_case_temperatures(edge_case_3d_temperature_array):
    result = mosq_surv_ed(edge_case_3d_temperature_array)
    assert result.shape == edge_case_3d_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_surv_ed_negative_temperatures(negative_3d_temperature_array):
    result = mosq_surv_ed(negative_3d_temperature_array)
    assert result.shape == negative_3d_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_surv_ed_large_temperatures(large_3d_temperature_array):
    result = mosq_surv_ed(large_3d_temperature_array)
    assert result.shape == large_3d_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_surv_ed_empty_array(empty_3d_temperature_array):
    result = mosq_surv_ed(empty_3d_temperature_array)
    assert result.shape == empty_3d_temperature_array.shape
    assert isinstance(result, xr.DataArray)


def test_mosq_surv_ed_non_numeric_input():
    arr = np.array([[["a"]]], dtype=object)
    with pytest.raises(TypeError):
        mosq_surv_ed(xr.DataArray(arr, dims=["x", "y", "t"]))


def test_mosq_surv_ed_multidimensional_input_2d(multidimensional_temperature_array):
    with pytest.raises(ValueError):
        mosq_surv_ed(multidimensional_temperature_array)


def test_mosq_surv_ed_raises_on_non_xarray():
    arr = np.array([1, 2, 3])
    with pytest.raises(
        ValueError, match="Input 'temperature' must be an xarray.DataArray."
    ):
        mosq_surv_ed(arr)


def test_mosq_surv_ed_chunking(chunked_temperature_array):
    # This will exercise the chunking code path in mosq_surv_ed
    result = mosq_surv_ed(chunked_temperature_array)
    assert isinstance(result, xr.DataArray)
    assert result.shape == chunked_temperature_array.shape


def test_mosq_surv_ed_regression(model_input_dummy_datasets, test_etl_settings):

    temperature = model_input_dummy_datasets.temperature
    time_step = test_etl_settings["ode_system"]["time_step"]

    result = mosq_surv_ed(temperature=temperature, step_t=time_step)

    base_slice = np.array([[0.8845, 0.9236], [0.9045, 0.9272], [0.9167, 0.9289]])

    # Tile the array to create 40 copies along the third dimension
    expected_values = np.tile(base_slice[:, :, np.newaxis], (1, 1, 40))

    # Create xarray DataArray with same structure as result
    expected = xr.DataArray(expected_values, dims=result.dims, coords=result.coords)

    # Compare result against expected values
    xr.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
