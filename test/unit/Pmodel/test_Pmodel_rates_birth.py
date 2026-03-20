"""Unit tests for heiplanet_models.Pmodel.Pmodel_rates_birth.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pytest

import numpy as np
import xarray as xr


from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
    water_hatching,
)
from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MOSQUITO_BIRTH,
    CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING,
    CONSTANTS_MOSQUITO_DIAPAUSE_LAY,
    CONSTANTS_WATER_HATCHING,
    DAYS_YEAR,
    HALF_DAYS_YEAR,
)


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def mock_lay_data():
    """Provides mock xarray data for diapause lay tests for a single year."""
    lons = [10.0]
    lats = [45.0]  # Mid-latitude
    times = np.arange(DAYS_YEAR)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_lay_data_multi_year():
    """Provides mock xarray data for diapause lay tests for two years."""
    lons = [10.0]
    lats = [45.0]
    times = np.arange(DAYS_YEAR * 2)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_lay_data_equator():
    """Provides mock xarray data for diapause lay tests at the equator."""
    lons = [10.0]
    lats = [0.0]  # Equator
    times = np.arange(DAYS_YEAR)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_lay_data_polar():
    """Provides mock xarray data for diapause lay tests at a polar latitude."""
    lons = [10.0]
    lats = [80.0]  # Polar latitude
    times = np.arange(DAYS_YEAR)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_hatch_data():
    """Provides mock xarray data for diapause hatching tests."""
    lons = [10.0, 11.0]
    lats = [45.0, 46.0]
    times = np.arange(60, dtype=int)  # 60 days
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_hatch_data_below_ctt():
    """Provides mock xarray data for diapause hatching tests with temperatures below CTT."""
    lons = [10.0, 11.0]
    lats = [45.0, 46.0]
    times = np.arange(60, dtype=int)  # 60 days
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    CTT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CTT"]
    # All temperatures set below CTT
    temp_data = xr.DataArray(
        np.full((len(lons), len(lats), len(times)), CTT - 5), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_hatch_data_high_latitude():
    """Provides mock xarray data for diapause hatching tests at high latitude."""
    lons = [10.0, 11.0]
    lats = [80.0]  # High latitude for short daylight
    times = np.arange(60, dtype=int)  # 60 days
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    CTT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CTT"]
    # Temperatures always above CTT
    temp_data = xr.DataArray(
        np.full((len(lons), len(lats), len(times)), CTT + 10), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def rainfall_data():
    # Create a simple rainfall DataArray: shape (time, lat, lon)
    data = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ]
    )
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [0, 1, 2],
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def population_data_2d(rainfall_data):
    """Provides a 2D population DataArray matching the spatial dims of rainfall_data."""
    data = np.array([[10, 20], [30, 40]])
    return xr.DataArray(
        data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": rainfall_data.latitude,
            "longitude": rainfall_data.longitude,
        },
    )


@pytest.fixture
def population_data_time_variant():
    # Population density with time dimension
    data = np.array(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
            [[90, 100], [110, 120]],
        ]
    )
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [0, 1, 2],
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def dummy_constants_water_hatching():
    return {
        "E_OPT": 8.0,
        "E_VAR": 0.05,
        "E_0": 1.5,
        "E_RAT": 0.2,
        "E_DENS": 0.01,
        "E_FAC": 0.01,
    }


def make_da(arr, dims=None, coords=None):
    # Helper to create a DataArray with dims and coords if not provided
    if dims is None:
        dims = ("dim_0",)
    if coords is None:
        coords = {dims[0]: np.arange(arr.shape[0])}
    return xr.DataArray(arr, dims=dims, coords=coords)


# ---- mosq_birth()
def test_mosq_birth_temp_above_threshold():
    """Test birth rate is zero when temperature is above the threshold."""
    const_1 = CONSTANTS_MOSQUITO_BIRTH["CONST_1"]
    arr = np.array([const_1 + 1, const_1 + 10, 40])
    temperature = make_da(arr)
    expected = np.array([0.0, 0.0, 0.0])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, atol=1e-9)


def test_mosq_birth_temp_at_threshold():
    """Test birth rate is zero when temperature is exactly at the threshold."""
    const_1 = CONSTANTS_MOSQUITO_BIRTH["CONST_1"]
    arr = np.array([const_1])
    temperature = make_da(arr)
    expected = np.array([0.0])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, atol=1e-9)


def test_mosq_birth_temp_below_threshold():
    """Test birth rate is calculated correctly for a temperature below the threshold."""
    C = CONSTANTS_MOSQUITO_BIRTH
    temp = 20.0
    temperature = make_da(np.array([temp]))
    expected_value = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp) ** C["CONST_6"]
    )
    expected = np.array([expected_value])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, rtol=1e-7)


def test_mosq_birth_mixed_temperatures():
    """Test birth rate with a mix of temperatures above, at, and below the threshold."""
    C = CONSTANTS_MOSQUITO_BIRTH
    const_1 = C["CONST_1"]
    temp_below = 15.0
    arr = np.array([temp_below, const_1, const_1 + 5])
    temperature = make_da(arr)
    expected_below = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp_below - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp_below) ** C["CONST_6"]
    )
    expected = np.array([expected_below, 0.0, 0.0])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, rtol=1e-7)


def test_mosq_birth_zero_temperature():
    """Test birth rate calculation when the temperature is zero."""
    C = CONSTANTS_MOSQUITO_BIRTH
    temp = 0.0
    temperature = make_da(np.array([temp]))
    expected_value = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp) ** C["CONST_6"]
    )
    expected = np.array([expected_value])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, rtol=1e-7)


def test_mosq_birth_multidimensional_input():
    """Test that the function handles multi-dimensional arrays correctly."""
    C = CONSTANTS_MOSQUITO_BIRTH
    const_1 = C["CONST_1"]
    temp_below_a = 25.0
    temp_below_b = -5.0
    arr = np.array([[temp_below_a, const_1 + 10], [const_1, temp_below_b]])
    temperature = xr.DataArray(arr, dims=("x", "y"), coords={"x": [0, 1], "y": [0, 1]})

    expected_below_1 = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp_below_a - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp_below_a) ** C["CONST_6"]
    )
    expected_below_2 = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * (((temp_below_b) - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - (-5.0)) ** C["CONST_6"]
    )
    expected = np.array([[expected_below_1, 0.0], [0.0, expected_below_2]])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    assert result.shape == arr.shape
    np.testing.assert_allclose(result.values, expected, rtol=1e-7)


def test_mosq_birth_empty_input():
    """Test that the function handles an empty array input gracefully."""
    temperature = make_da(np.array([]))
    expected = np.array([])
    result = mosq_birth(temperature)
    assert isinstance(result, xr.DataArray)
    assert result.shape == expected.shape


# ---- mosq_dia_hatch()
def test_mosq_dia_hatch_invalid_temp_dims_raises(mock_hatch_data):
    """Test that a non-3D temperature array raises a ValueError."""
    _, lat_data = mock_hatch_data
    invalid_temp = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Temperature array must be 3D"):
        mosq_dia_hatch(invalid_temp, lat_data)


def test_mosq_dia_hatch_invalid_lat_dims_raises(mock_hatch_data):
    """Test that a non-1D latitude array raises a ValueError."""
    temp_data, _ = mock_hatch_data
    invalid_lat = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Latitude array must be 1D."):
        mosq_dia_hatch(temp_data, invalid_lat)


def test_mosq_dia_hatch_temp_below_ctt(mock_hatch_data_below_ctt):
    """Test that hatching is zero if temperature is always below the critical threshold."""
    temp_data, lat_data = mock_hatch_data_below_ctt
    result = mosq_dia_hatch(temp_data, lat_data)
    assert np.all(result.values == 0)


def test_mosq_dia_hatch_daylight_below_cpp(mock_hatch_data_high_latitude):
    """Test that hatching is zero if daylight is always below the critical photoperiod."""
    temp_data, lat_data = mock_hatch_data_high_latitude
    result = mosq_dia_hatch(temp_data, lat_data)
    # Expect all zeros because daylight hours will be below CPP for the first 60 days
    assert np.all(result.values == 0)


def test_mosq_dia_hatch_with_nan_input(mock_hatch_data):
    """Test that NaN values in the input are handled correctly."""
    temp_data, lat_data = mock_hatch_data
    C = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING
    temp_data.values[:] = C["CTT"] + 10
    temp_data.values[0, 0, 30] = np.nan  # Introduce a NaN

    result = mosq_dia_hatch(temp_data, lat_data)
    # Ensure no NaNs are in the output
    assert not np.isnan(result.values).any()
    # The cell with the original NaN should result in 0 after nan_to_num
    assert result.values[0, 0, 30] == 0


def test_mosq_dia_hatch_regression(model_input_dummy_datasets):

    temperature_mean = model_input_dummy_datasets.temperature_mean
    latitude = model_input_dummy_datasets.latitude

    result = mosq_dia_hatch(temperature=temperature_mean, latitude=latitude)

    expected_values = np.array(
        [
            # latitude 0
            [[0, 0, 0.1, 0.1], [0, 0, 0.1, 0.1]],
            # latitude 1
            [[0, 0, 0.1, 0.1], [0, 0.1, 0.1, 0.1]],
            # latitude 2
            [[0, 0, 0.1, 0.1], [0, 0.1, 0.1, 0.1]],
        ]
    )

    # Create xarray DataArray with same structure as result
    expected = xr.DataArray(expected_values, dims=result.dims, coords=result.coords)

    # Compare result against expected values
    xr.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


# ---- mosq_dia_lay()
def test_mosq_dia_lay_invalid_temp_dims_raises(mock_lay_data):
    """Test that a non-3D temperature array raises a ValueError."""
    _, lat_data = mock_lay_data
    invalid_temp = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Temperature array must be 3D"):
        mosq_dia_lay(invalid_temp, lat_data)


def test_mosq_dia_lay_invalid_lat_dims_raises(mock_lay_data):
    """Test that a non-1D latitude array raises a ValueError."""
    temp_data, _ = mock_lay_data
    invalid_lat = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Latitude array must be 1D."):
        mosq_dia_lay(temp_data, invalid_lat)


def test_mosq_dia_lay_mid_latitude_seasonal_split(mock_lay_data):
    """Test that diapause is zero in the first half of the year."""
    temp_data, lat_data = mock_lay_data
    result = mosq_dia_lay(temp_data, lat_data)

    # Assert first half of the year is all zero
    assert np.all(result.values[:, :, :HALF_DAYS_YEAR] == 0)
    # Assert there is some diapause in the second half
    assert np.any(result.values[:, :, HALF_DAYS_YEAR:] > 0)


def test_mosq_dia_lay_equator_no_diapause(mock_lay_data_equator):
    """Test that no diapause occurs at the equator."""
    temp_data, lat_data = mock_lay_data_equator
    result = mosq_dia_lay(temp_data, lat_data)
    # At the equator, daylight is always long enough to prevent diapause
    assert np.all(result.values == 0)


def test_mosq_dia_lay_polar_handling(mock_lay_data_polar):
    """Test that the function handles polar regions without producing NaNs."""
    temp_data, lat_data = mock_lay_data_polar
    result = mosq_dia_lay(temp_data, lat_data)
    # Ensure no NaNs are in the output, which can happen with extreme daylight calculations
    assert not np.isnan(result.values).any()


def test_mosq_dia_lay_output_values(mock_lay_data):
    """Test that all non-zero output values are equal to RATIO_DIA_LAY."""
    temp_data, lat_data = mock_lay_data
    result = mosq_dia_lay(temp_data, lat_data)
    ratio = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["RATIO_DIA_LAY"]

    # Get all non-zero values from the result
    non_zero_values = result.values[result.values > 0]

    # Assert that all these non-zero values are equal to the expected ratio
    assert np.all(non_zero_values == ratio)


def test_mosq_dia_lay_regression(model_input_dummy_datasets):

    temperature_mean = model_input_dummy_datasets.temperature_mean
    latitude = model_input_dummy_datasets.latitude

    result = mosq_dia_lay(temperature=temperature_mean, latitude=latitude)

    expected_values = np.zeros(shape=(3, 2, 4))

    # Create xarray DataArray with same structure as result
    expected = xr.DataArray(expected_values, dims=result.dims, coords=result.coords)

    # Compare result against expected values
    xr.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


# ---- water_hatching()
def test_water_hatching_with_2d_population_data(rainfall_data, population_data_2d):
    """Test that a 2D population array is correctly broadcast to 3D."""
    result = water_hatching(rainfall_data, population_data_2d)
    assert result.shape == rainfall_data.shape
    assert "time" in result.dims


def test_water_hatching_population_effect_is_constant(
    rainfall_data, population_data_time_variant
):
    """Test that only the first time slice of population data is used."""
    # Manually calculate the population component using only the first time slice
    C = CONSTANTS_WATER_HATCHING
    pop_t0 = population_data_time_variant.isel(time=0)
    expected_pop_hatch = C["E_DENS"] / (C["E_DENS"] + np.exp(-C["E_FAC"] * pop_t0))

    # Calculate the rainfall component
    exp_term = np.exp(-C["E_VAR"] * (rainfall_data - C["E_OPT"]) ** 2)
    rainfall_hatch = (1 + C["E_0"]) * exp_term / (exp_term + C["E_0"])

    # Calculate the final expected result
    expected_result = ((1 - C["E_RAT"]) * rainfall_hatch) + (
        C["E_RAT"] * expected_pop_hatch
    )

    # Get the actual result from the function
    actual_result = water_hatching(rainfall_data, population_data_time_variant)

    xr.testing.assert_allclose(actual_result, expected_result)


def test_water_hatching_weighting(rainfall_data, population_data_2d, monkeypatch):
    """Test the E_RAT weighting between rainfall and population effects."""
    C = CONSTANTS_WATER_HATCHING.copy()

    # --- Test with E_RAT = 0 (only rainfall matters) ---
    C["E_RAT"] = 0.0
    monkeypatch.setitem(CONSTANTS_WATER_HATCHING, "E_RAT", 0.0)

    exp_term = np.exp(-C["E_VAR"] * (rainfall_data - C["E_OPT"]) ** 2)
    expected_rainfall_only = (1 + C["E_0"]) * exp_term / (exp_term + C["E_0"])
    result_rainfall_only = water_hatching(rainfall_data, population_data_2d)
    xr.testing.assert_allclose(result_rainfall_only, expected_rainfall_only)

    # --- Test with E_RAT = 1 (only population matters) ---
    monkeypatch.setitem(CONSTANTS_WATER_HATCHING, "E_RAT", 1.0)

    expected_pop_only_2d = C["E_DENS"] / (
        C["E_DENS"] + np.exp(-C["E_FAC"] * population_data_2d)
    )
    # Broadcast the 2D expected result to 3D to match the function's output shape
    expected_pop_only = expected_pop_only_2d.expand_dims(
        time=rainfall_data.coords["time"]
    )
    result_pop_only = water_hatching(rainfall_data, population_data_2d)
    xr.testing.assert_allclose(result_pop_only, expected_pop_only)

    # Restore original E_RAT for other tests
    monkeypatch.setitem(CONSTANTS_WATER_HATCHING, "E_RAT", C["E_RAT"])


def test_water_hatching_mismatched_spatial_dims_raises(rainfall_data):
    """Test that mismatched spatial coordinates raise a ValueError."""
    # Create population data with different coordinates
    mismatched_pop = xr.DataArray(
        np.zeros((2, 2)),
        dims=("latitude", "longitude"),
        coords={"latitude": [98, 99], "longitude": [100, 101]},
    )
    # The function should detect the misalignment and raise an error.
    with pytest.raises(ValueError, match="must be aligned"):
        water_hatching(rainfall_data, mismatched_pop)


def test_water_hatching_optimal_rainfall(
    rainfall_data, population_data_2d, monkeypatch
):
    """Test that the rainfall component is maximized at optimal rainfall."""
    C = CONSTANTS_WATER_HATCHING
    # Set rainfall to the optimal value
    optimal_rainfall = rainfall_data.copy(data=np.full(rainfall_data.shape, C["E_OPT"]))

    # Isolate the rainfall effect
    monkeypatch.setitem(CONSTANTS_WATER_HATCHING, "E_RAT", 0.0)
    result = water_hatching(optimal_rainfall, population_data_2d)

    # At optimal rainfall, the rainfall_hatch term should be 1.0
    # (1 + E_0) * exp(0) / (exp(0) + E_0) = (1 + E_0) / (1 + E_0) = 1
    expected = xr.full_like(rainfall_data, 1.0, dtype=float)
    xr.testing.assert_allclose(result, expected)


def test_water_hatching_zero_population(rainfall_data, population_data_2d, monkeypatch):
    """Test that the population component is minimized with zero population."""
    C = CONSTANTS_WATER_HATCHING
    # Set population to zero
    zero_population = population_data_2d.copy(data=np.zeros_like(population_data_2d))

    # Isolate the population effect
    monkeypatch.setitem(CONSTANTS_WATER_HATCHING, "E_RAT", 1.0)
    result = water_hatching(rainfall_data, zero_population)

    # With zero population, the population_hatch term should be at its minimum
    # E_DENS / (E_DENS + exp(0)) = E_DENS / (E_DENS + 1)
    expected_val = C["E_DENS"] / (C["E_DENS"] + 1.0)
    expected = xr.full_like(rainfall_data, expected_val, dtype=float)
    xr.testing.assert_allclose(result, expected)


def test_water_hatching_output_retains_coords(rainfall_data, population_data_2d):
    """Test that the output DataArray has the same coordinates as the input rainfall data."""
    result = water_hatching(rainfall_data, population_data_2d)
    assert result.dims == rainfall_data.dims
    for coord_name in rainfall_data.coords:
        xr.testing.assert_equal(
            result.coords[coord_name], rainfall_data.coords[coord_name]
        )


def test_water_hatching_regression(
    dummy_file_rainfall_dataset, dummy_file_population_dataset
):
    """
    Regression test: Compare water_hatching output against known expected values.

    This test ensures the function produces consistent results with real datasets.
    If this test fails, it indicates the implementation has changed.
    """
    result = water_hatching(dummy_file_rainfall_dataset, dummy_file_population_dataset)

    # Expected output from Octave baseline computation
    # Octave shape: (longitude=3, latitude=2, time=4)
    expected_values = np.array(
        [
            # longitude row 0
            [
                [2.4945e-02, 1.9995e-03, 1.9822e-03, 1.9822e-03],
                [2.9830e-03, 1.9882e-03, 1.9881e-03, 1.9881e-03],
            ],
            # longitude row 1
            [
                [1.0928e-02, 1.9878e-03, 1.9841e-03, 1.9841e-03],
                [2.2752e-03, 1.9900e-03, 1.9900e-03, 1.9900e-03],
            ],
            # longitude row 2
            [
                [5.1250e-03, 1.9868e-03, 1.9861e-03, 1.9861e-03],
                [2.0659e-03, 1.9920e-03, 1.9920e-03, 1.9920e-03],
            ],
        ]
    )

    # Compare dimensions
    expected_dim = (3, 2, 4)
    assert result.shape == expected_dim

    # Create xarray DataArray with same structure as result
    expected = xr.DataArray(expected_values, dims=result.dims, coords=result.coords)
    xr.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
