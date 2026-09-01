"""Unit tests for heiplanet_models.Pmodel.Pmodel_rates_development.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Third-party
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_CARRYING_CAPACITY,
    CONSTANTS_MOSQUITO_E,
)

# Local application
from heiplanet_models.Pmodel.Pmodel_rates_development import (
    carrying_capacity,
    mosq_dev_e,
    mosq_dev_i,
    mosq_dev_j,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def typical_temperature_array():
    """Provides a 1D numpy array of typical temperature values."""
    return np.array([10, 20, 25, 30], dtype=float)


@pytest.fixture
def typical_temperature_array_i():
    """Provides a 1D numpy array of typical temperature values for mosq_dev_i."""
    return np.array([10, 20, 25, 30], dtype=float)


@pytest.fixture
def typical_temperature_array_e():
    """Provides a 1D numpy array of typical temperature values for mosq_dev_e."""
    return np.array([15, 20, 25, 30], dtype=float)


@pytest.fixture
def rainfall_data_fixture():
    """Provides a sample rainfall DataArray."""
    return xr.DataArray(
        np.ones((2, 2, 3)),
        dims=["longitude", "latitude", "time"],
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def population_data_time_independent_fixture():
    """Provides a sample time-independent population DataArray."""
    return xr.DataArray(
        np.full((2, 2), 0.5),
        dims=["longitude", "latitude"],
        coords={"latitude": [10, 20], "longitude": [30, 40]},
    )


@pytest.fixture
def population_data_time_dependent_fixture():
    """Provides a sample time-dependent population DataArray."""

    generator = np.random.default_rng(42)
    data = generator.standard_normal((2, 2, 3))

    return xr.DataArray(
        data,
        dims=["longitude", "latitude", "time"],
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def single_time_step_rainfall_fixture():
    """Provides a rainfall DataArray with a single time step."""
    return xr.DataArray(
        np.ones((2, 2, 1)),
        dims=["longitude", "latitude", "time"],
        coords={
            "time": pd.to_datetime(["2024-01-01"]),
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def single_time_step_population_fixture():
    """Provides a population DataArray with a single time step."""
    return xr.DataArray(
        np.full((2, 2, 1), 0.5),
        dims=["longitude", "latitude", "time"],
        coords={
            "time": pd.to_datetime(["2024-01-01"]),
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def zero_rainfall_fixture():
    """Provides a rainfall DataArray with all zeros."""
    return xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=["longitude", "latitude", "time"],
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def zero_population_fixture():
    """Provides a population DataArray with all zeros."""
    return xr.DataArray(
        np.zeros((2, 2)),
        dims=["longitude", "latitude"],
        coords={"latitude": [10, 20], "longitude": [30, 40]},
    )


@pytest.fixture
def rainfall_with_nan_fixture():
    """Provides a rainfall DataArray containing NaN values."""
    data = np.ones((2, 2, 3))
    data[1, 0, 0] = np.nan
    return xr.DataArray(
        data,
        dims=["longitude", "latitude", "time"],
        coords={
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


# ------------------------------------
# ---- function under test: mosq_dev_j
# ------------------------------------
def test_mosq_dev_j_typical_temperatures(typical_temperature_array):
    """Test mosq_dev_j with typical temperature values."""
    result = mosq_dev_j(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_dev_j_scalar_input():
    """Test mosq_dev_j with a scalar temperature value."""
    temp = 25.0
    result = mosq_dev_j(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_dev_j_edge_case_temperatures():
    """Test mosq_dev_j with edge case temperatures."""
    temps = np.array([0, 50], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_j_negative_temperatures():
    """Test mosq_dev_j with negative temperature values."""
    temps = np.array([-10, 0, 10], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_j_large_temperatures():
    """Test mosq_dev_j with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_dev_j_empty_array():
    """Test mosq_dev_j with an empty array."""
    temps = np.array([])
    result = mosq_dev_j(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_dev_j_non_numeric_input():
    """Test mosq_dev_j with non-numeric input values."""
    with pytest.raises((TypeError, ValueError)):
        mosq_dev_j(np.array(["a", None, np.nan], dtype=object))


def test_mosq_dev_j_output_consistency():
    """Test mosq_dev_j output matches manual calculation for known input."""
    # Example constants (should match those in the model)
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_J

    CONST_1 = CONSTANTS_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_J["CONST_4"]
    T = np.array([15.0, 25.0])
    expected = CONST_4 / (CONST_1 - CONST_2 * T + CONST_3 * T**2)
    result = mosq_dev_j(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_dev_j_multidimensional_input():
    """Test mosq_dev_j with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_j_known_matrix_output():
    """Test mosq_dev_j returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.036536, 0.058754], [0.093720, 0.120192]])
    result = mosq_dev_j(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ------------------------------------
# ---- function under test: mosq_dev_i
# ------------------------------------
def test_mosq_dev_i_typical_temperatures(typical_temperature_array_i):
    """Test mosq_dev_i with typical temperature values."""
    result = mosq_dev_i(typical_temperature_array_i)
    assert result.shape == typical_temperature_array_i.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_dev_i_scalar_input():
    """Test mosq_dev_i with a scalar temperature value."""
    temp = 25.0
    result = mosq_dev_i(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_dev_i_edge_case_temperatures():
    """Test mosq_dev_i with edge case temperatures."""
    temps = np.array([0, 50], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_negative_temperatures():
    """Test mosq_dev_i with negative temperature values."""
    temps = np.array([-10, 0, 10], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_large_temperatures():
    """Test mosq_dev_i with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_non_numeric_input():
    """Test mosq_dev_i with non-numeric input values."""
    with pytest.raises((TypeError, ValueError)):
        mosq_dev_i(np.array(["a", None, np.nan], dtype=object))


def test_mosq_dev_i_output_consistency():
    """Test mosq_dev_i output matches manual calculation for known input."""
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_I

    CONST_1 = CONSTANTS_MOSQUITO_I["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_I["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_I["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_I["CONST_4"]
    T = np.array([15.0, 25.0])
    expected = CONST_4 / (CONST_1 - CONST_2 * T + CONST_3 * T**2)
    result = mosq_dev_i(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_dev_i_multidimensional_input():
    """Test mosq_dev_i with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_empty_array():
    """Test mosq_dev_i with an empty array."""
    temps = np.array([])
    result = mosq_dev_i(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_dev_i_known_matrix_output():
    """Test mosq_dev_i returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.083229, 0.160772], [0.258065, 0.200803]])
    result = mosq_dev_i(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ------------------------------------
# ---- function under test: mosq_dev_e
# ------------------------------------
def test_mosq_dev_e_scalar_input():
    """Test mosq_dev_e with a scalar temperature value."""
    temp = 25.0
    result = mosq_dev_e(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_dev_e_typical_temperatures(typical_temperature_array_e):
    """Test mosq_dev_e with typical temperature values."""
    result = mosq_dev_e(typical_temperature_array_e)
    assert result.shape == typical_temperature_array_e.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_dev_e_boundary_temperatures():
    """Test mosq_dev_e with boundary temperatures T0 and Tm."""
    temperature_min = CONSTANTS_MOSQUITO_E["T0"]
    temperature_max = CONSTANTS_MOSQUITO_E["Tm"]
    temps = np.array([temperature_min, temperature_max])
    result = mosq_dev_e(temps)
    expected = np.array([0.0, 0.0])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_mosq_dev_e_large_temperatures():
    """Test mosq_dev_e with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    with np.errstate(invalid="ignore"):
        result = mosq_dev_e(temps)
    assert result.shape == temps.shape
    # Values above Tm should be nan
    assert np.all(np.isnan(result[temps > CONSTANTS_MOSQUITO_E["Tm"]]))


def test_mosq_dev_e_empty_array():
    """Test mosq_dev_e with an empty array."""
    temps = np.array([])
    result = mosq_dev_e(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_dev_e_multidimensional_input():
    """Test mosq_dev_e with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    with np.errstate(invalid="ignore"):
        result = mosq_dev_e(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result[np.nonzero(temps <= CONSTANTS_MOSQUITO_E["Tm"])]))


def test_mosq_dev_e_output_consistency():
    """Test mosq_dev_e output matches manual calculation for known input."""
    q = CONSTANTS_MOSQUITO_E["q"]
    temperature_min = CONSTANTS_MOSQUITO_E["T0"]
    temperature_max = CONSTANTS_MOSQUITO_E["Tm"]
    T = np.array([15.0, 25.0])
    expected = q * T * (T - temperature_min) * ((temperature_max - T) ** (1 / 2))
    result = mosq_dev_e(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_dev_e_non_numeric_input():
    """Test mosq_dev_e with non-numeric input values."""
    with pytest.raises(TypeError):
        mosq_dev_e("a string")
    with pytest.raises(TypeError):
        mosq_dev_e([25.0, "30.0"])


def test_mosq_dev_e_known_matrix_output():
    """Test mosq_dev_e returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    # Expected values calculated manually based on the corrected formula
    expected = np.array([[0.1800, 0.2532, 0.3016, 0.2819]])
    # Reshape to match input
    expected = expected.reshape(2, 2)
    result = mosq_dev_e(T)
    np.testing.assert_allclose(result, expected, atol=1e-4)


# ---- Unit Tests for carrying_capacity
def test_carrying_capacity_time_independent_population(
    rainfall_data_fixture, population_data_time_independent_fixture
):
    """
    Test carrying_capacity with time-independent population data.

    This test validates that the function correctly handles population_data
    without a 'time' dimension by broadcasting it across all time steps.
    """
    # Call the function with test data
    result = carrying_capacity(
        rainfall_data_fixture, population_data_time_independent_fixture
    )

    # 1. Verify that the output is an xarray.DataArray
    assert isinstance(result, xr.DataArray)

    # 2. Verify that the output shape matches the rainfall_data shape
    assert result.shape == rainfall_data_fixture.shape

    # # 3. Verify the initial condition calculation
    rainfall_weight = CONSTANTS_CARRYING_CAPACITY["ALPHA_RAIN"]
    population_weight = CONSTANTS_CARRYING_CAPACITY["ALPHA_DENS"]
    capacity_scale_factor = CONSTANTS_CARRYING_CAPACITY["LAMBDA"]

    expected_initial_capacity_unscaled = (
        rainfall_weight * rainfall_data_fixture.isel(time=0)
        + population_weight * population_data_time_independent_fixture
    )
    expected_initial_capacity_scaled = (
        expected_initial_capacity_unscaled * capacity_scale_factor
    )

    # Check if the calculated initial result matches the expected one
    xr.testing.assert_allclose(result.isel(time=0), expected_initial_capacity_scaled)


def test_carrying_capacity_time_dependent_population(
    rainfall_data_fixture, population_data_time_dependent_fixture
):
    """
    Test carrying_capacity with time-dependent population data.
    """
    result = carrying_capacity(
        rainfall_data_fixture, population_data_time_dependent_fixture
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == rainfall_data_fixture.shape
    assert not np.isnan(result.values).any()


def test_carrying_capacity_single_time_step(
    single_time_step_rainfall_fixture, single_time_step_population_fixture
):
    """
    Test carrying_capacity with a single time step.
    """
    result = carrying_capacity(
        single_time_step_rainfall_fixture, single_time_step_population_fixture
    )
    assert result.shape == single_time_step_rainfall_fixture.shape
    assert result.time.size == 1


def test_carrying_capacity_multiple_time_steps(
    rainfall_data_fixture, population_data_time_independent_fixture
):
    """
    Test the recursive calculation over multiple time steps.
    """
    result = carrying_capacity(
        rainfall_data_fixture, population_data_time_independent_fixture
    )

    rainfall_weight = CONSTANTS_CARRYING_CAPACITY["ALPHA_RAIN"]
    population_weight = CONSTANTS_CARRYING_CAPACITY["ALPHA_DENS"]
    gamma = CONSTANTS_CARRYING_CAPACITY["GAMMA"]
    lambda_val = CONSTANTS_CARRYING_CAPACITY["LAMBDA"]

    # Manually calculate the first few steps
    # Step k=0
    initial_capacity_unscaled = (
        rainfall_weight * rainfall_data_fixture.isel(time=0)
        + population_weight * population_data_time_independent_fixture
    )
    # The scaling factor is not applied for k=0 in the implementation's loop,
    # so we only apply the final LAMBDA scaling.
    result0 = initial_capacity_unscaled * lambda_val
    xr.testing.assert_allclose(result.isel(time=0), result0)

    # Step k=1
    # The recursive step uses the unscaled previous value of A.
    second_step_capacity_unscaled = (
        gamma * initial_capacity_unscaled
        + rainfall_weight * rainfall_data_fixture.isel(time=1)
        + population_weight * population_data_time_independent_fixture
    )
    factor1 = (1 - gamma) / (1 - gamma**2)
    second_step_capacity_scaled = second_step_capacity_unscaled * factor1
    expected_result1 = second_step_capacity_scaled * lambda_val

    # Assign the correct time coordinate to the expected result
    expected_result1["time"] = rainfall_data_fixture.time[1]

    xr.testing.assert_allclose(result.isel(time=1), expected_result1)


def test_carrying_capacity_with_zeros(zero_rainfall_fixture, zero_population_fixture):
    """
    Test carrying_capacity with zero rainfall and population.
    """
    result = carrying_capacity(zero_rainfall_fixture, zero_population_fixture)
    assert (result.values == 0).all()


def test_carrying_capacity_output_properties(
    rainfall_data_fixture, population_data_time_independent_fixture
):
    """
    Test the data type and shape of the output.
    """
    result = carrying_capacity(
        rainfall_data_fixture, population_data_time_independent_fixture
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == rainfall_data_fixture.shape
    assert result.dtype == float


def test_carrying_capacity_regression(
    dummy_file_rainfall_dataset, dummy_file_population_dataset
):
    """
    Regression test: Compare carrying_capacity output against known expected values.

    This test ensures the function produces consistent results with real datasets.
    If this test fails, it indicates the implementation has changed.
    """
    result = carrying_capacity(
        dummy_file_rainfall_dataset, dummy_file_population_dataset
    )

    # Expected output from Octave baseline computation
    # Octave shape: (latitude=3, longitude=2, time=4)
    expected_values = np.array(
        [
            # latitude row 0
            [
                [-6.2438e07, -2.5981e08, -4.6373e08, -6.7417e08],
                [-2.4975e08, -4.4712e08, -6.5104e08, -8.6148e08],
            ],
            # latitude row 1
            [
                [-1.2488e08, -3.2224e08, -5.2617e08, -7.3661e08],
                [-3.1219e08, -5.0956e08, -7.1348e08, -9.2392e08],
            ],
            # latitude row 2
            [
                [-1.8731e08, -3.8468e08, -5.8860e08, -7.9905e08],
                [-3.7462e08, -5.7199e08, -7.7592e08, -9.8636e08],
            ],
        ]
    )
    expected_values = xr.DataArray(
        expected_values, dims=result.dims, coords=result.coords
    )

    # Compare dimensions
    expected_dim = (3, 2, 4)
    assert result.shape == expected_dim

    # Create xarray DataArray with same structure as result

    xr.testing.assert_allclose(result, expected_values, rtol=1e-4, atol=1e-4)
