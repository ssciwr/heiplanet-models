"""Unit tests for heiplanet_models.Pmodel.Pmodel_ode.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pytest
import xarray as xr
from heiplanet_models.Pmodel.Pmodel_ode import (
    rk4_step,
    albopictus_ode_system,
    albopictus_log_ode_system,
    solve_system,
)

# =============================================================================
# TEST UTILITIES
# =============================================================================

# ---- Modern NumPy Random Generator
rng = np.random.default_rng(12345)

# ---- Common Test Utilities


def create_test_state(shape=(2, 2, 5)):
    """Create a random test state array using modern NumPy generator.
    Default shape: (lon, lat, variables) = (2, 2, 5)
    - lon: 2 longitude points
    - lat: 2 latitude points
    - variables: 5 population compartments

    Note: ODE functions operate on single time steps.
    For multi-timestep state, use shape like (2, 2, 5, time_steps)
    """
    return rng.random(shape)


def create_ode_params(time_idx=0):
    """Create standard parameter tuple for ODE system tests.
    Parameters are for a single time step.
    """
    return (
        time_idx,  # time_index
        1.0,  # time_step
        rng.random((2, 2)),  # carrying_capacity (lon, lat) - single time
        rng.random((2, 2)),  # birth_rate (lon, lat)
        rng.random((2, 2)),  # diapause_laying_fraction (lon, lat)
        rng.random((2, 2)),  # diapause_hatching_fraction (lon, lat)
        rng.random((2, 2)),  # egg_mortality (lon, lat)
        rng.random((2, 2)),  # juvenile_mortality (lon, lat)
        rng.random((2, 2)),  # adult_mortality (lon, lat)
        rng.random((2, 2)),  # egg_diapause_survival (lon, lat)
        rng.random((2, 2)),  # juvenile_development (lon, lat)
        rng.random((2, 2)),  # immature_development (lon, lat)
        0.5,  # egg_development
        rng.random((2, 2)),  # water_hatching_rate (lon, lat)
    )


def assert_shape_preserved(result, expected_shape):
    """Assert that result has expected shape."""
    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {result.shape}"
    )


def assert_all_finite(result):
    """Assert that all values in result are finite."""
    assert np.all(np.isfinite(result)), "Result contains non-finite values"


def assert_no_negatives(result):
    """Assert that result contains no negative values."""
    assert np.all(result >= 0), "Result contains negative values"


def assert_nan_propagation(result):
    """Assert that result contains NaN values (for NaN propagation tests)."""
    assert np.any(np.isnan(result)), "NaN values were not propagated"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def test_state():
    """Fixture for common test state.
    Shape: (lon, lat, variables) = (2, 2, 5) - single time step
    """
    return create_test_state()


@pytest.fixture
def ode_params():
    """Fixture for common ODE parameters."""
    return create_ode_params()


@pytest.fixture
def temperature_array():
    """Fixture for temperature array (lon, lat, time) = (2, 2, 10)."""
    return xr.DataArray(rng.random((2, 2, 10)), dims=["longitude", "latitude", "time"])


@pytest.fixture
def temperature_mean_array():
    """Fixture for temperature mean array (lon, lat, time) = (2, 2, 10).
    Time dimension is based on temperature.shape[2] / time_step = 10 / 1.0 = 10."""
    return xr.DataArray(rng.random((2, 2, 10)), dims=["longitude", "latitude", "time"])


@pytest.fixture
def latitudes_array():
    """Fixture for latitudes 1D array (lat,) = (2,)."""
    return xr.DataArray(rng.random(2), dims=["latitude"])


@pytest.fixture
def carrying_capacity_array():
    """Fixture for carrying capacity array (lon, lat, time) = (2, 2, 10).
    Time dimension is based on temperature.shape[2] / time_step = 10 / 1.0 = 10."""
    return xr.DataArray(rng.random((2, 2, 10)), dims=["longitude", "latitude", "time"])


@pytest.fixture
def egg_activate_array():
    """Fixture for egg activation array (lon, lat, time) = (2, 2, 10).
    Time dimension is based on temperature.shape[2] / time_step = 10 / 1.0 = 10."""
    return xr.DataArray(rng.random((2, 2, 10)), dims=["longitude", "latitude", "time"])


@pytest.fixture
def call_function_test_arrays(
    temperature_array,
    temperature_mean_array,
    latitudes_array,
    carrying_capacity_array,
    egg_activate_array,
):
    """Fixture combining all arrays needed for call_function tests.
    All arrays have matching time dimensions (10 time steps)."""
    return {
        "temperature": temperature_array,
        "temperature_mean": temperature_mean_array,
        "latitudes": latitudes_array,
        "carrying_capacity": carrying_capacity_array,
        "egg_activate": egg_activate_array,
    }


@pytest.fixture
def call_function_initial_state():
    """Initial state for call_function tests.
    Shape: (lon, lat, variables) = (2, 2, 5) - single time step for initial condition
    """
    return create_test_state()


@pytest.fixture
def call_function_random_state():
    """Random state for call_function tests.
    Shape: (lon, lat, variables) = (2, 2, 5) - single time step for initial condition
    """
    return create_test_state()


# ---- Helper Functions


def negative_ode(state, params):
    """Returns a negative derivative, which would drive state negative."""
    return -2.0 * state - 1.0


def dummy_log_ode_nonneg(state, params):
    """Returns zeros, so log-correction should keep state at minimum allowed."""
    return np.zeros_like(state)


def zero_ode(state, params):
    """Returns zero derivative, so state should remain unchanged unless log correction is triggered."""
    return np.zeros_like(state)


def dummy_log_ode_safe(state, params):
    return np.ones_like(state) * 0.01


def shape_ode(state, params):
    return rng.random(state.shape)


def shape_log_ode(state, params):
    return rng.random(state.shape)


# ---- Tests RK4 Method


def test_rk4_step_negative_value_correction():
    state = create_test_state()
    params = ()
    time_step = 1.0

    result = rk4_step(
        ode_func=negative_ode,
        log_ode_func=dummy_log_ode_nonneg,
        state=state,
        model_params=params,
        time_step=time_step,
    )

    assert_shape_preserved(result, state.shape)
    assert_no_negatives(result)


def test_rk4_step_log_ode_path_trigger():
    state = np.full((2, 2, 5), 1e-30)
    params = ()
    time_step = 1.0

    result = rk4_step(
        ode_func=zero_ode,
        log_ode_func=dummy_log_ode_safe,
        state=state,
        model_params=params,
        time_step=time_step,
    )

    assert_shape_preserved(result, state.shape)
    assert_all_finite(result)
    assert_no_negatives(result)


def test_rk4_step_shape_preservation():
    state = create_test_state()
    params = ()
    time_step = 1.0

    result = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=state,
        model_params=params,
        time_step=time_step,
    )

    assert_shape_preserved(result, state.shape)


def test_rk4_step_shape_preservation_multidim():
    """Test with extra dimensions (lon, lat, variables, extra_dim)"""
    state = create_test_state(shape=(3, 4, 5, 6))
    params = ()
    time_step = 1.0

    result = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=state,
        model_params=params,
        time_step=time_step,
    )

    assert_shape_preserved(result, state.shape)


def test_rk4_step_no_side_effects():
    state = create_test_state()
    state_copy = state.copy()
    params = ()
    time_step = 1.0

    rk4_step(shape_ode, shape_log_ode, state, params, time_step)

    assert np.array_equal(state, state_copy), "Input state was modified"


def test_rk4_step_zero_state():
    state = np.zeros((2, 2, 5))
    params = ()
    time_step = 1.0

    result = rk4_step(zero_ode, dummy_log_ode_safe, state, params, time_step)

    assert_all_finite(result)
    assert_no_negatives(result)


def test_rk4_step_parameter_passing():
    state = create_test_state()
    test_value = 42.0
    params = (test_value,)
    time_step = 1.0

    def param_ode(s, p):
        assert p[0] == test_value
        return np.ones_like(s)

    def param_log_ode(s, p):
        assert p[0] == test_value
        return np.ones_like(s)

    rk4_step(
        ode_func=param_ode,
        log_ode_func=param_log_ode,
        state=state,
        model_params=params,
        time_step=time_step,
    )


def test_rk4_step_nan_in_state():
    state = create_test_state()
    state[0, 0, 0] = np.nan
    params = ()
    time_step = 1.0

    result = rk4_step(shape_ode, shape_log_ode, state, params, time_step)

    assert_nan_propagation(result)


# ---- Tests for albopictus_ode_system


def test_albopictus_ode_system_shape_preservation(test_state, ode_params):
    result = albopictus_ode_system(test_state, ode_params)
    assert_shape_preserved(result, test_state.shape)


def test_albopictus_ode_system_positive_inputs_finite_output(test_state, ode_params):
    result = albopictus_ode_system(test_state, ode_params)
    assert_all_finite(result)


def test_albopictus_ode_system_zero_state(ode_params):
    state = np.zeros((2, 2, 5))
    result = albopictus_ode_system(state, ode_params)
    assert_all_finite(result)


def test_albopictus_ode_system_nan_handling(test_state, ode_params):
    test_state[0, 0, 0] = np.nan
    result = albopictus_ode_system(test_state, ode_params)
    assert_nan_propagation(result)


def test_albopictus_ode_system_internal_nan_generation():
    state = create_test_state()
    params = create_ode_params()
    # Force -inf by setting carrying_capacity to zero
    params = (params[0], params[1], np.zeros((2, 2)), *params[3:])

    result = albopictus_ode_system(state, params)

    # When carrying capacity is zero, division by zero produces -inf
    # Check that the function completes without crashing
    assert result.shape == state.shape
    # The function allows -inf values when CC=0, so we check for that specific behavior
    assert np.any(np.isinf(result)), (
        "Expected -inf values when carrying capacity is zero"
    )


def test_albopictus_ode_system_parameter_unpacking(test_state):
    params = create_ode_params()
    result = albopictus_ode_system(test_state, params)
    assert_shape_preserved(result, test_state.shape)


# ---- Tests for albopictus_log_ode_system


def test_albopictus_log_ode_system_shape_preservation(test_state, ode_params):
    result = albopictus_log_ode_system(test_state, ode_params)
    assert_shape_preserved(result, test_state.shape)


def test_albopictus_log_ode_system_positive_inputs_finite_output(
    test_state, ode_params
):
    result = albopictus_log_ode_system(test_state, ode_params)
    assert_all_finite(result)


def test_albopictus_log_ode_system_zero_state(ode_params):
    state = np.zeros((2, 2, 5))
    result = albopictus_log_ode_system(state, ode_params)
    assert_all_finite(result)


def test_albopictus_log_ode_system_nan_handling(test_state, ode_params):
    test_state[0, 0, 0] = np.nan
    result = albopictus_log_ode_system(test_state, ode_params)
    assert_nan_propagation(result)


def test_albopictus_log_ode_system_internal_nan_correction():
    state = create_test_state()
    params = create_ode_params()
    # Force potential NaN/Inf by setting carrying_capacity to zero
    params = (params[0], params[1], np.zeros((2, 2)), *params[3:])

    result = albopictus_log_ode_system(state, params)

    # When carrying capacity is zero, division by zero produces -inf
    # Check that the function completes without crashing
    assert result.shape == state.shape
    # The function allows -inf values when CC=0, so we check for that specific behavior
    assert np.any(np.isinf(result)), (
        "Expected -inf values when carrying capacity is zero"
    )


def test_albopictus_log_ode_system_parameter_unpacking(test_state):
    params = create_ode_params()
    result = albopictus_log_ode_system(test_state, params)
    assert_shape_preserved(result, test_state.shape)


def test_albopictus_log_ode_system_negative_state_correction():
    state = -1.0 * create_test_state()
    params = create_ode_params()

    result = albopictus_log_ode_system(state, params)

    assert_all_finite(result)


# ---- Tests for call_function


def test_call_function_shape_preservation(
    call_function_initial_state,
    temperature_array,
    temperature_mean_array,
    latitudes_array,
    carrying_capacity_array,
    egg_activate_array,
):
    """Test that call_function preserves expected shape.
    Input state: (lon, lat, variables) = (2, 2, 5)
    Temperature: (lon, lat, time) = (2, 2, 10)
    Output time steps: 10 / 1.0 = 10
    Expected output: (lon, lat, variables, time) = (2, 2, 5, 10)
    """
    result = solve_system(
        state=call_function_initial_state,
        temperature=temperature_array,
        temperature_mean=temperature_mean_array,
        latitudes=latitudes_array,
        carrying_capacity=carrying_capacity_array,
        egg_activate=egg_activate_array,
        time_step=1.0,
    )

    expected_shape = (2, 2, 5, 10)  # (lon, lat, variables, output_time_steps)
    assert_shape_preserved(result, expected_shape)


def test_call_function_initial_state_propagation(
    call_function_initial_state,
    temperature_array,
    temperature_mean_array,
    latitudes_array,
    carrying_capacity_array,
    egg_activate_array,
):
    """Test that call_function produces evolved state (not initial state at t=0).

    The ODE solver integrates forward from the initial state, so result[..., 0]
    contains the state after the first integration step, not the initial state.
    """
    result = solve_system(
        call_function_initial_state,
        temperature_array,
        temperature_mean_array,
        latitudes_array,
        carrying_capacity_array,
        egg_activate_array,
        1.0,
    )

    # Verify that the first time slice differs from initial state (integration occurred)
    # result shape: (lon, lat, variables, time)
    assert not np.array_equal(result[..., 0], call_function_initial_state)
    # All values should still be finite and non-negative
    assert np.all(np.isfinite(result[..., 0]))
    assert np.all(result[..., 0] >= 0)


def test_call_function_integration_progression(
    call_function_test_arrays, call_function_initial_state
):
    """Test that integration progresses through all time steps."""
    result = solve_system(
        call_function_initial_state,
        call_function_test_arrays["temperature"],
        call_function_test_arrays["temperature_mean"],
        call_function_test_arrays["latitudes"],
        call_function_test_arrays["carrying_capacity"],
        call_function_test_arrays["egg_activate"],
        1.0,
    )

    # Check time dimension (last axis): temperature has 10 steps, output has 10/1.0=10 steps
    assert result.shape[3] == 10
    assert_all_finite(result)


def test_call_function_zero_state(call_function_test_arrays):
    """Test call_function with zero initial state."""
    state = np.zeros((2, 2, 5))  # (lon, lat, variables)

    result = solve_system(
        state,
        call_function_test_arrays["temperature"],
        call_function_test_arrays["temperature_mean"],
        call_function_test_arrays["latitudes"],
        call_function_test_arrays["carrying_capacity"],
        call_function_test_arrays["egg_activate"],
        1.0,
    )

    assert_all_finite(result)


def test_call_function_single_time_step(call_function_initial_state, latitudes_array):
    """Test call_function with single time step.
    Temperature has 1 raw time step, output has 1/1.0=1 time step.
    """
    # Create single raw timestep arrays
    temp = xr.DataArray(rng.random((2, 2, 1)), dims=["longitude", "latitude", "time"])
    # Derived arrays need 1/1.0=1 output time step
    temp_mean = xr.DataArray(
        rng.random((2, 2, 1)), dims=["longitude", "latitude", "time"]
    )
    k = xr.DataArray(rng.random((2, 2, 1)), dims=["longitude", "latitude", "time"])
    egg_act = xr.DataArray(
        rng.random((2, 2, 1)), dims=["longitude", "latitude", "time"]
    )

    result = solve_system(
        call_function_initial_state, temp, temp_mean, latitudes_array, k, egg_act, 1.0
    )

    expected_shape = (2, 2, 5, 1)  # (lon, lat, variables, output_time_steps: 1/1.0=1)
    assert_shape_preserved(result, expected_shape)


def test_call_function_seasonal_diapause_reset_branch(
    call_function_initial_state,
    latitudes_array,
):
    """Cover seasonal reset branch where diapause egg compartment is zeroed on day 200."""
    n_time = 201
    temp = xr.DataArray(
        rng.random((2, 2, n_time)), dims=["longitude", "latitude", "time"]
    )
    temp_mean = xr.DataArray(
        rng.random((2, 2, n_time)),
        dims=["longitude", "latitude", "time"],
    )
    carrying_capacity = xr.DataArray(
        rng.random((2, 2, n_time)),
        dims=["longitude", "latitude", "time"],
    )
    egg_activate = xr.DataArray(
        rng.random((2, 2, n_time)),
        dims=["longitude", "latitude", "time"],
    )

    result = solve_system(
        state=call_function_initial_state,
        temperature=temp,
        temperature_mean=temp_mean,
        latitudes=latitudes_array,
        carrying_capacity=carrying_capacity,
        egg_activate=egg_activate,
        time_step=1.0,
    )

    # Compartment index 1 (diapause eggs) is reset on day 200 and then stored.
    assert np.allclose(result[:, :, 1, 200], 0.0)
