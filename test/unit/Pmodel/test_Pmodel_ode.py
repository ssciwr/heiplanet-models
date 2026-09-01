"""Unit tests for heiplanet_models.Pmodel.Pmodel_ode.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel import Pmodel_ode
from heiplanet_models.Pmodel.Pmodel_ode import (
    SolverFallbackMetrics,
    _ensure_1d_dataarray,
    _ensure_3d_dataarray,
    _log_fallback_metrics,
    _log_scipy_capacity_mask,
    _normalize_solver_inputs,
    _solve_system_scipy_loaded_chunk,
    _spatial_chunks,
    _validate_daily_time_lengths,
    _validate_scipy_method,
    _validate_spatial_shapes,
    _validate_steps_per_day,
    albopictus_log_ode_system,
    albopictus_ode_system,
    rk4_step,
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


def test_log_fallback_metrics_warns_when_recovery_occurs(caplog):
    """Test that numerical recovery is reported once with both event counts."""
    metrics = SolverFallbackMetrics(
        derivative_substitutions=2,
        output_normalizations=3,
    )

    with caplog.at_level("WARNING"):
        _log_fallback_metrics("scipy_chunked", metrics)

    assert "2 derivative substitutions, 3 output normalizations" in caplog.text


@pytest.mark.parametrize(
    ("time_step", "message"),
    [(np.nan, "finite"), (1.5, "integer-like"), (0, "positive")],
)
def test_solver_time_step_validation_rejects_invalid_values(time_step, message):
    """Test that solver time steps must be finite positive integers."""
    with pytest.raises(ValueError, match=message):
        _validate_steps_per_day(time_step)


def test_solver_validation_rejects_invalid_method_and_shapes():
    """Test validation errors for unsupported methods and malformed arrays."""
    with pytest.raises(ValueError, match="Unknown scipy_method"):
        _validate_scipy_method("invalid")

    with pytest.raises(ValueError, match="must contain dimensions"):
        _ensure_3d_dataarray(xr.DataArray([1.0], dims=["time"]), "temperature")
    with pytest.raises(ValueError, match="must be a 3D array"):
        _ensure_3d_dataarray(np.ones((1, 1)), "temperature")
    with pytest.raises(ValueError, match="must contain dimension"):
        _ensure_1d_dataarray(xr.DataArray([1.0], dims=["longitude"]), "latitudes")
    with pytest.raises(ValueError, match="must be a 1D array"):
        _ensure_1d_dataarray(np.ones((1, 1)), "latitudes")


def test_solver_normalization_accepts_numpy_arrays_and_rejects_extra_dimensions():
    """Test canonical NumPy conversion and dimensional validation paths."""
    spatial_data = np.ones((1, 1, 1))
    latitude_data = np.ones(1)

    normalized = _normalize_solver_inputs(
        state=np.ones((1, 1, 5)),
        temperature=None,
        temperature_mean=spatial_data,
        latitudes=latitude_data,
        carrying_capacity=spatial_data,
        egg_activate=spatial_data,
        time_step=1,
        require_temperature=False,
        validate_temperature_spatial_shape=False,
    )

    assert normalized.temperature is None
    assert normalized.temperature_mean.dims == ("longitude", "latitude", "time")
    assert normalized.latitudes.dims == ("latitude",)

    with pytest.raises(ValueError, match="temperature spatial shape"):
        _normalize_solver_inputs(
            state=np.ones((1, 1, 5)),
            temperature=None,
            temperature_mean=np.ones((2, 1, 1)),
            latitudes=latitude_data,
            carrying_capacity=np.ones((1, 1, 1)),
            egg_activate=np.ones((1, 1, 1)),
            time_step=1,
            require_temperature=False,
            validate_temperature_spatial_shape=True,
        )

    with pytest.raises(ValueError, match="must be 3D"):
        _ensure_3d_dataarray(
            xr.DataArray(
                np.ones((1, 1, 1, 1)),
                dims=["longitude", "latitude", "time", "extra"],
            ),
            "temperature",
        )
    with pytest.raises(ValueError, match="must be 1D"):
        _ensure_1d_dataarray(
            xr.DataArray(np.ones((1, 1)), dims=["latitude", "extra"]),
            "latitudes",
        )


def test_solver_validation_rejects_incompatible_input_dimensions():
    """Test validation errors for incompatible spatial and temporal inputs."""
    daily = xr.DataArray(np.ones((1, 1, 1)), dims=["longitude", "latitude", "time"])
    with pytest.raises(ValueError, match="state must have shape"):
        _validate_spatial_shapes(
            state=np.ones((1, 1)),
            temperature=daily,
            temperature_mean=daily,
            carrying_capacity=daily.values,
            egg_activate=daily.values,
        )
    with pytest.raises(ValueError, match="must contain 5 model variables"):
        _validate_spatial_shapes(
            state=np.ones((1, 1, 4)),
            temperature=daily,
            temperature_mean=daily,
            carrying_capacity=daily.values,
            egg_activate=daily.values,
        )
    with pytest.raises(ValueError, match="temperature spatial shape"):
        _validate_spatial_shapes(
            state=np.ones((1, 1, 5)),
            temperature=xr.DataArray(
                np.ones((2, 1, 1)), dims=["longitude", "latitude", "time"]
            ),
            temperature_mean=daily,
            carrying_capacity=daily.values,
            egg_activate=daily.values,
        )
    with pytest.raises(ValueError, match="at least one daily"):
        _validate_daily_time_lengths(
            temperature=None,
            temperature_mean=xr.DataArray(
                np.ones((1, 1, 0)), dims=["longitude", "latitude", "time"]
            ),
            steps_per_day=1,
        )
    with pytest.raises(ValueError, match="Expected temperature time length"):
        _validate_daily_time_lengths(
            temperature=daily,
            temperature_mean=daily,
            steps_per_day=2,
        )


def test_solver_validation_requires_temperature_for_legacy_backends():
    """Test that legacy solver normalization requires sub-daily temperature."""
    with pytest.raises(ValueError, match="temperature is required"):
        _normalize_solver_inputs(
            state=np.ones((1, 1, 5)),
            temperature=None,
            temperature_mean=np.ones((1, 1, 1)),
            latitudes=np.ones(1),
            carrying_capacity=np.ones((1, 1, 1)),
            egg_activate=np.ones((1, 1, 1)),
            time_step=1,
            require_temperature=True,
            validate_temperature_spatial_shape=True,
        )


def test_spatial_chunk_validation_and_capacity_logging(caplog):
    """Test invalid spatial chunks and both capacity status log messages."""
    with pytest.raises(ValueError, match="must be positive"):
        list(_spatial_chunks(1, 1, 0, 1))

    with caplog.at_level("INFO"):
        _log_scipy_capacity_mask("scipy_chunked", np.array([[True]]))
    assert "solving all 1 spatial cells" in caplog.text

    with caplog.at_level("WARNING"):
        _log_scipy_capacity_mask("scipy_chunked", np.array([[False]]))
    assert "skipped 1/1 spatial cells" in caplog.text


def test_rk4_step_requires_a_step_count():
    """Test that legacy RK4 requires either step-count argument."""
    with pytest.raises(TypeError, match="requires steps_per_day"):
        rk4_step(zero_ode, dummy_log_ode_safe, np.ones((1, 1, 5)), ())


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


# ---- Tests for solver backends


def create_stable_backend_inputs(n_time=3):
    state = xr.DataArray(
        np.full((1, 1, 5), 2.0, dtype=np.float64),
        dims=["longitude", "latitude", "variable"],
    )
    temperature = xr.DataArray(
        np.full((1, 1, n_time), 22.0, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    temperature_mean = xr.DataArray(
        np.full((1, 1, n_time), 22.0, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    latitudes = xr.DataArray(
        np.array([45.0], dtype=np.float64),
        dims=["latitude"],
    )
    carrying_capacity = xr.DataArray(
        np.full((1, 1, n_time), 1000.0, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    egg_activate = xr.DataArray(
        np.full((1, 1, n_time), 0.5, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    return (
        state,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
    )


def create_repeated_temperature_backend_inputs(n_days=3, time_step=2, n_lon=1, n_lat=1):
    state = xr.DataArray(
        np.full((n_lon, n_lat, 5), 2.0, dtype=np.float64),
        dims=["longitude", "latitude", "variable"],
    )
    temperature_mean = xr.DataArray(
        np.full((n_lon, n_lat, n_days), 22.0, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    temperature = xr.DataArray(
        np.repeat(temperature_mean.values, repeats=time_step, axis=2),
        dims=["longitude", "latitude", "time"],
    )
    latitudes = xr.DataArray(
        np.linspace(45.0, 46.0, n_lat, dtype=np.float64),
        dims=["latitude"],
    )
    carrying_capacity = xr.DataArray(
        np.full((n_lon, n_lat, n_days), 1000.0, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    egg_activate = xr.DataArray(
        np.full((n_lon, n_lat, n_days), 0.5, dtype=np.float64),
        dims=["longitude", "latitude", "time"],
    )
    return (
        state,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
    )


def test_solve_system_rejects_unknown_backend():
    inputs = create_stable_backend_inputs()

    with pytest.raises(ValueError, match="Unknown solver backend"):
        solve_system(*inputs, time_step=1.0, backend="unknown")


def test_legacy_optimized_matches_legacy_on_stable_fixture():
    inputs = create_stable_backend_inputs()

    legacy = solve_system(*inputs, time_step=1.0, backend="legacy")
    optimized = solve_system(*inputs, time_step=1.0, backend="legacy_optimized")

    np.testing.assert_allclose(optimized, legacy, rtol=1e-10, atol=1e-10)


def test_spatial_chunks_cover_grid_once():
    covered = np.zeros((5, 4), dtype=np.int64)

    for lon_slice, lat_slice in _spatial_chunks(5, 4, 2, 3):
        covered[lon_slice, lat_slice] += 1

    np.testing.assert_array_equal(covered, 1)


def test_scipy_chunked_backend_smoke_test_on_repeated_temperature_fixture():
    inputs = create_repeated_temperature_backend_inputs(
        n_days=2,
        time_step=2,
        n_lon=2,
        n_lat=2,
    )

    result = solve_system(
        *inputs,
        time_step=2,
        backend="scipy_chunked",
        scipy_method="RK45",
        scipy_rtol=1e-6,
        scipy_atol=1e-9,
        chunk_lon=1,
        chunk_lat=1,
    )

    assert result.shape == (2, 2, 5, 2)
    assert_all_finite(result)
    assert_no_negatives(result)


def test_scipy_chunked_backend_skips_invalid_carrying_capacity_cell():
    inputs = list(
        create_repeated_temperature_backend_inputs(
            n_days=2,
            time_step=2,
            n_lon=2,
            n_lat=1,
        )
    )
    carrying_capacity = inputs[4].copy()
    carrying_capacity.values[1, 0, :] = np.nan
    inputs[4] = carrying_capacity

    result = solve_system(
        *inputs,
        time_step=2,
        backend="scipy_chunked",
        chunk_lon=1,
        chunk_lat=1,
    )

    assert result.shape == (2, 1, 5, 2)
    assert_all_finite(result)
    assert_no_negatives(result)
    assert np.any(result[0, 0, :, :] > 0)
    np.testing.assert_array_equal(result[1, 0, :, :], 0.0)


def test_scipy_chunked_backend_returns_zero_when_all_capacity_is_invalid():
    """Test that a fully invalid capacity grid avoids SciPy integration."""
    inputs = list(create_repeated_temperature_backend_inputs(n_days=2))
    inputs[4].values[:] = np.nan

    result = solve_system(*inputs, time_step=2, backend="scipy_chunked")

    np.testing.assert_array_equal(result, 0.0)


def test_scipy_loaded_chunk_raises_when_solver_fails(monkeypatch):
    """Test that SciPy failures retain backend, method, and day context."""
    inputs = create_repeated_temperature_backend_inputs(n_days=1, time_step=1)

    class FailedSolution:
        success = False
        message = "integration failed"

    monkeypatch.setattr(
        Pmodel_ode, "solve_ivp", lambda *_args, **_kwargs: FailedSolution()
    )

    with pytest.raises(RuntimeError, match="test-backend/RK45 day 1 failed"):
        _solve_system_scipy_loaded_chunk(
            *inputs,
            time_step=1,
            scipy_method="RK45",
            scipy_rtol=1e-6,
            scipy_atol=1e-9,
            backend_label="test-backend",
        )


@pytest.mark.parametrize("backend", ["legacy", "legacy_optimized"])
def test_legacy_backends_reset_diapause_eggs_on_seasonal_day(monkeypatch, backend):
    """Test the day-200 seasonal reset without performing RK4 integration."""
    inputs = create_repeated_temperature_backend_inputs(n_days=200, time_step=1)
    monkeypatch.setattr(
        Pmodel_ode, "rk4_step", lambda _ode, _log_ode, state, _params, _steps: state
    )

    result = solve_system(*inputs, time_step=1, backend=backend)

    np.testing.assert_array_equal(result[..., 1, -1], 0.0)


def test_scipy_chunked_is_independent_of_chunk_partition_on_stable_fixture():
    inputs = create_repeated_temperature_backend_inputs(
        n_days=3,
        time_step=2,
        n_lon=2,
        n_lat=2,
    )

    daily = solve_system(
        *inputs,
        time_step=2,
        backend="scipy_chunked",
        scipy_method="DOP853",
        scipy_rtol=1e-9,
        scipy_atol=1e-12,
        chunk_lon=2,
        chunk_lat=2,
    )
    chunked = solve_system(
        *inputs,
        time_step=2,
        backend="scipy_chunked",
        scipy_method="DOP853",
        scipy_rtol=1e-9,
        scipy_atol=1e-12,
        chunk_lon=1,
        chunk_lat=1,
    )

    np.testing.assert_allclose(chunked, daily, rtol=1e-8, atol=1e-9)
