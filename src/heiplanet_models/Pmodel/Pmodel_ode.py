"""
ODE solvers for the Aedes albopictus population model.

This module intentionally separates two classes of solver backends:

1. Legacy-compatible backends
   - Preserve the numerical policy of the original Octave/Matlab-style code.
   - Useful for migration validation and NetCDF-to-NetCDF comparison.

2. Modern SciPy backend
   - Use scipy.integrate.solve_ivp.
   - Useful for numerical experimentation and cleaner ODE formulations.
   - Not expected to be bitwise equivalent to the legacy RK4 implementation.

Important terminology
---------------------
The historical parameter name ``time_step`` is preserved in the public API for
backward compatibility. In this model it represents the number of sub-steps per
day, not the numerical integration step size dt. Internally it is normalized to
``steps_per_day`` where appropriate.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp

from heiplanet_models.Pmodel import Pmodel_initial
from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
)
from heiplanet_models.Pmodel.Pmodel_rates_development import (
    mosq_dev_i,
    mosq_dev_j,
)
from heiplanet_models.Pmodel.Pmodel_rates_mortality import (
    mosq_mort_a,
    mosq_mort_e,
    mosq_mort_j,
    mosq_surv_ed,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


MODEL_VARIABLES = 5  # eggs, diapause eggs, juveniles, immature adults, mature adults
DEFAULT_EGG_DEVELOPMENT_DAYS = 7.1
DEFAULT_EGG_DEVELOPMENT_RATE = 1.0 / DEFAULT_EGG_DEVELOPMENT_DAYS

VALID_BACKENDS = frozenset(
    {
        "legacy",
        "legacy_optimized",
        "scipy_chunked",
    }
)

SCIPY_METHODS = frozenset({"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"})

EXPECTED_SPATIOTEMPORAL_DIMS = ("longitude", "latitude", "time")
EXPECTED_LATITUDE_DIMS = ("latitude",)


# -----------------------------------------------------------------------------
# Configuration and shared input helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendPolicy:
    """Human-readable backend semantics for reporting and debugging."""

    solver: str
    forcing_resolution: str
    spatial_policy: str
    octave_compatible: bool
    notes: str


@dataclass(frozen=True)
class NormalizedSolverInputs:
    """Canonical arrays shared by solver implementations."""

    steps_per_day: int
    state: np.ndarray
    temperature: xr.DataArray | None
    temperature_mean: xr.DataArray
    latitudes: xr.DataArray
    carrying_capacity: np.ndarray
    egg_activate: np.ndarray


@dataclass
class SolverFallbackMetrics:
    """Count numerical recovery actions performed during one solver run."""

    derivative_substitutions: int = 0
    output_normalizations: int = 0


_FALLBACK_METRICS: ContextVar[SolverFallbackMetrics | None] = ContextVar(
    "pmodel_fallback_metrics", default=None
)


BACKEND_POLICIES: dict[str, BackendPolicy] = {
    "legacy": BackendPolicy(
        solver="manual RK4 with legacy log-space correction",
        forcing_resolution="sub-daily temperature, daily aggregate arrays where legacy code uses them",
        spatial_policy="full spatial array integrated together with NumPy vector operations",
        octave_compatible=True,
        notes="Reference backend for Octave/legacy output reproduction.",
    ),
    "legacy_optimized": BackendPolicy(
        solver="manual RK4 with legacy log-space correction",
        forcing_resolution="same numerical policy as legacy, but rates are precomputed",
        spatial_policy="full spatial array integrated together with NumPy vector operations",
        octave_compatible=True,
        notes="Preferred backend for legacy-equivalence validation.",
    ),
    "scipy_chunked": BackendPolicy(
        solver="solve_ivp on flattened chunk-level state vectors for each day",
        forcing_resolution="daily mean forcing",
        spatial_policy="spatial chunks, but cells inside a chunk remain solver-coupled",
        octave_compatible=False,
        notes="Only supported SciPy backend. Uses daily forcing and spatial chunks.",
    ),
}


def _validate_scipy_method(scipy_method: str) -> None:
    if scipy_method not in SCIPY_METHODS:
        methods = ", ".join(sorted(SCIPY_METHODS))
        raise ValueError(
            f"Unknown scipy_method '{scipy_method}'. Choose one of: {methods}."
        )


def _validate_steps_per_day(time_step: float) -> int:
    """Normalize the historical ``time_step`` value to integer sub-steps per day."""

    if not np.isfinite(time_step):
        raise ValueError(f"time_step must be finite; got {time_step!r}.")

    if not float(time_step).is_integer():
        raise ValueError(
            "time_step must be integer-like because this model uses it as the "
            f"number of sub-steps per day; got {time_step!r}."
        )

    steps_per_day = int(time_step)
    if steps_per_day <= 0:
        raise ValueError(f"time_step must be positive; got {time_step!r}.")

    return steps_per_day


def _ensure_3d_dataarray(
    value: xr.DataArray | np.ndarray,
    name: str,
    expected_dims: tuple[str, str, str] = EXPECTED_SPATIOTEMPORAL_DIMS,
) -> xr.DataArray:
    """Return a 3D DataArray with canonical dimension order.

    Existing DataArrays are validated and transposed to
    ``("longitude", "latitude", "time")``. Plain NumPy arrays are assumed to
    already follow that order.
    """

    if isinstance(value, xr.DataArray):
        missing_dims = set(expected_dims) - set(value.dims)
        if missing_dims:
            raise ValueError(
                f"{name} must contain dimensions {expected_dims}; "
                f"missing {sorted(missing_dims)} from {value.dims}."
            )
        if value.ndim != 3:
            raise ValueError(f"{name} must be 3D; got dims {value.dims}.")
        return value.transpose(*expected_dims)

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array.")

    return xr.DataArray(array, dims=expected_dims, name=name)


def _ensure_1d_dataarray(
    value: xr.DataArray | np.ndarray,
    name: str,
    expected_dims: tuple[str] = EXPECTED_LATITUDE_DIMS,
) -> xr.DataArray:
    """Return a 1D latitude DataArray."""

    if isinstance(value, xr.DataArray):
        missing_dims = set(expected_dims) - set(value.dims)
        if missing_dims:
            raise ValueError(
                f"{name} must contain dimension {expected_dims}; "
                f"missing {sorted(missing_dims)} from {value.dims}."
            )
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1D; got dims {value.dims}.")
        return value.transpose(*expected_dims)

    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    return xr.DataArray(array, dims=expected_dims, name=name)


def _as_float_array(value: xr.DataArray | np.ndarray) -> np.ndarray:
    if isinstance(value, xr.DataArray):
        value = value.values
    return np.asarray(value, dtype=np.float64)


def _output_state_slice(state: np.ndarray) -> np.ndarray:
    metrics = _FALLBACK_METRICS.get()
    if metrics is not None:
        metrics.output_normalizations += int(
            np.count_nonzero(~np.isfinite(state) | (state < 0.0))
        )
    output = np.maximum(state, 0.0)
    return np.where(np.isfinite(output), output, 0.0)


def _record_derivative_substitutions(mask: np.ndarray) -> None:
    """Record derivative values replaced by the established recovery policy."""

    metrics = _FALLBACK_METRICS.get()
    if metrics is not None:
        metrics.derivative_substitutions += int(np.count_nonzero(mask))


def _log_fallback_metrics(backend: str, metrics: SolverFallbackMetrics) -> None:
    """Report numerical recovery actions after a completed or failed solver run."""

    if metrics.derivative_substitutions or metrics.output_normalizations:
        logger.warning(
            "%s backend numerical recovery summary: %d derivative substitutions, "
            "%d output normalizations.",
            backend,
            metrics.derivative_substitutions,
            metrics.output_normalizations,
        )


def _day_index(step_number: int, steps_per_day: int) -> int:
    return math.ceil(step_number / steps_per_day) - 1


def _valid_capacity_cell_mask(carrying_capacity: np.ndarray) -> np.ndarray:
    """Return spatial cells with positive finite carrying capacity for all days."""
    return np.all(np.isfinite(carrying_capacity) & (carrying_capacity > 0.0), axis=2)


def _log_scipy_capacity_mask(backend: str, valid_cell_mask: np.ndarray) -> None:
    solved_cells = int(valid_cell_mask.sum())
    skipped_cells = int(valid_cell_mask.size - solved_cells)
    if skipped_cells:
        logger.warning(
            "%s backend skipped %d/%d spatial cells with nonpositive or non-finite "
            "carrying_capacity values. Skipped cells are written as zero output.",
            backend,
            skipped_cells,
            valid_cell_mask.size,
        )
    else:
        logger.info("%s backend solving all %d spatial cells.", backend, solved_cells)


def _validate_spatial_shapes(
    *,
    state: np.ndarray,
    temperature: xr.DataArray,
    temperature_mean: xr.DataArray,
    carrying_capacity: np.ndarray,
    egg_activate: np.ndarray,
) -> None:
    """Validate shared spatial dimensions and model variable count."""

    if state.ndim != 3:
        raise ValueError(
            "state must have shape (longitude, latitude, model_variable); "
            f"got shape {state.shape}."
        )

    if state.shape[2] != MODEL_VARIABLES:
        raise ValueError(
            f"state must contain {MODEL_VARIABLES} model variables in axis 2; "
            f"got shape {state.shape}."
        )

    spatial_shape = state.shape[:2]
    named_shapes = {
        "temperature": temperature.shape[:2],
        "temperature_mean": temperature_mean.shape[:2],
        "carrying_capacity": carrying_capacity.shape[:2],
        "egg_activate": egg_activate.shape[:2],
    }

    for name, shape in named_shapes.items():
        if shape != spatial_shape:
            raise ValueError(
                f"{name} spatial shape {shape} does not match state spatial "
                f"shape {spatial_shape}."
            )


def _validate_daily_time_lengths(
    *,
    temperature: xr.DataArray | None,
    temperature_mean: xr.DataArray,
    steps_per_day: int,
) -> None:
    """Validate consistency between sub-daily and daily time axes."""

    n_days = temperature_mean.shape[2]
    if n_days <= 0:
        raise ValueError("temperature_mean must contain at least one daily time step.")

    if temperature is None:
        return

    expected_substeps = n_days * steps_per_day
    if temperature.shape[2] != expected_substeps:
        raise ValueError(
            "Expected temperature time length to equal "
            "temperature_mean time length * time_step; got "
            f"{temperature.shape[2]} and {n_days} * {steps_per_day}."
        )


# -----------------------------------------------------------------------------
# Input normalization
# -----------------------------------------------------------------------------


def _normalize_solver_inputs(
    *,
    state: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray | None,
    temperature_mean: xr.DataArray | np.ndarray,
    latitudes: xr.DataArray | np.ndarray,
    carrying_capacity: xr.DataArray | np.ndarray,
    egg_activate: xr.DataArray | np.ndarray,
    time_step: float,
    require_temperature: bool,
    validate_temperature_spatial_shape: bool,
) -> NormalizedSolverInputs:
    """Normalize common solver inputs without changing numerical policy."""

    if require_temperature and temperature is None:
        raise ValueError("temperature is required for this solver backend.")

    steps_per_day = _validate_steps_per_day(time_step)
    temperature_da = (
        _ensure_3d_dataarray(temperature, "temperature")
        if temperature is not None
        else None
    )
    temperature_mean_da = _ensure_3d_dataarray(temperature_mean, "temperature_mean")
    latitudes_da = _ensure_1d_dataarray(latitudes, "latitudes")

    state_values = _as_float_array(state)
    carrying_capacity_values = _as_float_array(carrying_capacity)
    egg_activate_values = _as_float_array(egg_activate)

    if validate_temperature_spatial_shape:
        _validate_spatial_shapes(
            state=state_values,
            temperature=(
                temperature_da if temperature_da is not None else temperature_mean_da
            ),
            temperature_mean=temperature_mean_da,
            carrying_capacity=carrying_capacity_values,
            egg_activate=egg_activate_values,
        )

    _validate_daily_time_lengths(
        temperature=temperature_da,
        temperature_mean=temperature_mean_da,
        steps_per_day=steps_per_day,
    )

    return NormalizedSolverInputs(
        steps_per_day=steps_per_day,
        state=state_values,
        temperature=temperature_da,
        temperature_mean=temperature_mean_da,
        latitudes=latitudes_da,
        carrying_capacity=carrying_capacity_values,
        egg_activate=egg_activate_values,
    )


# -----------------------------------------------------------------------------
# Spatial chunk helpers
# -----------------------------------------------------------------------------


def _spatial_chunks(
    n_lon: int,
    n_lat: int,
    chunk_lon: int,
    chunk_lat: int,
) -> Iterable[tuple[slice, slice]]:
    if chunk_lon <= 0 or chunk_lat <= 0:
        raise ValueError("chunk_lon and chunk_lat must be positive integers.")

    for lon_start in range(0, n_lon, chunk_lon):
        lon_stop = min(lon_start + chunk_lon, n_lon)
        for lat_start in range(0, n_lat, chunk_lat):
            lat_stop = min(lat_start + chunk_lat, n_lat)
            yield slice(lon_start, lon_stop), slice(lat_start, lat_stop)


def _isel_spatial_chunk(
    value: xr.DataArray | np.ndarray,
    lon_slice: slice,
    lat_slice: slice,
) -> xr.DataArray | np.ndarray:
    if isinstance(value, xr.DataArray):
        return value.isel(longitude=lon_slice, latitude=lat_slice).load()
    return value[lon_slice, lat_slice, ...]


# -----------------------------------------------------------------------------
# Shared ODE equations
# -----------------------------------------------------------------------------


def albopictus_ode_system(
    state: np.ndarray,
    model_params: tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Compute derivatives of the Aedes albopictus population compartments.

    Compartments:
        0. non-diapause eggs
        1. diapause eggs
        2. juveniles
        3. immature adults
        4. mature adults
    """

    (
        t_idx,
        steps_per_day,
        carrying_capacity,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = model_params

    derivatives = np.zeros_like(state)

    derivatives[..., 0] = (
        state[..., 4] * birth * (1.0 - dia_lay)
        - (mort_e + water_hatch * dev_e) * state[..., 0]
    )

    derivatives[..., 1] = (
        state[..., 4] * birth * dia_lay - water_hatch * dia_hatch * state[..., 1]
    )

    derivatives[..., 2] = (
        water_hatch * dev_e * state[..., 0]
        + water_hatch * dia_hatch * ed_surv * state[..., 1]
        - (mort_j + dev_j) * state[..., 2]
        - (state[..., 2] ** 2) / carrying_capacity[..., t_idx - 1]
    )

    derivatives[..., 3] = 0.5 * dev_j * state[..., 2] - (mort_a + dev_i) * state[..., 3]

    derivatives[..., 4] = dev_i * state[..., 3] - mort_a * state[..., 4]

    nan_mask = np.isnan(-derivatives)
    _record_derivative_substitutions(nan_mask)
    derivatives[nan_mask] = -state[nan_mask] * steps_per_day

    return derivatives


def albopictus_log_ode_system(
    state: np.ndarray,
    model_params: tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Compute derivatives in the legacy log-correction space.

    This function intentionally supports complex-valued intermediate states
    because Octave's log(negative_real) enters complex arithmetic, while
    NumPy's log on real negative values would otherwise return NaN.
    """

    (
        t_idx,
        steps_per_day,
        carrying_capacity,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = model_params

    log_derivatives = np.zeros_like(state, dtype=np.result_type(state, np.complex128))

    log_derivatives[..., 0] = state[..., 4] * birth * (1.0 - dia_lay) / state[
        ..., 0
    ] - (mort_e + water_hatch * dev_e)

    log_derivatives[..., 1] = (
        state[..., 4] * birth * dia_lay / state[..., 1] - water_hatch * dia_hatch
    )

    log_derivatives[..., 2] = (
        water_hatch * dev_e * state[..., 0] / state[..., 2]
        + water_hatch * dia_hatch * ed_surv * state[..., 1] / state[..., 2]
        - (mort_j + dev_j)
        - state[..., 2] / carrying_capacity[..., t_idx - 1]
    )

    log_derivatives[..., 3] = 0.5 * dev_j * state[..., 2] / state[..., 3] - (
        mort_a + dev_i
    )

    log_derivatives[..., 4] = dev_i * state[..., 3] / state[..., 4] - mort_a

    nan_mask = np.isnan(-log_derivatives)
    _record_derivative_substitutions(nan_mask)
    log_derivatives[nan_mask] = -state[nan_mask] * steps_per_day

    return log_derivatives


def rk4_step(
    ode_func: Callable[[np.ndarray, tuple], np.ndarray],
    log_ode_func: Callable[[np.ndarray, tuple], np.ndarray],
    state: np.ndarray,
    model_params: tuple,
    steps_per_day: int | None = None,
    time_step: float | None = None,
) -> np.ndarray:
    """
    Perform one legacy-compatible Octave-style RK4 step.

    This intentionally reproduces the behavior of the legacy Octave RK4.m code,
    including its non-scalar any(...) condition for triggering the log-space
    correction.

    The correction logic is not a recommended modern numerical policy; it is
    preserved here to reproduce the legacy NetCDF output during migration.
    """

    if steps_per_day is None:
        if time_step is None:
            raise TypeError("rk4_step requires steps_per_day or time_step.")
        steps_per_day = _validate_steps_per_day(time_step)

    k1 = ode_func(state, model_params)
    k2_state = state + 0.5 * k1 / steps_per_day

    k2 = ode_func(k2_state, model_params)
    k3_state = state + 0.5 * k2 / steps_per_day

    k3 = ode_func(k3_state, model_params)
    k4_state = state + k3 / steps_per_day

    k4 = ode_func(k4_state, model_params)

    rk4_step_out_array = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / (steps_per_day * 6.0)

    mask_v1 = rk4_step_out_array < 0.0
    mask_k2 = k2_state < 0.0
    mask_k3 = k3_state < 0.0
    mask_k4 = k4_state < 0.0

    octave_condition_array = (
        np.any(mask_v1, axis=0)
        | np.any(mask_k2, axis=0)
        | np.any(mask_k3, axis=0)
        | np.any(mask_k4, axis=0)
    )

    octave_condition = bool(np.all(octave_condition_array))

    if octave_condition:
        correction_mask = mask_v1 | mask_k2 | mask_k3 | mask_k4

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v2 = np.log(state.astype(np.complex128))
            ft2 = log_ode_func(v2, model_params)
            v2 = v2 + ft2 / steps_per_day

            corrected_values = np.exp(v2[correction_mask])
            rk4_step_out_array[correction_mask] = np.real(corrected_values)

    return rk4_step_out_array


# -----------------------------------------------------------------------------
# Legacy backend
# -----------------------------------------------------------------------------


def solve_system_legacy(
    state: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray,
    temperature_mean: xr.DataArray | np.ndarray,
    latitudes: xr.DataArray | np.ndarray,
    carrying_capacity: xr.DataArray,
    egg_activate: xr.DataArray,
    time_step: float,
) -> np.ndarray:
    """
    Solve the mosquito population ODE system over time using the legacy policy.

    The loop indexing intentionally follows the legacy Octave convention.
    """

    steps_per_day = _validate_steps_per_day(time_step)

    if isinstance(state, xr.DataArray):
        state = state.values

    state = np.asarray(state, dtype=np.float64)

    diapause_lay = mosq_dia_lay(temperature_mean, latitudes)
    diapause_hatch = mosq_dia_hatch(temperature_mean, latitudes)
    ed_survival = mosq_surv_ed(temperature, steps_per_day)

    v_out = Pmodel_initial.create_model_output(
        dataset_base_shape=temperature.shape,
        initial_conditions_shape=state.shape,
        time_step=steps_per_day,
    )

    for t in range(temperature.shape[2]):
        step_number = t + 1
        idx_time = _day_index(step_number, steps_per_day)

        temperature_slice = temperature[:, :, t]

        birth = mosq_birth(temperature_slice)
        dev_j = mosq_dev_j(temperature_slice)
        dev_i = mosq_dev_i(temperature_slice)
        dev_e = DEFAULT_EGG_DEVELOPMENT_RATE

        dia_lay = diapause_lay.values[:, :, idx_time]
        dia_hatch = diapause_hatch.values[:, :, idx_time]
        ed_surv = ed_survival[:, :, t]
        water_hatch = egg_activate.values[:, :, idx_time]

        mort_e = mosq_mort_e(temperature_slice)
        mort_j = mosq_mort_j(temperature_slice)

        temperature_mean_slice = temperature_mean[:, :, idx_time]
        mort_a = mosq_mort_a(temperature_mean_slice)

        model_params = (
            idx_time + 1,
            steps_per_day,
            carrying_capacity.values,
            birth.values,
            dia_lay,
            dia_hatch,
            mort_e.values,
            mort_j.values,
            mort_a.values,
            ed_surv.values,
            dev_j.values,
            dev_i.values,
            dev_e,
            water_hatch,
        )

        state = rk4_step(
            albopictus_ode_system,
            albopictus_log_ode_system,
            state,
            model_params,
            steps_per_day,
        )

        if step_number % steps_per_day == 0:
            day_number = int(step_number // steps_per_day)

            if day_number % 365 == 200:
                state[..., 1] = 0.0

            output_index = int(day_number - 1)

            v_out[..., output_index] = _output_state_slice(state)

    return v_out


# -----------------------------------------------------------------------------
# Legacy optimized implementation
# -----------------------------------------------------------------------------


def solve_system_legacy_optimized(
    state: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray,
    temperature_mean: xr.DataArray | np.ndarray,
    latitudes: xr.DataArray | np.ndarray,
    carrying_capacity: xr.DataArray | np.ndarray,
    egg_activate: xr.DataArray | np.ndarray,
    time_step: float,
) -> np.ndarray:
    """Solve with the legacy RK4 method using precomputed NumPy rate arrays."""

    inputs = _normalize_solver_inputs(
        state=state,
        temperature=temperature,
        temperature_mean=temperature_mean,
        latitudes=latitudes,
        carrying_capacity=carrying_capacity,
        egg_activate=egg_activate,
        time_step=time_step,
        require_temperature=True,
        validate_temperature_spatial_shape=True,
    )

    steps_per_day = inputs.steps_per_day
    state = inputs.state
    temperature_da = inputs.temperature
    temperature_mean_da = inputs.temperature_mean
    latitudes_da = inputs.latitudes
    carrying_capacity_values = inputs.carrying_capacity
    egg_activate_values = inputs.egg_activate

    diapause_lay = mosq_dia_lay(temperature_mean_da, latitudes_da).values
    diapause_hatch = mosq_dia_hatch(temperature_mean_da, latitudes_da).values
    ed_survival = mosq_surv_ed(temperature_da, steps_per_day).values

    birth = mosq_birth(temperature_da).values
    mort_e = mosq_mort_e(temperature_da).values
    mort_j = mosq_mort_j(temperature_da).values
    dev_j = np.asarray(mosq_dev_j(temperature_da), dtype=np.float64)
    dev_i = np.asarray(mosq_dev_i(temperature_da), dtype=np.float64)
    mort_a = mosq_mort_a(temperature_mean_da).values
    dev_e = DEFAULT_EGG_DEVELOPMENT_RATE

    v_out = Pmodel_initial.create_model_output(
        dataset_base_shape=temperature_da.shape,
        initial_conditions_shape=state.shape,
        time_step=steps_per_day,
    )

    for t in range(temperature_da.shape[2]):
        step_number = t + 1
        idx_time = _day_index(step_number, steps_per_day)

        model_params = (
            idx_time + 1,
            steps_per_day,
            carrying_capacity_values,
            birth[:, :, t],
            diapause_lay[:, :, idx_time],
            diapause_hatch[:, :, idx_time],
            mort_e[:, :, t],
            mort_j[:, :, t],
            mort_a[:, :, idx_time],
            ed_survival[:, :, t],
            dev_j[:, :, t],
            dev_i[:, :, t],
            dev_e,
            egg_activate_values[:, :, idx_time],
        )

        state = rk4_step(
            albopictus_ode_system,
            albopictus_log_ode_system,
            state,
            model_params,
            steps_per_day,
        )

        if step_number % steps_per_day == 0:
            day_number = int(step_number // steps_per_day)

            if day_number % 365 == 200:
                state[..., 1] = 0.0

            v_out[..., int(day_number - 1)] = _output_state_slice(state)

    return v_out


# -----------------------------------------------------------------------------
# SciPy backend
# -----------------------------------------------------------------------------


def _scipy_rhs(
    _t: float,
    y: np.ndarray,
    rates: np.ndarray,
    carrying_capacity: np.ndarray,
    dev_e: float,
    n_cells: int,
) -> np.ndarray:
    state = y.reshape(n_cells, MODEL_VARIABLES)

    eggs = state[:, 0]
    diapause_eggs = state[:, 1]
    juveniles = state[:, 2]
    immature_adults = state[:, 3]
    mature_adults = state[:, 4]

    birth = rates[:, 0]
    mort_e = rates[:, 1]
    mort_j = rates[:, 2]
    dev_j = rates[:, 3]
    dev_i = rates[:, 4]
    ed_surv = rates[:, 5]
    mort_a = rates[:, 6]
    dia_lay = rates[:, 7]
    dia_hatch = rates[:, 8]
    water_hatch = rates[:, 9]

    derivatives = np.empty_like(state)
    derivatives[:, 0] = (
        mature_adults * birth * (1.0 - dia_lay) - (mort_e + water_hatch * dev_e) * eggs
    )
    derivatives[:, 1] = (
        mature_adults * birth * dia_lay - water_hatch * dia_hatch * diapause_eggs
    )
    derivatives[:, 2] = (
        water_hatch * dev_e * eggs
        + water_hatch * dia_hatch * ed_surv * diapause_eggs
        - (mort_j + dev_j) * juveniles
        - juveniles * juveniles / carrying_capacity
    )
    derivatives[:, 3] = 0.5 * dev_j * juveniles - (mort_a + dev_i) * immature_adults
    derivatives[:, 4] = dev_i * immature_adults - mort_a * mature_adults

    invalid = ~np.isfinite(derivatives)
    _record_derivative_substitutions(invalid)
    derivatives[invalid] = -state[invalid]
    return derivatives.ravel()


def solve_system_scipy_chunked(
    state: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray | None,
    temperature_mean: xr.DataArray | np.ndarray,
    latitudes: xr.DataArray | np.ndarray,
    carrying_capacity: xr.DataArray | np.ndarray,
    egg_activate: xr.DataArray | np.ndarray,
    time_step: float,
    scipy_method: str = "RK45",
    scipy_rtol: float = 1e-6,
    scipy_atol: float = 1e-9,
    chunk_lon: int = 20,
    chunk_lat: int = 20,
) -> np.ndarray:
    """Solve with ``solve_ivp`` over spatial chunks using daily forcing."""

    _validate_scipy_method(scipy_method)
    steps_per_day = _validate_steps_per_day(time_step)

    state_da = state if isinstance(state, xr.DataArray) else np.asarray(state)
    temperature_da = (
        _ensure_3d_dataarray(temperature, "temperature")
        if temperature is not None
        else None
    )
    temperature_mean_da = _ensure_3d_dataarray(temperature_mean, "temperature_mean")
    latitudes_da = _ensure_1d_dataarray(latitudes, "latitudes")
    carrying_capacity_da = _ensure_3d_dataarray(carrying_capacity, "carrying_capacity")
    egg_activate_da = _ensure_3d_dataarray(egg_activate, "egg_activate")

    _validate_daily_time_lengths(
        temperature=temperature_da,
        temperature_mean=temperature_mean_da,
        steps_per_day=steps_per_day,
    )

    n_lon, n_lat = state_da.shape[:2]
    n_days = temperature_mean_da.shape[2]

    output = np.zeros((n_lon, n_lat, MODEL_VARIABLES, n_days), dtype=np.float64)
    solved_cells_total = 0
    skipped_cells_total = 0
    chunks = list(_spatial_chunks(n_lon, n_lat, chunk_lon, chunk_lat))
    backend_start = time.perf_counter()

    logger.info(
        "scipy_chunked backend starting %d chunks "
        "(chunk_lon=%d, chunk_lat=%d, n_days=%d).",
        len(chunks),
        chunk_lon,
        chunk_lat,
        n_days,
    )

    for chunk_index, (lon_slice, lat_slice) in enumerate(chunks, start=1):
        chunk_start = time.perf_counter()
        state_chunk = _isel_spatial_chunk(state_da, lon_slice, lat_slice)
        temperature_chunk = (
            _isel_spatial_chunk(temperature_da, lon_slice, lat_slice)
            if temperature_da is not None
            else None
        )
        temperature_mean_chunk = _isel_spatial_chunk(
            temperature_mean_da, lon_slice, lat_slice
        )
        carrying_capacity_chunk = _isel_spatial_chunk(
            carrying_capacity_da, lon_slice, lat_slice
        )
        egg_activate_chunk = _isel_spatial_chunk(egg_activate_da, lon_slice, lat_slice)

        capacity_values = _as_float_array(carrying_capacity_chunk)
        valid_mask = _valid_capacity_cell_mask(capacity_values)
        solved_cells = int(valid_mask.sum())
        skipped_cells = int(valid_mask.size - solved_cells)

        logger.info(
            "START scipy_chunked chunk %d/%d lon=%d:%d lat=%d:%d "
            "solved_cells=%d skipped_cells=%d",
            chunk_index,
            len(chunks),
            lon_slice.start,
            lon_slice.stop,
            lat_slice.start,
            lat_slice.stop,
            solved_cells,
            skipped_cells,
        )

        chunk_output = _solve_system_scipy_loaded_chunk(
            state=state_chunk,
            temperature=temperature_chunk,
            temperature_mean=temperature_mean_chunk,
            latitudes=latitudes_da.isel(latitude=lat_slice).load(),
            carrying_capacity=carrying_capacity_chunk,
            egg_activate=egg_activate_chunk,
            time_step=steps_per_day,
            scipy_method=scipy_method,
            scipy_rtol=scipy_rtol,
            scipy_atol=scipy_atol,
            backend_label=(
                "scipy_chunked"
                f"[lon={lon_slice.start}:{lon_slice.stop},"
                f"lat={lat_slice.start}:{lat_slice.stop}]"
            ),
            log_capacity_mask=False,
        )

        output[lon_slice, lat_slice, :, :] = chunk_output
        solved_cells_total += solved_cells
        skipped_cells_total += skipped_cells

        logger.info(
            "END scipy_chunked chunk %d/%d lon=%d:%d lat=%d:%d in %.2fs "
            "(elapsed %.2fs)",
            chunk_index,
            len(chunks),
            lon_slice.start,
            lon_slice.stop,
            lat_slice.start,
            lat_slice.stop,
            time.perf_counter() - chunk_start,
            time.perf_counter() - backend_start,
        )

    total_cells = solved_cells_total + skipped_cells_total
    if skipped_cells_total:
        logger.warning(
            "scipy_chunked backend skipped %d/%d spatial cells with "
            "nonpositive or non-finite carrying_capacity values. Skipped cells "
            "are written as zero output.",
            skipped_cells_total,
            total_cells,
        )
    else:
        logger.info("scipy_chunked backend solving all %d spatial cells.", total_cells)

    logger.info(
        "scipy_chunked backend finished %d chunks in %.2fs.",
        len(chunks),
        time.perf_counter() - backend_start,
    )

    return output


def _solve_system_scipy_loaded_chunk(
    state: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray | None,
    temperature_mean: xr.DataArray | np.ndarray,
    latitudes: xr.DataArray | np.ndarray,
    carrying_capacity: xr.DataArray | np.ndarray,
    egg_activate: xr.DataArray | np.ndarray,
    time_step: float,
    scipy_method: str,
    scipy_rtol: float,
    scipy_atol: float,
    backend_label: str,
    log_capacity_mask: bool = True,
) -> np.ndarray:
    """Solve one already-loaded spatial chunk with the SciPy daily policy."""

    _validate_scipy_method(scipy_method)
    inputs = _normalize_solver_inputs(
        state=state,
        temperature=temperature,
        temperature_mean=temperature_mean,
        latitudes=latitudes,
        carrying_capacity=carrying_capacity,
        egg_activate=egg_activate,
        time_step=time_step,
        require_temperature=False,
        validate_temperature_spatial_shape=True,
    )

    steps_per_day = inputs.steps_per_day
    temperature_mean_da = inputs.temperature_mean
    latitudes_da = inputs.latitudes
    state_values = inputs.state
    carrying_capacity_values = inputs.carrying_capacity
    egg_activate_values = inputs.egg_activate

    n_days = temperature_mean_da.shape[2]
    n_lon, n_lat = state_values.shape[:2]
    valid_cell_mask = _valid_capacity_cell_mask(carrying_capacity_values)
    if log_capacity_mask:
        _log_scipy_capacity_mask(backend_label, valid_cell_mask)

    n_cells = int(valid_cell_mask.sum())
    dev_e = DEFAULT_EGG_DEVELOPMENT_RATE

    output = np.zeros((n_lon * n_lat, MODEL_VARIABLES, n_days), dtype=np.float64)
    if n_cells == 0:
        return output.reshape(n_lon, n_lat, MODEL_VARIABLES, n_days)

    valid_flat = valid_cell_mask.reshape(n_lon * n_lat)

    birth = mosq_birth(temperature_mean_da).values.reshape(n_lon * n_lat, n_days)[
        valid_flat
    ]
    mort_e = mosq_mort_e(temperature_mean_da).values.reshape(n_lon * n_lat, n_days)[
        valid_flat
    ]
    mort_j = mosq_mort_j(temperature_mean_da).values.reshape(n_lon * n_lat, n_days)[
        valid_flat
    ]
    dev_j = np.asarray(mosq_dev_j(temperature_mean_da), dtype=np.float64).reshape(
        n_lon * n_lat, n_days
    )[valid_flat]
    dev_i = np.asarray(mosq_dev_i(temperature_mean_da), dtype=np.float64).reshape(
        n_lon * n_lat, n_days
    )[valid_flat]

    # Preserve the explicit time_step argument. If mosq_surv_ed supports only one
    # argument in your local implementation, adapt this line accordingly.
    ed_survival = mosq_surv_ed(temperature_mean_da, steps_per_day).values.reshape(
        n_lon * n_lat, n_days
    )[valid_flat]

    mort_a = mosq_mort_a(temperature_mean_da).values.reshape(n_lon * n_lat, n_days)[
        valid_flat
    ]
    diapause_lay = mosq_dia_lay(temperature_mean_da, latitudes_da).values.reshape(
        n_lon * n_lat, n_days
    )[valid_flat]
    diapause_hatch = mosq_dia_hatch(temperature_mean_da, latitudes_da).values.reshape(
        n_lon * n_lat, n_days
    )[valid_flat]
    carrying_capacity_flat = carrying_capacity_values.reshape(n_lon * n_lat, n_days)[
        valid_flat
    ]
    egg_activate_flat = egg_activate_values.reshape(n_lon * n_lat, n_days)[valid_flat]

    y = np.ascontiguousarray(
        state_values.reshape(n_lon * n_lat, MODEL_VARIABLES)[valid_flat]
    ).ravel()
    rates = np.empty((n_cells, 10), dtype=np.float64)

    for day in range(n_days):
        rates[:, 0] = birth[:, day]
        rates[:, 1] = mort_e[:, day]
        rates[:, 2] = mort_j[:, day]
        rates[:, 3] = dev_j[:, day]
        rates[:, 4] = dev_i[:, day]
        rates[:, 5] = ed_survival[:, day]
        rates[:, 6] = mort_a[:, day]
        rates[:, 7] = diapause_lay[:, day]
        rates[:, 8] = diapause_hatch[:, day]
        rates[:, 9] = egg_activate_flat[:, day]

        sol = solve_ivp(
            _scipy_rhs,
            (0.0, 1.0),
            y,
            args=(rates, carrying_capacity_flat[:, day], dev_e, n_cells),
            method=scipy_method,
            rtol=scipy_rtol,
            atol=scipy_atol,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(
                f"{backend_label}/{scipy_method} day {day + 1} failed: {sol.message}"
            )

        state_day = sol.y[:, -1].reshape(n_cells, MODEL_VARIABLES)

        day_number = day + 1
        if day_number % 365 == 200:  # 365 days in a year, 200 seasonal constant
            state_day[:, 1] = 0.0

        # Keep the internal state biologically meaningful for the next daily solve.
        state_day = _output_state_slice(state_day)
        y = np.ascontiguousarray(state_day.ravel(), dtype=np.float64)
        output[valid_flat, :, day] = state_day

    return output.reshape(n_lon, n_lat, MODEL_VARIABLES, n_days)


# -----------------------------------------------------------------------------
# Public solver dispatcher
# -----------------------------------------------------------------------------


def solve_system(
    state: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray | None,
    temperature_mean: xr.DataArray | np.ndarray,
    latitudes: xr.DataArray | np.ndarray,
    carrying_capacity: xr.DataArray | np.ndarray,
    egg_activate: xr.DataArray | np.ndarray,
    time_step: float,
    backend: str = "scipy_chunked",
    scipy_method: str = "RK45",
    scipy_rtol: float = 1e-6,
    scipy_atol: float = 1e-9,
    chunk_lon: int = 20,
    chunk_lat: int = 20,
) -> np.ndarray:
    """Solve the mosquito population ODE system with a selected backend.

    Parameters
    ----------
    time_step:
        Historical name preserved for API compatibility. It represents the
        number of sub-steps per day, not dt.

    Notes
    -----
    Existing numerical recovery behavior is preserved: non-finite derivatives
    are replaced by decay terms, and negative or non-finite daily outputs are
    normalized to zero. A warning summarizes either action for each run.
    """

    metrics = SolverFallbackMetrics()
    metrics_token = _FALLBACK_METRICS.set(metrics)
    try:
        if backend not in VALID_BACKENDS:
            backends = ", ".join(sorted(VALID_BACKENDS))
            raise ValueError(
                f"Unknown solver backend '{backend}'. Choose one of: {backends}."
            )

        legacy_backends = {
            "legacy": solve_system_legacy,
            "legacy_optimized": solve_system_legacy_optimized,
        }
        if backend in legacy_backends:
            return legacy_backends[backend](
                state=state,
                temperature=temperature,
                temperature_mean=temperature_mean,
                latitudes=latitudes,
                carrying_capacity=carrying_capacity,
                egg_activate=egg_activate,
                time_step=time_step,
            )

        if backend == "scipy_chunked":
            # SciPy-specific options are intentionally used only by this backend.
            return solve_system_scipy_chunked(
                state=state,
                temperature=temperature,
                temperature_mean=temperature_mean,
                latitudes=latitudes,
                carrying_capacity=carrying_capacity,
                egg_activate=egg_activate,
                time_step=time_step,
                scipy_method=scipy_method,
                scipy_rtol=scipy_rtol,
                scipy_atol=scipy_atol,
                chunk_lon=chunk_lon,
                chunk_lat=chunk_lat,
            )

        raise AssertionError(
            f"Unhandled validated backend: {backend}"
        )  # pragma: no cover
    finally:
        _log_fallback_metrics(backend, metrics)
        _FALLBACK_METRICS.reset(metrics_token)
