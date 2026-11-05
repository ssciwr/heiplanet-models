import logging

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_CARRYING_CAPACITY,
    CONSTANTS_MOSQUITO_E,
    CONSTANTS_MOSQUITO_I,
    CONSTANTS_MOSQUITO_J,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mosq_dev_j(temperature: np.ndarray) -> np.ndarray:
    """Calculate mosquito juvenile development rate as a function of temperature.

    Args:
        temperature (numpy.ndarray): Array of temperature values (°C).

    Returns:
        numpy.ndarray: Array of development rates, elementwise for each temperature.
    """

    CONST_1 = CONSTANTS_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_J["CONST_4"]

    t_out = CONST_1 - CONST_2 * temperature + CONST_3 * temperature**2
    t_out = CONST_4 / t_out
    return t_out


def mosq_dev_i(temperature: np.ndarray) -> np.ndarray:
    """Calculate mosquito development rate for the 'i' stage as a function of temperature.

    This function implements the Octave mosq_dev_i formula for the 'i' developmental
    stage of mosquitoes, using a temperature-dependent mathematical model.

    Args:
        temperature (numpy.ndarray): Array of temperature values in degrees Celsius.

    Returns:
        numpy.ndarray: Array of development rates, elementwise for each temperature value.
    """

    CONST_1 = CONSTANTS_MOSQUITO_I["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_I["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_I["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_I["CONST_4"]

    t_out = CONST_1 - CONST_2 * temperature + CONST_3 * temperature**2
    t_out = CONST_4 / t_out
    return t_out


def mosq_dev_e(temperature: np.ndarray) -> np.ndarray:
    """
    Calculates the mosquito egg development rate using the Brière model.

    Args:
        temperature (numpy.ndarray): Array of temperature values in degrees Celsius.

    Returns:
        numpy.ndarray: Array of development rates, elementwise for each temperature value.

    Raises:
        ValueError: If input is not a numpy ndarray.
    """
    q = CONSTANTS_MOSQUITO_E["q"]
    temperature_min = CONSTANTS_MOSQUITO_E["T0"]
    temperature_max = CONSTANTS_MOSQUITO_E["Tm"]

    # Briere function
    t_out = (
        q
        * temperature
        * (temperature - temperature_min)
        * ((temperature_max - temperature) ** (1 / 2))
    )

    return t_out


def _carrying_capacity_kernel(
    rainfall: np.ndarray,
    population: np.ndarray,
    alpha_rain: float,
    alpha_dens: float,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """Compute the carrying capacity recursion along the last (time) axis.

    Args:
        rainfall: Array of shape ``(..., time)``.
        population: Array broadcastable to ``rainfall.shape[:-1]``.

    Returns:
        float64 array with the same shape as ``rainfall``.
    """
    time_len = rainfall.shape[-1]
    carrying_capacity_array = np.zeros(rainfall.shape, dtype=np.float64)
    if time_len == 0:
        return carrying_capacity_array

    # Initial condition
    carrying_capacity_array[..., 0] = (
        alpha_rain * rainfall[..., 0] + alpha_dens * population
    )

    # Recursive computation (cannot be vectorized due to time dependency)
    for k in range(1, time_len):
        carrying_capacity_array[..., k] = (
            gamma * carrying_capacity_array[..., k - 1]
            + alpha_rain * rainfall[..., k]
            + alpha_dens * population
        )

    # Vectorized scaling factor for all time steps (except the first)
    k_idx = np.arange(1, time_len)
    factors = (1 - gamma) / (1 - gamma ** (k_idx + 1))
    carrying_capacity_array[..., 1:] *= factors

    # Final scaling
    carrying_capacity_array *= lambda_
    return carrying_capacity_array


def carrying_capacity(
    rainfall_data: xr.DataArray, population_data: xr.DataArray
) -> xr.DataArray:
    """
    Calculate the juvenile carrying capacity based on rainfall and population data.

    Args:
        rainfall_data: xarray DataArray with dimensions ('longitude', 'latitude', 'time').
        population_data: xarray DataArray with dimensions ('longitude', 'latitude') or ('longitude', 'latitude', 'time').

    Returns:
        xarray DataArray with the same shape as rainfall_data, representing carrying capacity.
    """
    ALPHA_RAIN = CONSTANTS_CARRYING_CAPACITY["ALPHA_RAIN"]
    ALPHA_DENS = CONSTANTS_CARRYING_CAPACITY["ALPHA_DENS"]
    GAMMA = CONSTANTS_CARRYING_CAPACITY["GAMMA"]
    LAMBDA = CONSTANTS_CARRYING_CAPACITY["LAMBDA"]

    # Only the first population time slice is used (constant within a run)
    if "time" in population_data.dims:
        population_data = population_data.isel(time=0)
    population_data = population_data.reset_coords(drop=True)
    logger.debug(f"Population shape: {population_data.shape}")

    # The recursion runs along time, so time must be a single dask chunk.
    # Spatial chunks are kept, which lets callers compute one region at a time.
    if rainfall_data.chunks is not None:
        rainfall_data = rainfall_data.chunk({"time": -1})

    result = xr.apply_ufunc(
        _carrying_capacity_kernel,
        rainfall_data,
        population_data,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        kwargs={
            "alpha_rain": ALPHA_RAIN,
            "alpha_dens": ALPHA_DENS,
            "gamma": GAMMA,
            "lambda_": LAMBDA,
        },
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    return result.transpose(*rainfall_data.dims).rename("carrying_capacity")
