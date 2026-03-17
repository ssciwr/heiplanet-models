import logging

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MOSQUITO_J,
    CONSTANTS_MOSQUITO_I,
    CONSTANTS_MOSQUITO_E,
    CONSTANTS_CARRYING_CAPACITY,
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

    # Ensure population_data has a time dimension for broadcasting
    if "time" not in population_data.dims:
        population_data = population_data.expand_dims(time=rainfall_data.time)

    # Prepare output array
    rainfall_np = rainfall_data.values  # shape: (longitude, latitude, time)
    logger.debug(f"Rainfall np shape: {rainfall_np.shape}")

    population_np = population_data.isel(time=0).values  # shape: (longitude, latitude)
    logger.debug(f"Population np shape: {population_np.shape}")

    lon_len, lat_len, time_len = rainfall_np.shape
    carrying_capacity_array = np.zeros((lon_len, lat_len, time_len), dtype=np.float64)

    logger.debug(f"Shape carrying_capacity_array: {carrying_capacity_array.shape}")

    # Initial condition
    carrying_capacity_array[..., 0] = (
        ALPHA_RAIN * rainfall_np[..., 0] + ALPHA_DENS * population_np
    )

    # Recursive computation (cannot be vectorized due to time dependency)
    for k in range(1, time_len):
        carrying_capacity_array[..., k] = (
            GAMMA * carrying_capacity_array[..., k - 1]
            + ALPHA_RAIN * rainfall_np[..., k]
            + ALPHA_DENS * population_np
        )

    # Vectorized scaling factor for all time steps (except the first)
    k_idx = np.arange(1, time_len)
    factors = (1 - GAMMA) / (1 - GAMMA ** (k_idx + 1))
    carrying_capacity_array[..., 1:] = factors * carrying_capacity_array[..., 1:]

    logger.debug(f"shape A after scaling: {carrying_capacity_array.shape}")
    logger.debug(f"shape rainfall_data: {rainfall_data.shape}")

    # Final scaling
    result = (
        xr.DataArray(
            carrying_capacity_array,
            coords=rainfall_data.coords,
            dims=rainfall_data.dims,
            name="carrying_capacity",
        )
        * LAMBDA
    )

    return result
