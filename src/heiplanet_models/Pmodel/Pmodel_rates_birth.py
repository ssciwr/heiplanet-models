"""Calculates biological rates for the P-model mosquito population simulation.

This module provides a collection of functions to compute key biological and
environmental rates that drive the mosquito population dynamics in the P-model.
It includes calculations for birth rates, diapause (a state of suspended
development) for both egg hatching and laying, and water-dependent hatching
probabilities.

The functions rely on environmental inputs such as temperature, rainfall,
latitude, and the day of the year to model the mosquito life cycle. It also
contains helper functions for astronomical calculations, such as determining
the Earth's revolution angle, the sun's declination, and the length of daylight,
which are critical for modeling photoperiod-sensitive processes like diapause.

Key functions include:
- mosq_birth: Calculates temperature-dependent birth rates.
- mosq_dia_hatch: Determines the rate of diapause termination for hatching.
- mosq_dia_lay: Models the induction of diapause for egg-laying.
- water_hatching: Computes hatching probability based on rainfall and human
  population density.
"""

import logging

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MOSQUITO_BIRTH,
    CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING,
    CONSTANTS_MOSQUITO_DIAPAUSE_LAY,
    CONSTANTS_WATER_HATCHING,
    DAYS_YEAR,
    HALF_DAYS_YEAR,
)
from heiplanet_models.utils import validate_spatial_alignment
from heiplanet_models.utils_pmodel.solar_calculations import (
    daylight_forsythe,
    declination_angle,
    revolution_angle,
)

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mosq_birth(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the mosquito birth rate based on temperature.

    Args:
        temperature (xr.DataArray): An xarray DataArray of temperature values.

    Returns:
        xr.DataArray: An xarray DataArray of the same shape as the input, containing
            the calculated mosquito birth rates.
    """
    CONST_1 = CONSTANTS_MOSQUITO_BIRTH["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_BIRTH["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_BIRTH["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_BIRTH["CONST_4"]
    CONST_5 = CONSTANTS_MOSQUITO_BIRTH["CONST_5"]
    CONST_6 = CONSTANTS_MOSQUITO_BIRTH["CONST_6"]

    temp_values = temperature.values
    out = temp_values.copy()
    mask = out < CONST_1
    out[mask] = (
        CONST_2
        * np.exp(CONST_3 * ((out[mask] - CONST_4) / CONST_5) ** 2)
        * (CONST_1 - out[mask]) ** CONST_6
    )
    out[~mask] = 0

    # Return as xarray.DataArray with same dims/coords as input
    return xr.DataArray(out, dims=temperature.dims, coords=temperature.coords)


def mosq_dia_hatch(temperature: xr.DataArray, latitude: xr.DataArray) -> xr.DataArray:
    """Calculates mosquito diapause hatching based on temperature and daylight.

    Hatching is predicted when the rolling average temperature is above a
    critical threshold (`CTT`) and the daily photoperiod is longer than a
    critical length (`CPP`).

    Args:
        temperature (xr.DataArray): 3D DataArray of temperature values
            (longitude, latitude, time).
        latitude (xr.DataArray): 1D DataArray of latitude values.

    Returns:
        xr.DataArray: A 3D DataArray indicating where hatching occurs. Values
            are a constant ratio where conditions are met, and 0 otherwise.

    Raises:
        ValueError: If input arrays have incompatible dimensions.
    """

    PERIOD = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["PERIOD"]
    CPP = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CPP"]
    CTT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CTT"]
    RATIO_DIA_HATCH = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["RATIO_DIA_HATCH"]
    DAYLENGTH_COEFFICIENT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING[
        "DAYLENGTH_COEFFICIENT"
    ]

    # Validate input dimensions
    if temperature.ndim != 3:
        raise ValueError("Temperature array must be 3D (longitude, latitude, time).")
    if latitude.ndim != 1:
        raise ValueError("Latitude array must be 1D.")

    n_longitude, n_latitude, n_time = temperature.shape

    # Calculate mean temperature of the last 'PERIOD' days and compare to CTT
    # out = temperature.copy().values
    # for k in range(n_time - 1, PERIOD - 2, -1):
    #    out[:, :, k] = np.mean(out[:, :, (k - PERIOD + 1) : (k + 1)], axis=2)

    # Efficient rolling mean using xarray
    # temperature_rolling = temperature.rolling(time=PERIOD, min_periods=PERIOD).mean().fillna(0)
    # out = temp_rolling.values

    temperature_rolling = temperature.rolling(time=PERIOD, min_periods=PERIOD).mean()
    hatch_signal = temperature_rolling.values
    hatch_signal[:, :, : PERIOD - 1] = temperature.values[:, :, : PERIOD - 1]

    # Mask values below critical temperature threshold
    hatch_signal[hatch_signal < CTT] = 0

    days = np.arange(1, n_time + 1)
    theta = revolution_angle(days)
    phi = declination_angle(theta)  # shape: (n_time,)

    latitude_degrees = latitude.values

    for k in range(n_latitude):
        lat = latitude_degrees[k]

        daylight = daylight_forsythe(
            latitude=lat,
            declination_angle=phi,
            daylight_coefficient=DAYLENGTH_COEFFICIENT,
        )

        # Replicate daylight shape to (x, z)
        daylight_matrix = np.tile(daylight, (n_longitude, 1))

        # Mask where daylight < CPP
        latitude_mask = hatch_signal[:, k, :]
        latitude_mask[daylight_matrix < CPP] = 0
        hatch_signal[:, k, :] = latitude_mask

    # Set NaNs to 0
    hatch_signal = np.nan_to_num(hatch_signal, nan=0.0)

    # Binarize and scale
    hatch_signal[hatch_signal > 0] = RATIO_DIA_HATCH

    # Return as xarray.DataArray with same dims/coords as input
    return xr.DataArray(hatch_signal, dims=temperature.dims, coords=temperature.coords)


def mosq_dia_lay(temperature: xr.DataArray, latitude: xr.DataArray) -> xr.DataArray:
    """
    Calculate mosquito diapause induction based on daylight and latitude.

    Args:
        temperature (xr.DataArray): Temperature data with dimensions (longitude, latitude, time).
        latitude (xr.DataArray): Latitude values (1D array).

    Returns:
        xr.DataArray: Diapause induction array with same dims/coords as input.

    Raises:
        ValueError: If input dimensions are incompatible.
    """

    RATIO_DIA_LAY = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["RATIO_DIA_LAY"]
    CONST_1 = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["CONST_2"]
    DAYLENGTH_COEFFICIENT = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["DAYLENGTH_COEFFICIENT"]

    # Validate input dimensions
    if temperature.ndim != 3:
        raise ValueError("Temperature array must be 3D (longitude, latitude, time).")
    if latitude.ndim != 1:
        raise ValueError("Latitude array must be 1D.")

    n_longitude, n_latitude, n_time = temperature.shape

    # Calculate declination angle for each day
    days = np.arange(1, n_time + 1)
    theta = revolution_angle(days)
    phi = declination_angle(theta)  # shape: (n_time,)

    # Prepare output array
    out = temperature.copy().values

    # Conversion from degrees to radians
    latitude_degrees = latitude.values

    for k in range(n_latitude):
        lat = latitude_degrees[k]

        daylight = daylight_forsythe(
            latitude=lat,
            declination_angle=phi,
            daylight_coefficient=DAYLENGTH_COEFFICIENT,
        )
        # Set non-real daylight values to zero (e.g., above Arctic Circle)
        daylight[~np.isreal(daylight)] = 0
        CPP = CONST_1 + CONST_2 * lat
        daylight[daylight > CPP] = 0

        # Assign daylight to all longitudes for this latitude
        out[:, k, :] = np.tile(daylight, (n_longitude, 1))

    # No diapause induction in the first half of each year
    n_years = n_time // DAYS_YEAR
    for k in range(n_years):
        start = k * DAYS_YEAR
        end = start + HALF_DAYS_YEAR
        out[:, :, start:end] = 0

    out[out > 0] = 1
    out = out * RATIO_DIA_LAY

    return xr.DataArray(out, dims=temperature.dims, coords=temperature.coords)


def water_hatching(
    rainfall_data: xr.DataArray,
    population_data: xr.DataArray,
) -> xr.DataArray:
    """Compute the water hatching probability or rate as a function of rainfall and population data.

    This function calculates a hatching rate based on two factors: a rainfall-dependent term that
    peaks at an optimal rainfall level, and a population-density-dependent term modeling artificial
    breeding sites. The two factors are combined in a weighted average.

    Ref. Equation: # 13
    Name: Hatching fraction depending in human density and rainfall

    Args:
        rainfall_data (xr.DataArray): A 3D DataArray of rainfall data with
            dimensions ('time', 'latitude', 'longitude').
        population_data (xr.DataArray): A DataArray of population density.
            Can be 3D ('time', 'latitude', 'longitude') or 2D
            ('latitude', 'longitude').

    Returns:
        xr.DataArray: A DataArray of the combined hatching probability, with
            the same shape as `rainfall_data`.
    """

    E_OPT = CONSTANTS_WATER_HATCHING["E_OPT"]
    E_VAR = CONSTANTS_WATER_HATCHING["E_VAR"]
    E_0 = CONSTANTS_WATER_HATCHING["E_0"]
    E_RAT = CONSTANTS_WATER_HATCHING["E_RAT"]
    E_DENS = CONSTANTS_WATER_HATCHING["E_DENS"]
    E_FAC = CONSTANTS_WATER_HATCHING["E_FAC"]

    # Validate input alignment first
    validate_spatial_alignment(rainfall_data, population_data)

    # Handle missing 'time' dimension in population_data
    if "time" not in population_data.dims:
        population_data = population_data.expand_dims(time=rainfall_data.time)

    population_hatch = E_DENS / (E_DENS + np.exp(-E_FAC * population_data))
    logger.debug(f"Population shape: {population_hatch.shape}")

    exp_term = np.exp(-E_VAR * (rainfall_data - E_OPT) ** 2)
    rainfall_hatch = (1 + E_0) * exp_term / (exp_term + E_0)

    logger.debug(f"Dimension rainfall_hatch: {rainfall_hatch.shape}")
    logger.debug(f"Dimension rainfall_hatch: {rainfall_hatch.dims}")
    logger.debug(f"Dimension population_hatch: {population_hatch.shape}")
    logger.debug(f"Dimension population_hatch: {population_hatch.dims}")

    # Always use only the first time slice of population data for the population effect
    population_hatch_no_time = population_hatch.isel(time=0).drop_vars("time")

    # Broadcast population data to match rainfall's time dimension
    # Use assign_coords to ensure proper dimension order matching rainfall_hatch
    population_hatch_broadcasted = population_hatch_no_time.expand_dims(
        {"time": rainfall_hatch.coords["time"]}, axis=2
    )

    # Weighted combination (element-wise)
    result = ((1 - E_RAT) * rainfall_hatch) + (E_RAT * population_hatch_broadcasted)
    logger.debug(f"Shape result water hatching: {result.shape}")

    return result
