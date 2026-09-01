import logging

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MORTALITY_MOSQUITO_A,
    CONSTANTS_MORTALITY_MOSQUITO_E,
    CONSTANTS_MORTALITY_MOSQUITO_ED,
    CONSTANTS_MORTALITY_MOSQUITO_J,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mosq_mort_e(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the daily mortality rate for eggs as an xarray.DataArray."""
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_5"]

    mortality_rate_egg = CONST_1 * np.exp(
        CONST_2 * ((temperature.data - CONST_3) / CONST_4) ** CONST_5
    )
    mortality_rate_egg = -np.log(mortality_rate_egg)

    return xr.DataArray(
        mortality_rate_egg,
        coords=temperature.coords,
        dims=temperature.dims,
        name="mosq_mort_e",
    )


def mosq_mort_j(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the daily mortality rate for juvenile mosquitoes as an xarray.DataArray."""
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_5"]

    mortality_rate_juvenile = CONST_1 * np.exp(
        CONST_2 * ((temperature.data - CONST_3) / CONST_4) ** CONST_5
    )
    mortality_rate_juvenile = np.where(
        mortality_rate_juvenile > 0, mortality_rate_juvenile, 1e-12
    )
    mortality_rate_juvenile = -np.log(mortality_rate_juvenile)

    return xr.DataArray(
        mortality_rate_juvenile,
        coords=temperature.coords,
        dims=temperature.dims,
        name="mosq_mort_j",
    )


def mosq_mort_a(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the daily mortality rate for adult mosquitoes as an xarray.DataArray."""
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_5"]
    CONST_6 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_6"]

    T = temperature.data
    mask_pos = T > 0

    # Compute both branches
    pos_val = (
        CONST_1 * np.exp(CONST_2 * ((T - CONST_3) / CONST_4) ** CONST_5) * T**CONST_6
    )
    zero_val = CONST_1 * np.exp(CONST_2 * ((T - CONST_3) / CONST_4) ** CONST_5)

    # Use np.where to select the correct branch
    mortality_rate_adult = np.where(mask_pos, pos_val, zero_val)
    mortality_rate_adult = -np.log(mortality_rate_adult)

    return xr.DataArray(
        mortality_rate_adult,
        coords=temperature.coords,
        dims=temperature.dims,
        name="mosq_mort_a",
    )


def mosq_surv_ed(temperature: xr.DataArray, step_t: int | None = None) -> xr.DataArray:
    """
    Calculates mosquito survival rate as a function of temperature.

    Args:
        temperature (xr.DataArray): 3D DataArray of temperature values (e.g., (x, y, t)).
        step_t (int, optional): Time step, used for winter mortality calculation. Defaults to None.

    Returns:
        xr.DataArray: DataArray of mosquito survival rates with the same shape as `temperature`.
    """
    if not isinstance(temperature, xr.DataArray):
        raise ValueError("Input 'temperature' must be an xarray.DataArray.")  # noqa: TRY004
    if temperature.ndim != 3:
        raise ValueError("Input 'temperature' must be a 3D DataArray.")

    ED_SURV_BL = CONSTANTS_MORTALITY_MOSQUITO_ED["ED_SURV_BL"]
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_5"]

    # Find the time dimension name
    time_dim = temperature.dims[-1]

    # Rechunk so that the time dimension is a single chunk (if using Dask)
    if hasattr(temperature.data, "chunks"):
        temperature = temperature.chunk({time_dim: -1})

    t_cummin = xr.apply_ufunc(
        np.minimum.accumulate,
        temperature,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[temperature.dtype],
    )

    # Apply the survival formula
    t_out = (
        ED_SURV_BL
        * CONST_1
        * np.exp(CONST_2 * ((t_cummin - CONST_3) / CONST_4) ** CONST_5)
    )

    return xr.DataArray(
        t_out.values,
        coords=temperature.coords,
        dims=temperature.dims,
        name="mosq_surv_ed",
    )
