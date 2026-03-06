from __future__ import annotations

import xarray as xr
import pandas as pd
import dask.array as da
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class JModelData:
    name: str
    input: str | Path | None = None  # Placeholder for input data, to be
    output: str | Path | None = None  # Placeholder for output data, to be defined later
    run_mode: str = "forbidden"  # dask run mode used by xarray
    r0_data: pd.DataFrame | None = None  # Placeholder for R0 data
    min_temp: float = 0.0  # Minimum temperature for interpolation
    max_temp: float = 45.0  # Maximum temperature for interpolation
    step: float = 0.1  # Step size for temperature interpolation
    temp_colname: str = "t2m"
    out_colname: str = "R0"
    grid_data_baseurl: str | None = None
    year: int | None = None  # Year for the model


def read_default_config() -> dict[str, str | np.int64 | None]:
    """Reads the default configuration for the JModel from a JSON file.

    Returns:
        dict[str, str | np.int64 | None]: A dictionary containing the default configuration.
    """
    config_path = Path(__file__).parent / "config_Jmodel.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def setup_modeldata(
    input: str | Path | None = None,
    output: str | Path | None = None,
    r0_path: str | Path | None = None,
    run_mode: str = "forbidden",
    grid_data_baseurl: str | None = None,
    year: int | None = None,
    temp_colname: str = "t2m",
    out_colname: str = "R0",
) -> JModelData:
    """Initializes the JModel with the given configuration.

    Args:
        input (str | None): Path to the input data file.
        output (str | None): Path to the output data file.
        r0_path (str | None): Path to the R0 data file.
        run_mode (str): Dask run mode used by xarray, default is "forbidden".
        grid_data_baseurl (str | None): Base URL for the grid data.
        year (int | None): Year for the model, default is None.
        temp_colname (str): Name of the temperature column in the input data, default is "t2m".
        out_colname (str): Name of the output column for R0 data, default is "R0".

    """

    # set up plumbing for the model
    if run_mode not in ["forbidden", "parallelized"]:
        raise ValueError(
            f"Invalid run mode: {run_mode}. Supported modes are 'forbidden', 'parallelized'. For the meaning of these modes, see the documentation. of xarray.apply_ufunc"
        )

    # set data paths and get r0 data
    if input is None:
        raise ValueError("Input data path must be provided in the configuration.")

    if output is None:
        raise ValueError("Output data path must be provided in the configuration.")

    # read R0 data from the given path
    if r0_path is None:
        raise ValueError("R0 data path must be provided in the configuration.")
    else:
        r0_data = pd.read_csv(r0_path)
    step_temp = (
        r0_data.Temperature[1] - r0_data.Temperature[0]
    )  # assume uniform step size
    min_temp = r0_data.Temperature.min()
    max_temp = r0_data.Temperature.max()

    isnone_params = [
        grid_data_baseurl is None,
        year is None,
    ]

    if any(isnone_params) and not all(isnone_params):
        raise ValueError(
            "Grid data configuration is incomplete. Please provide all parameters: grid_data_baseurl, and year, or do not set any to have them all set to 'None'."
        )
    else:
        # don´t do anything here, because None indicates the grid data is not used
        pass

    return JModelData(
        name="JModel",
        input=input,
        output=output,
        run_mode=run_mode,
        r0_data=r0_data,
        min_temp=min_temp,
        max_temp=max_temp,
        step=step_temp,
        temp_colname=temp_colname,
        out_colname=out_colname,
        grid_data_baseurl=grid_data_baseurl,
        year=year,
    )


def _interpolate_r0(
    temp: np.ndarray | da.core.Array,
    r0_data: pd.DataFrame,
    min_temp: float,
    max_temp: float,
) -> np.ndarray | da.core.Array:
    """Interpolates R0 values based on temperature using the stored R0 data.
    Args:
        temp (np.ndarray | da.core.Array): Temperature values for which to interpolate R0.
        r0_data (pd.DataFrame): DataFrame containing R0 data with 'Temperature'
            and 'Median_R0' columns.
        min_temp (np.float64): Minimum temperature for interpolation.
        max_temp (np.float64): Maximum temperature for interpolation.
    Returns:
        np.ndarray | da.core.Array: Interpolated R0 values corresponding to the input temperature values.
    """
    temp_values = temp.compute() if hasattr(temp, "compute") else temp
    if not isinstance(temp_values, np.ndarray) and not isinstance(temp, da.core.Array):
        temp_values = temp_values.values  # Ensure temp is a numpy array

    # Create result array with same shape as input
    result = np.full_like(temp, np.nan, dtype=float)

    # Find non-nan temperature values (equivalent to R's valid mask)
    # this applies to wherever there is water, since that will carry a NaN temperature
    valid_mask_land = ~np.isnan(temp_values)
    # Find valid temperature values within the interpolation range
    valid_mask_temp = (temp_values >= min_temp) & (temp_values <= max_temp)

    # Only interpolate where we have valid values
    valid_mask = valid_mask_land & valid_mask_temp

    if np.any(valid_mask):
        result[valid_mask] = np.interp(
            temp_values[valid_mask],  # Only pass valid values
            r0_data.Temperature.values,
            r0_data.Median_R0.values,
            left=np.nan,
            right=np.nan,
        )

    # set all values for temperatures outside the valid range to 0
    result[~valid_mask_temp] = 0.0
    # all the values for water areas should not exist
    result[~valid_mask_land] = np.nan

    return result


def read_input_data(model_data: JModelData) -> xr.Dataset:
    """Read input data from given source 'model_data.input'

    Args:
        model_data (JModelData): Data class containing the model configuration and input data path.

    Returns:
        xr.Dataset: xarray dataset containing the input data for the model.
    """

    try:
        data = xr.open_dataset(
            model_data.input,
            chunks=None if model_data.run_mode == "forbidden" else "auto",
        )
    except Exception as e:
        raise RuntimeError("Input data source could not be read.") from e

    return data


def run_model(
    model_data: JModelData, data: xr.Dataset | pd.DataFrame
) -> xr.Dataset | pd.DataFrame:
    """Runs the JModel with the provided input data. Applies the R0 interpolation based on temperature values from the stored R0 data and returns a new dataset or dataframe with the R0 data.

    Args:
        model_data (JModelData): Data class containing the model configuration and R0 data.
        data (xr.Dataset | pd.DataFrame): Input data containing temperature values.

    Returns:
        xr.Dataset | pd.DataFrame: A dataset or dataframe with the incoming R0 data interpolated based on the temperature values at each grid point.
    """
    r0_map = xr.apply_ufunc(
        lambda t: _interpolate_r0(
            t,
            model_data.r0_data,
            model_data.min_temp,
            model_data.max_temp,
        ),
        data[model_data.temp_colname],
        input_core_dims=[[]],
        output_core_dims=[[]],
        dask=model_data.run_mode,
        keep_attrs=True,
    ).rename(model_data.out_colname)

    # R0 is a derived, dimensionless quantity, so we need to adjust attributes here.
    r0_map = r0_map.assign_attrs(
        long_name=model_data.out_colname,
        standard_name="basic_reproduction_number",
        units="-",  # R0 is dimensionless
    )

    return r0_map


def store_output_data(
    model_data: JModelData, data: xr.Dataset | xr.DataArray | pd.DataFrame
) -> None:
    data.to_netcdf(model_data.output)
