"""Output utilities for allocating P-model simulation arrays."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@dataclass
class PmodelOutput:
    """Container for model output arrays.

    Attributes:
        model_output (np.ndarray): Array with shape
            ``(longitude, latitude, ode_variable, time)``.
    """

    model_output: np.ndarray

    def __repr__(self):
        attr_strings = []
        for attr, value in self.__dict__.items():
            type_name = type(value).__name__
            shape_str = ""
            if hasattr(value, "shape"):
                shape_str = f", shape={value.shape}"
            attr_strings.append(f"\n\t{attr}: {type_name}{shape_str}")
        attrs = ",".join(attr_strings)
        return f"{self.__class__.__name__}({attrs})"


def build_output_dataset(state, model_data, compartments) -> xr.Dataset:
    data_vars = {}
    for i, name in enumerate(compartments):
        data_vars[name] = xr.DataArray(
            state[..., i, :],  # shape: (longitude, latitude, time)
            dims=("longitude", "latitude", "time"),
            coords={
                "longitude": model_data.temperature_mean["longitude"],
                "latitude": model_data.temperature_mean["latitude"],
                "time": model_data.temperature_mean["time"],
            },
            name=name,
        )

    return xr.Dataset(data_vars)


def assemble_output_filepath(
    year: int | None = None, **serving_settings: dict[str, Any]
) -> Path:
    """Assemble output file path from serving settings."""
    path_root = Path(serving_settings["path_output_datasets"])
    filename_components = serving_settings["filename_components"]

    prefix = filename_components.get("prefix", "")
    suffix = filename_components.get("suffix") or ""
    extension = filename_components.get("extension") or ".nc"

    if year is not None:
        filename = f"{prefix}{year}{suffix}{extension}"
    else:
        filename = f"{prefix}{suffix}{extension}"

    return path_root / filename


def save_output_dataset(
    dataset: xr.Dataset,
    year: int | None = None,
    **serving_settings: dict[str, Any],
) -> Path:
    """Save output dataset to NetCDF and return output path."""
    output_path = assemble_output_filepath(year=year, **serving_settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    return output_path
