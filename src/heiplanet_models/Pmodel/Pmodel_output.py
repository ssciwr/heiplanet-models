"""Output utilities for allocating P-model simulation arrays."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import xarray as xr
from netCDF4 import Dataset


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


class IncrementalNetCDFWriter:
    """Write P-model output chunks without holding the full state in memory."""

    def __init__(
        self,
        output_path: Path,
        longitude: xr.DataArray,
        latitude: xr.DataArray,
        time: xr.DataArray,
        compartments: list[str],
        chunk_lon: int,
        chunk_lat: int,
        attrs: dict[str, Any] | None = None,
        compression: bool = False,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.compartments = compartments
        self.dataset = Dataset(output_path, "w", format="NETCDF4")

        n_lon = longitude.size
        n_lat = latitude.size
        n_time = time.size

        self.dataset.createDimension("longitude", n_lon)
        self.dataset.createDimension("latitude", n_lat)
        self.dataset.createDimension("time", n_time)

        lon_var = self.dataset.createVariable("longitude", "f8", ("longitude",))
        lat_var = self.dataset.createVariable("latitude", "f8", ("latitude",))
        time_var = self.dataset.createVariable("time", "f8", ("time",))

        lon_var[:] = np.asarray(longitude.values, dtype=np.float64)
        lat_var[:] = np.asarray(latitude.values, dtype=np.float64)

        time_values = time.values
        if np.issubdtype(time_values.dtype, np.datetime64):
            time_var[:] = time_values.astype("datetime64[D]").astype(np.int64)
            time_var.units = "days since 1970-01-01 00:00:00"
            time_var.calendar = "proleptic_gregorian"
        else:
            time_var[:] = np.asarray(time_values, dtype=np.float64)

        lon_var.units = longitude.attrs.get("units", "")
        lat_var.units = latitude.attrs.get("units", "")

        self.variables = {}
        variable_chunks = (min(chunk_lon, n_lon), min(chunk_lat, n_lat), n_time)
        for name in compartments:
            self.variables[name] = self.dataset.createVariable(
                name,
                "f8",
                ("longitude", "latitude", "time"),
                zlib=compression,
                chunksizes=variable_chunks,
            )

        if attrs:
            for key, value in attrs.items():
                self.dataset.setncattr(key, value)

    def write_chunk(
        self,
        lon_slice: slice,
        lat_slice: slice,
        output_chunk: np.ndarray,
    ) -> None:
        for variable_index, name in enumerate(self.compartments):
            self.variables[name][lon_slice, lat_slice, :] = output_chunk[
                :, :, variable_index, :
            ]

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def create_incremental_netcdf_writer(
    output_path: Path,
    model_data,
    compartments: list[str],
    chunk_lon: int,
    chunk_lat: int,
    attrs: dict[str, Any] | None = None,
    compression: bool = False,
) -> IncrementalNetCDFWriter:
    """Create a NetCDF writer for chunked production runs."""

    return IncrementalNetCDFWriter(
        output_path=output_path,
        longitude=model_data.temperature_mean["longitude"],
        latitude=model_data.temperature_mean["latitude"],
        time=model_data.temperature_mean["time"],
        compartments=compartments,
        chunk_lon=chunk_lon,
        chunk_lat=chunk_lat,
        attrs=attrs,
        compression=compression,
    )
