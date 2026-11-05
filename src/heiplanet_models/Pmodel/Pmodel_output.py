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
        dtype: str = "f8",
        complevel: int = 4,
        variable_attrs: dict[str, Any] | None = None,
    ) -> None:
        """Create the NetCDF file and its variables.

        Args:
            dtype: NetCDF storage type of the compartment variables, e.g.
                ``"f8"`` (float64) or ``"f4"`` (float32, half the size).
            compression: Enable zlib compression of compartment variables.
            complevel: zlib compression level (1-9), used if ``compression``.
            variable_attrs: Attributes set on every compartment variable,
                e.g. ``{"cell_methods": "time: mean"}``.
        """
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
                dtype,
                ("longitude", "latitude", "time"),
                zlib=compression,
                complevel=complevel,
                chunksizes=variable_chunks,
            )
            for key, value in (variable_attrs or {}).items():
                self.variables[name].setncattr(key, value)

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


TIME_AGGREGATIONS = ("daily", "monthly_mean")


def monthly_groups(time: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group a daily time axis into consecutive calendar months.

    Args:
        time: Daily datetime64 coordinate, sorted ascending.

    Returns:
        ``(starts, counts, month_starts)``: index of each month's first day,
        number of days in each month, and each month's first-of-month date.

    Raises:
        ValueError: If ``time`` is not a datetime coordinate or not sorted.
    """
    values = np.asarray(time.values)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError("Monthly aggregation needs a datetime64 time coordinate.")
    if values.size == 0:
        raise ValueError("Monthly aggregation needs at least one time step.")
    if np.any(np.diff(values.astype("int64")) < 0):
        raise ValueError("Monthly aggregation needs a sorted time coordinate.")

    month_keys = values.astype("datetime64[M]")
    starts = np.flatnonzero(np.r_[True, month_keys[1:] != month_keys[:-1]])
    counts = np.diff(np.r_[starts, values.size])
    month_starts = month_keys[starts].astype("datetime64[ns]")
    return starts, counts, month_starts


class MonthlyMeanWriter:
    """Chunk writer that averages daily chunks per calendar month.

    Each chunk ``(lon, lat, variable, day)`` is reduced to
    ``(lon, lat, variable, month)`` before being passed to ``inner``.
    """

    def __init__(self, inner, time: xr.DataArray) -> None:
        self.inner = inner
        self.starts, self.counts, self.month_starts = monthly_groups(time)

    def write_chunk(
        self, lon_slice: slice, lat_slice: slice, output_chunk: np.ndarray
    ) -> None:
        sums = np.add.reduceat(output_chunk, self.starts, axis=-1)
        self.inner.write_chunk(lon_slice, lat_slice, sums / self.counts)


class LastStepWriter:
    """Chunk writer that passes on only the last time step of each chunk."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def write_chunk(
        self, lon_slice: slice, lat_slice: slice, output_chunk: np.ndarray
    ) -> None:
        self.inner.write_chunk(lon_slice, lat_slice, output_chunk[..., -1:])


class TeeWriter:
    """Chunk writer that forwards every chunk to several writers."""

    def __init__(self, *writers) -> None:
        self.writers = writers

    def write_chunk(
        self, lon_slice: slice, lat_slice: slice, output_chunk: np.ndarray
    ) -> None:
        for writer in self.writers:
            writer.write_chunk(lon_slice, lat_slice, output_chunk)


def aggregate_dataset_monthly(dataset: xr.Dataset) -> xr.Dataset:
    """Average a daily output dataset per calendar month (first-of-month time)."""

    starts, counts, month_starts = monthly_groups(dataset["time"])
    data_vars = {}
    for name, data in dataset.data_vars.items():
        axis = data.get_axis_num("time")
        sums = np.add.reduceat(data.values, starts, axis=axis)
        shape = [1] * sums.ndim
        shape[axis] = counts.size
        data_vars[name] = (
            data.dims,
            sums / counts.reshape(shape),
            {**data.attrs, "cell_methods": "time: mean"},
        )
    coords = {name: coord for name, coord in dataset.coords.items() if name != "time"}
    coords["time"] = month_starts
    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs={**dataset.attrs, "time_aggregation": "monthly_mean"},
    )


def final_state_filepath(
    year: int | None = None, **serving_settings: dict[str, Any]
) -> Path:
    """Path of the end-of-run state file next to the model output.

    The file holds the last daily state of every compartment and is used as
    the initial condition of the following year.
    """
    output_path = assemble_output_filepath(year=year, **serving_settings)
    return output_path.with_name(f"{output_path.stem}_final_state{output_path.suffix}")


def output_is_daily(path: Path) -> bool:
    """Whether a model output file stores daily values (files without the
    ``time_aggregation`` attribute predate it and are daily)."""

    with Dataset(path, "r") as dataset:
        aggregation = getattr(dataset, "time_aggregation", "daily")
    return aggregation == "daily"


class FinalStateCollector:
    """Chunk writer that keeps only the last time step of each solved chunk.

    Used for spin-up runs: the full output is discarded and only the final
    state, shape ``(longitude, latitude, variable)``, is kept in memory.
    """

    def __init__(self, n_lon: int, n_lat: int, n_variables: int) -> None:
        self.state = np.zeros((n_lon, n_lat, n_variables), dtype=np.float64)

    def write_chunk(
        self,
        lon_slice: slice,
        lat_slice: slice,
        output_chunk: np.ndarray,
    ) -> None:
        self.state[lon_slice, lat_slice, :] = output_chunk[..., -1]


def create_incremental_netcdf_writer(
    output_path: Path,
    model_data,
    compartments: list[str],
    chunk_lon: int,
    chunk_lat: int,
    attrs: dict[str, Any] | None = None,
    compression: bool = False,
    dtype: str = "f8",
    complevel: int = 4,
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
        dtype=dtype,
        complevel=complevel,
    )
