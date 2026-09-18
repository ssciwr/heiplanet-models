from __future__ import annotations

import importlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

pkg = resources.files("heiplanet_models")
DEFAULT_CONFIG_FILE = Path(pkg / "config_Jmodel.json")


def load_config(config_path: Path | str = "default") -> dict[str, Any]:
    """Load the Jmodel configuration (run settings + computation graph).

    Args:
        config_path (Path | str): Path to a config file, or "default" to use
            the packaged `config_Jmodel.json`. Defaults to "default".

    Returns:
        dict[str, Any]: The parsed configuration.
    """
    config_file = DEFAULT_CONFIG_FILE if config_path == "default" else Path(config_path)
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


# helpers for loading code
def load_module(module_name: str, file_path: str):
    """
    load_module Load a python module from 'path' with alias 'alias'

    Args:
        module_name (str): module alias.
        file_path (str): Path to load the module from

    Returns:
        module: Python module that has been loaded
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error in loading module {file_path}") from e
    return module


def load_name_from_module(module_name: str, file_path: str, name: str):
    """
    load_name_from_module Load a python module from 'path' with alias 'alias'

    Args:
        module_name (str): module alias.
        file_path (str): Path to load the module from
        name (str): name to import
    Returns:
        module: Python module that has been loaded
    """
    module = load_module(module_name, file_path)
    return getattr(module, name)


def validate_spatial_alignment(arr1: xr.DataArray, arr2: xr.DataArray) -> None:
    """Validates that two xarray DataArrays have aligned spatial coordinates.

    Args:
        arr1 (xr.DataArray): The first DataArray.
        arr2 (xr.DataArray): The second DataArray.

    Raises:
        ValueError: If the 'latitude' or 'longitude' coordinates do not match
                    or if the coordinates are missing.
    """
    # Check latitude
    try:
        if not np.array_equal(arr1.latitude.values, arr2.latitude.values):
            raise ValueError(
                "Spatial coordinate 'latitude' of input arrays must be aligned."
            )
    except AttributeError:
        raise ValueError("Input DataArrays must have a 'latitude' coordinate.")

    # Check longitude
    try:
        if not np.array_equal(arr1.longitude.values, arr2.longitude.values):
            raise ValueError(
                "Spatial coordinate 'longitude' of input arrays must be aligned."
            )
    except AttributeError:
        raise ValueError("Input DataArrays must have a 'longitude' coordinate.")
