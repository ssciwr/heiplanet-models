"""Initial data loading and preprocessing utilities for the heiplanet model backend.

This module provides functions to efficiently load, preprocess, and align large
geospatial datasets (temperature, rainfall, population) for use in the PModel.
It supports chunked reading via Dask, robust error handling, and logging.

Typical usage example:
    model_input = load_data()
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Optional, Union

import xarray as xr
import numpy as np

from heiplanet_models.Pmodel.Pmodel_input import PmodelInput
from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_INITIAL_CONDITIONS

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---- Utility functions
def read_global_settings(filepath_configuration_file: str) -> dict[str, Any]:
    """Load global ETL settings from a YAML configuration file.

    Args:
        filepath_configuration_file (str): Absolute or relative path to the YAML configuration file containing ETL settings.

    Returns:
        dict[str, Any]: Parsed settings as a dictionary with string keys and values of any type, as loaded from the YAML file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    # TODO: move to utils.py in the future
    with open(filepath_configuration_file, "r") as f:
        global_settings = yaml.safe_load(f)
    return global_settings


def check_all_paths_exist(path_dict: dict[str, Union[str, Path]]) -> bool:
    """Check if all values in the dictionary are existing filesystem paths.

    Args:
        path_dict (dict[str, str | Path]): Dictionary where keys are descriptive names and values are file or directory paths to check.

    Returns:
        bool: True if all paths exist, False otherwise.
    """
    # TODO: move to utils.py in the future

    if not path_dict:
        logger.warning("Provided path dictionary is empty.")
        return False

    all_exist = True
    for key, p in path_dict.items():
        path_obj = Path(p)
        if path_obj.exists():
            logger.debug(f"Path for '{key}': {path_obj} ... OK")
        else:
            logger.error(f"Path for '{key}': {path_obj} ... Not Found")
            all_exist = False

    if not all_exist:
        logger.warning("One or more paths do not exist.")
    else:
        logger.info("All paths exist.")

    return all_exist


# ---- ETL Functions
def assemble_filepaths(year: int, **etl_settings) -> dict[str, Path]:
    """Assemble file paths for datasets for a given year based on ETL settings.

    Args:
        year (int): The year for which to assemble dataset file paths.
        **etl_settings: Arbitrary keyword arguments containing ETL configuration, must include
            'ingestion' with 'path_root_datasets' and 'filename_components'.

    Returns:
        dict[str, Path]: Dictionary mapping dataset names to their corresponding file paths as Path objects.

    Raises:
        KeyError: If required keys are missing in etl_settings.
        TypeError: If the year is not an integer or settings are malformed.
    """
    # TODO: move to utils.py in the future

    if not isinstance(year, int):
        logger.error(f"Year {year} is not an integer.")
        raise TypeError

    path_root = Path(etl_settings["ingestion"]["path_root_datasets"])
    filename_components = etl_settings["ingestion"]["filename_components"]

    dict_paths = {
        dataset_name: path_root
        / f"{comp['prefix']}{year}{comp['suffix'] or ''}{comp['extension']}"
        for dataset_name, comp in filename_components.items()
    }
    return dict_paths


def load_dataset(path_dataset: Union[Path, str], **kwargs) -> xr.Dataset:
    """Load an xarray dataset from a file path.

    Args:
        path_dataset (Union[Path, str]): Path to the dataset file (e.g., NetCDF file).
        **kwargs: Additional keyword arguments passed to xarray.open_dataset (e.g., engine, chunks).

    Returns:
        xr.Dataset: The loaded xarray Dataset object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        OSError: If the file cannot be opened or read.
        ValueError: If the file is not a valid dataset or cannot be parsed by xarray.
    """
    dataset = xr.open_dataset(filename_or_obj=path_dataset, **kwargs)
    return dataset


def preprocess_dataset(dataset: xr.Dataset, **kwargs) -> xr.Dataset:
    """Preprocess an xarray Dataset by renaming and/or transposing dimensions.

    Args:
        dataset (xr.Dataset): The xarray Dataset to preprocess.
        **kwargs: Optional keyword arguments:
            - names_dimensions (dict[str, str]): Mapping of old to new dimension names for renaming.
            - dimension_order (Union[list[str], tuple[str, ...]]): Desired order of dimensions for transposing.

    Returns:
        xr.Dataset: The preprocessed xarray Dataset.

    Raises:
        KeyError: If a specified dimension to rename or transpose does not exist in the dataset.
        ValueError: If the dimension order is invalid or incompatible with the dataset.
        Exception: For any other errors during renaming or transposing.
    """
    # --- Rename dimensions if specified
    names_dimensions = kwargs.get("names_dimensions")
    if names_dimensions:
        try:
            logger.debug(f"Before renaming dimensions: {dataset.dims}")
            dataset = dataset.rename(name_dict=names_dimensions)
            logger.debug(f"After renaming dimensions: {dataset.dims}")
        except Exception as e:
            logger.exception(f"Error during rename: {e}")
            logger.debug(f"Available dimensions: {dataset.dims}")
            raise

    # --- Transpose dimensions if specified
    dimension_order = kwargs.get("dimension_order")
    if dimension_order:
        # Check if dimension_order is a permutation of all dataset.dims
        dims_set = set(dataset.dims)

        if set(dimension_order) != dims_set or len(dimension_order) != len(
            dataset.dims
        ):
            msg = f"dimension_order {dimension_order} must be a permutation of all dataset dimensions {tuple(dataset.dims)}."
            logger.error(msg)
            raise ValueError(msg)
        try:
            logger.debug(f"Before transpose dimensions: {dataset.dims}")
            dataset = dataset.transpose(*dimension_order)
        except Exception as e:
            logger.exception(f"Error during transpose: {e}")
            logger.debug(f"Available dimensions: {dataset.dims}")
            raise

    return dataset


def postprocess_dataset(
    dataset: xr.Dataset, reference_dataset: Optional[xr.Dataset] = None, **kwargs
) -> xr.Dataset:
    """Postprocess an xarray Dataset.

    Args:
        dataset (xr.Dataset): The xarray Dataset to postprocess.
        reference_dataset (Optional[xr.Dataset]): Reference dataset to align coordinates to, if provided.
        **kwargs: Optional keyword arguments:
            - align_dataset (bool): If True and reference_dataset is provided, align coordinates to reference.

    Returns:
        xr.Dataset: The postprocessed (and possibly aligned) xarray Dataset.

    Raises:
        Exception: If alignment fails or an unexpected error occurs during postprocessing.
    """
    # --- Align dataset if specified
    align_dataset = kwargs.get("align_dataset")
    if align_dataset and reference_dataset:
        try:
            dataset = align_xarray_datasets(
                misaligned_dataset=dataset, fixed_dataset=reference_dataset
            )
        except Exception as e:
            logger.exception(f"Error during alignment: {e}")
            raise

    return dataset


def align_xarray_datasets(
    misaligned_dataset: xr.Dataset,
    fixed_dataset: xr.Dataset,
) -> xr.Dataset:
    """Align coordinates of one Dataset to another using interpolation.

    Args:
        misaligned_dataset: Dataset to be interpolated (e.g., population_density).
        fixed_dataset: Dataset providing target longitude and latitude coordinates (e.g., rainfall).

    Returns:
        xr.Dataset: Interpolated Dataset aligned to the target grid.

    Raises:
        Exception: If interpolation fails.
    """
    if not misaligned_dataset.data_vars:
        logger.debug(
            "Misaligned dataset is empty; returning a new dataset with reference coordinates."
        )
        return xr.Dataset(coords=fixed_dataset.coords)

    try:
        return misaligned_dataset.interp(
            longitude=fixed_dataset.longitude,
            latitude=fixed_dataset.latitude,
            method="linear",
        )
    except Exception as e:
        logger.error("Failed to align coordinates using interpolation: %s", e)
        raise


def load_temperature_dataset(
    path_dataset: Union[Path, str], **etl_settings
) -> xr.Dataset:
    """Load and preprocess the temperature dataset for a given path and ETL settings.

    Args:
        path_dataset (Union[Path, str]): Path to the temperature dataset file (e.g., NetCDF file).
        **etl_settings: Arbitrary keyword arguments containing ETL configuration.

    Returns:
        xr.Dataset: The loaded and preprocessed temperature dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        OSError: If the file cannot be opened or read.
        ValueError: If the file is not a valid dataset or cannot be parsed by xarray.
        Exception: For any other errors during preprocessing.
    """
    # -- Load dataset
    xarray_params = etl_settings["ingestion"]["xarray_load_settings"]
    dataset = load_dataset(path_dataset=path_dataset, **xarray_params)

    # -- Preprocess dataset
    logger.debug(f"Dataset Name: {dataset.data_vars}")
    preprocess_params = etl_settings["transformation"]["temperature_dataset"].get(
        "preprocessing"
    )
    if preprocess_params:
        dataset = preprocess_dataset(dataset=dataset, **preprocess_params)
    return dataset


def load_rainfall_dataset(path_dataset: Union[Path, str], **etl_settings) -> xr.Dataset:
    """Load and preprocess the rainfall dataset for a given path and ETL settings.

    Args:
        path_dataset (Union[Path, str]): Path to the rainfall dataset file (e.g., NetCDF file).
        **etl_settings: Arbitrary keyword arguments containing ETL configuration.

    Returns:
        xr.Dataset: The loaded and preprocessed rainfall dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        OSError: If the file cannot be opened or read.
        ValueError: If the file is not a valid dataset or cannot be parsed by xarray.
        Exception: For any other errors during preprocessing.
    """
    # -- Load dataset
    xarray_params = etl_settings["ingestion"]["xarray_load_settings"]
    dataset = load_dataset(path_dataset=path_dataset, **xarray_params)

    # -- Preprocess dataset
    preprocess_params = etl_settings["transformation"]["rainfall_dataset"].get(
        "preprocessing"
    )
    if preprocess_params:
        dataset = preprocess_dataset(dataset=dataset, **preprocess_params)

    return dataset


def load_population_dataset(
    path_dataset: Union[Path, str], **etl_settings
) -> xr.Dataset:
    """Load and preprocess the human population dataset for a given path and ETL settings.

    Args:
        path_dataset (Union[Path, str]): Path to the human population dataset file (e.g., NetCDF file).
        **etl_settings: Arbitrary keyword arguments containing ETL configuration.

    Returns:
        xr.Dataset: The loaded and preprocessed human population dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        OSError: If the file cannot be opened or read.
        ValueError: If the file is not a valid dataset or cannot be parsed by xarray.
        Exception: For any other errors during preprocessing.
    """
    # -- Load dataset
    xarray_params = etl_settings["ingestion"]["xarray_load_settings"]
    dataset = load_dataset(
        path_dataset=path_dataset, decode_times=False, **xarray_params
    )

    # -- Preprocess dataset
    preprocess_params = etl_settings["transformation"]["human_population_dataset"].get(
        "preprocessing"
    )
    if preprocess_params:
        dataset = preprocess_dataset(dataset=dataset, **preprocess_params)

    return dataset


def create_temperature_daily(
    temperature_dataset: xr.Dataset, **etl_settings
) -> tuple[xr.DataArray, xr.DataArray]:
    """Create a daily temperature DataArray by expanding the mean temperature along the time axis.

    Args:
        temperature_dataset (xr.Dataset): The xarray Dataset containing temperature data.
        **etl_settings: Arbitrary keyword arguments containing ETL configuration.

    Returns:
        tuple[xr.DataArray, xr.DataArray]:
            - temperature_daily: Expanded daily temperature DataArray.
            - temperature_mean: Original temperature DataArray.

    Raises:
        KeyError: If required keys are missing in etl_settings or dataset variables.
        ValueError: If the temperature data cannot be expanded as required.
        Exception: For any other errors during array creation.
    """
    time_step = etl_settings["ode_system"]["time_step"]
    data_variable_temperature = etl_settings["transformation"]["temperature_dataset"][
        "data_variable"
    ]

    temperature_mean = temperature_dataset[data_variable_temperature]

    try:
        temperature_daily = xr.DataArray(
            np.repeat(
                temperature_mean.data,
                repeats=time_step,
                axis=temperature_mean.get_axis_num("time"),
            ),
            dims=temperature_mean.dims,
            coords={
                "longitude": temperature_mean.longitude,
                "latitude": temperature_mean.latitude,
            },
            name="temperature_daily",
        )
    except Exception as e:
        logger.error(f"Failed to expand temperature array: {e}")
        raise

    return temperature_daily, temperature_mean


def load_initial_conditions(
    filepath: Optional[Union[Path, str]] = None,
    sizes: tuple[int, int] = (0, 0),
    **etl_settings,
) -> xr.DataArray:
    """Load or initialize the model state variables for the simulation as an xarray.DataArray."""

    CONST_K1 = CONSTANTS_INITIAL_CONDITIONS["CONST_K1"]
    CONST_K2 = CONSTANTS_INITIAL_CONDITIONS["CONST_K2"]

    MODEL_VARIABLES = etl_settings["ode_system"]["model_variables"]

    n_longitude, n_latitude = sizes
    n_vars = len(MODEL_VARIABLES)

    coords = {
        "longitude": np.arange(n_longitude),
        "latitude": np.arange(n_latitude),
        "variable": MODEL_VARIABLES,
    }

    if filepath is None or not Path(filepath).exists():
        data = np.zeros((n_longitude, n_latitude, n_vars), dtype=np.float64)
        data[:, :, 1] = CONST_K1 * CONST_K2
        logger.info("Initialized initial conditions with default values.")
    else:
        try:
            ds = load_dataset(filepath)
        except Exception as e:
            logger.error(
                f"Failed to load previous initial conditions from '{filepath}': {e}"
            )
            raise
        data = np.zeros((n_longitude, n_latitude, n_vars), dtype=np.float64)
        for i, var in enumerate(MODEL_VARIABLES):
            if var not in ds:
                logger.error(
                    f"Variable '{var}' not found in previous conditions dataset."
                )
                raise KeyError(
                    f"Variable '{var}' not found in previous conditions dataset."
                )
            try:
                data[:, :, i] = ds[var].isel(time=-1).values
            except Exception as e:
                logger.error(
                    f"Failed to extract variable '{var}' from previous conditions: {e}"
                )
                raise
        logger.info("Loaded initial conditions from previous file.")

    v0_xr = xr.DataArray(
        data,
        dims=("longitude", "latitude", "variable"),
        coords=coords,
        name="initial_conditions",
    )
    return v0_xr


def load_all_data(paths: dict[str, Any], etl_settings: dict[str, Any]) -> PmodelInput:
    """Load, preprocess, and assemble all required datasets and arrays for the model.

    Args:
        paths (dict[str, Any]): Dictionary mapping dataset names to their file paths.
        etl_settings (dict[str, Any]): Dictionary containing ETL configuration and transformation settings.

    Returns:
        PmodelInput: An object containing all loaded and processed model input arrays.

    Raises:
        FileNotFoundError: If any required dataset file does not exist.
        KeyError: If required keys are missing in etl_settings or datasets.
        Exception: For any other errors during data loading or processing.
    """
    # ===========================
    # ===    Load Datasets    ===
    # ===========================
    # --- Load temperature dataset
    try:
        temperature = load_temperature_dataset(
            path_dataset=paths["temperature_dataset"], **etl_settings
        )
    except Exception as e:
        logger.exception(f"Failed to load temperature dataset {e}")
        raise

    # --- Load rainfall dataset
    try:
        rainfall = load_rainfall_dataset(
            path_dataset=paths["rainfall_dataset"], **etl_settings
        )
    except Exception as e:
        logger.exception(f"Failed to load rainfall dataset {e}")
        raise

    # --- Load human population dataset
    try:
        human_population = load_population_dataset(
            path_dataset=paths["human_population_dataset"], **etl_settings
        )
    except Exception as e:
        logger.exception(f"Failed to load human_population dataset {e}")
        raise

    # ==== Posprocess datasets
    # --- Human population
    params = etl_settings["transformation"]["human_population_dataset"][
        "postprocessing"
    ]
    human_population = postprocess_dataset(
        dataset=human_population, reference_dataset=temperature, **params
    )

    # ========================================
    # ===    Extract/Create Data Arrays    ===
    # ========================================
    # --- Temperature arrays
    da_temperature, da_temperature_mean = create_temperature_daily(
        temperature_dataset=temperature, **etl_settings
    )

    # --- Rainfall Array
    rainfall_variable_name = etl_settings["transformation"]["rainfall_dataset"][
        "data_variable"
    ]
    da_rainfall = rainfall[rainfall_variable_name]
    logger.debug(f"Rainfall shape: {da_rainfall.shape}")

    # --- Human population Array
    human_population_variable_name = etl_settings["transformation"][
        "human_population_dataset"
    ]["data_variable"]
    da_population = human_population[human_population_variable_name]
    logger.debug(f"Population shape: {da_population.shape}")

    # --- Latitude Array
    da_latitude = da_temperature_mean["latitude"]

    # --- Create/Load initial conditions
    # Extract longitude and latitude dimensions from temperature mean shape
    n_longitude, n_latitude = da_temperature_mean.shape[:2]
    filepath_initial_conditions = etl_settings["ingestion"]["initial_conditions"][
        "file_path_initial_conditions"
    ]
    initial_conditions = load_initial_conditions(
        filepath=filepath_initial_conditions,
        sizes=(n_longitude, n_latitude),
        **etl_settings,
    )

    # ================================================
    # ===    Return Container for all variables    ===
    # ================================================
    return PmodelInput(
        initial_conditions=initial_conditions,
        latitude=da_latitude,
        population_density=da_population,
        rainfall=da_rainfall,
        temperature=da_temperature,
        temperature_mean=da_temperature_mean,
    )
