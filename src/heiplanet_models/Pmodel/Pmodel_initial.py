"""Initial data loading and preprocessing utilities for the heiplanet model backend.

This module provides functions to efficiently load, preprocess, and align large
geospatial datasets (temperature, rainfall, population) for use in the PModel.
It supports chunked reading via Dask, robust error handling, and logging.

Typical usage example:
    model_input = load_data()
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from heiplanet_models.Pmodel.Pmodel_input import PmodelInput
from heiplanet_models.Pmodel.Pmodel_output import PmodelOutput
from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_INITIAL_CONDITIONS

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
COORDINATE_ALIGNMENT_ATOL = 1e-12
EXPECTED_MODEL_VARIABLES = ["eggs", "ed", "juv", "imm", "adults"]


# ---- Utility functions
def _validate_pmodel_settings(settings: dict[str, Any]) -> None:
    """Validate the Pmodel numerical and execution settings contract."""

    ode_system = settings.get("ode_system")
    if not isinstance(ode_system, dict):
        raise ValueError(  # noqa: TRY004
            "Settings must contain an 'ode_system' mapping."
        )

    time_step = ode_system.get("time_step")
    if isinstance(time_step, bool) or not isinstance(time_step, int):
        raise ValueError(  # noqa: TRY004
            "ode_system.time_step must be a positive integer."
        )
    if time_step <= 0:
        raise ValueError("ode_system.time_step must be a positive integer.")

    model_variables = ode_system.get("model_variables")
    if model_variables != EXPECTED_MODEL_VARIABLES:
        raise ValueError(
            "ode_system.model_variables must be "
            f"{EXPECTED_MODEL_VARIABLES!r}; got {model_variables!r}."
        )

    execution = settings.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(  # noqa: TRY004
            "Settings must contain an 'execution' mapping."
        )

    initial_year = execution.get("initial_year")
    final_year = execution.get("final_year")
    if (
        isinstance(initial_year, bool)
        or isinstance(final_year, bool)
        or not isinstance(initial_year, int)
        or not isinstance(final_year, int)
    ):
        raise ValueError(  # noqa: TRY004
            "execution.initial_year and execution.final_year must be integers."
        )
    if final_year < initial_year:
        raise ValueError(
            "execution.final_year must be greater than or equal to "
            "execution.initial_year."
        )


def read_global_settings(filepath_configuration_file: str) -> dict[str, Any]:
    """Load and validate global Pmodel settings from a YAML configuration file.

    Args:
        filepath_configuration_file (str): Absolute or relative path to the YAML configuration file containing ETL settings.

    Returns:
        dict[str, Any]: Parsed settings after numerical and execution validation.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
        ValueError: If required numerical or execution settings are invalid.
    """

    with open(filepath_configuration_file, "r", encoding="utf-8") as f:
        global_settings = yaml.safe_load(f)
    if not isinstance(global_settings, dict):
        raise ValueError("Settings YAML root must be a mapping.")  # noqa: TRY004
    _validate_pmodel_settings(global_settings)
    return global_settings


def check_all_paths_exist(path_dict: dict[str, str | Path]) -> bool:
    """Check if all values in the dictionary are existing filesystem paths.

    Args:
        path_dict (dict[str, str | Path]): Dictionary where keys are descriptive names and values are file or directory paths to check.

    Returns:
        bool: True if all paths exist, False otherwise.
    """

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
def assemble_filepaths(year: int | None = None, **etl_settings) -> dict[str, Path]:
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

    if (year is not None) and (not isinstance(year, int)):
        logger.error(f"Year {year} is not an integer.")
        raise TypeError

    path_root = Path(etl_settings["ingestion"]["path_root_datasets"])
    filename_components = etl_settings["ingestion"]["filename_components"]

    if year:
        dict_paths = {
            dataset_name: path_root
            / f"{comp['prefix']}{year}{comp['suffix'] or ''}{comp['extension']}"
            for dataset_name, comp in filename_components.items()
        }

    else:
        dict_paths = {
            dataset_name: path_root
            / f"{comp['prefix']}{comp['suffix'] or ''}{comp['extension']}"
            for dataset_name, comp in filename_components.items()
        }

    return dict_paths


def load_dataset(path_dataset: Path | str, **kwargs) -> xr.Dataset:
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
        except Exception:
            logger.exception("Error during rename")
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
        except Exception:
            logger.exception("Error during transpose")
            logger.debug(f"Available dimensions: {dataset.dims}")
            raise

    return dataset


def postprocess_dataset(
    dataset: xr.Dataset, reference_dataset: xr.Dataset | None = None, **kwargs
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
        except Exception:
            logger.exception("Error during alignment")
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

    for coord_name in ("longitude", "latitude"):
        if coord_name not in misaligned_dataset.coords:
            raise ValueError(
                f"misaligned_dataset is missing coordinate {coord_name!r}."
            )
        if coord_name not in fixed_dataset.coords:
            raise AttributeError(f"fixed_dataset is missing coordinate {coord_name!r}.")

    source_longitude = np.asarray(misaligned_dataset.longitude.values)
    source_latitude = np.asarray(misaligned_dataset.latitude.values)
    target_longitude = np.asarray(fixed_dataset.longitude.values)
    target_latitude = np.asarray(fixed_dataset.latitude.values)

    longitude_matches = (
        source_longitude.shape == target_longitude.shape
        and np.allclose(
            source_longitude,
            target_longitude,
            rtol=0.0,
            atol=COORDINATE_ALIGNMENT_ATOL,
            equal_nan=True,
        )
    )
    latitude_matches = source_latitude.shape == target_latitude.shape and np.allclose(
        source_latitude,
        target_latitude,
        rtol=0.0,
        atol=COORDINATE_ALIGNMENT_ATOL,
        equal_nan=True,
    )

    if longitude_matches and latitude_matches:
        logger.info(
            "Skipping spatial interpolation because longitude/latitude grids already match."
        )
        return misaligned_dataset

    try:
        logger.info(
            "Interpolating dataset to reference longitude/latitude grid because coordinates differ."
        )
        return misaligned_dataset.interp(
            longitude=fixed_dataset.longitude,
            latitude=fixed_dataset.latitude,
            method="linear",
        )
    except Exception:
        logger.exception("Failed to align coordinates using interpolation.")
        raise


def load_temperature_dataset(path_dataset: Path | str, **etl_settings) -> xr.Dataset:
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


def load_rainfall_dataset(path_dataset: Path | str, **etl_settings) -> xr.Dataset:
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


def load_population_dataset(path_dataset: Path | str, **etl_settings) -> xr.Dataset:
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
    except Exception:
        logger.exception("Failed to expand temperature array.")
        raise

    return temperature_daily, temperature_mean


def load_initial_conditions(
    filepath: Path | str | None = None,
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
        except Exception:
            logger.exception(
                "Failed to load previous initial conditions from '%s'.", filepath
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
            except Exception:
                logger.exception(
                    "Failed to extract variable '%s' from previous conditions.", var
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


def _load_and_align_datasets(
    paths: dict[str, Any], etl_settings: dict[str, Any]
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Load model source datasets and align population data to temperature."""

    try:
        temperature = load_temperature_dataset(
            path_dataset=paths["temperature_dataset"], **etl_settings
        )
    except Exception:
        logger.exception("Failed to load temperature dataset")
        raise

    # --- Load rainfall dataset
    try:
        rainfall = load_rainfall_dataset(
            path_dataset=paths["rainfall_dataset"], **etl_settings
        )
    except Exception:
        logger.exception("Failed to load rainfall dataset")
        raise

    # --- Load human population dataset
    try:
        human_population = load_population_dataset(
            path_dataset=paths["human_population_dataset"], **etl_settings
        )
    except Exception:
        logger.exception("Failed to load human_population dataset")
        raise

    # ==== Posprocess datasets
    # --- Human population
    params = etl_settings["transformation"]["human_population_dataset"][
        "postprocessing"
    ]
    human_population = postprocess_dataset(
        dataset=human_population, reference_dataset=temperature, **params
    )

    return temperature, rainfall, human_population


def _extract_common_model_inputs(
    temperature: xr.Dataset,
    rainfall: xr.Dataset,
    human_population: xr.Dataset,
    etl_settings: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Extract shared model arrays and initialize the population state."""

    temperature_variable_name = etl_settings["transformation"]["temperature_dataset"][
        "data_variable"
    ]
    temperature_mean = temperature[temperature_variable_name]

    rainfall_variable_name = etl_settings["transformation"]["rainfall_dataset"][
        "data_variable"
    ]
    rainfall_data = rainfall[rainfall_variable_name]
    logger.debug(f"Rainfall shape: {rainfall_data.shape}")

    human_population_variable_name = etl_settings["transformation"][
        "human_population_dataset"
    ]["data_variable"]
    population_data = human_population[human_population_variable_name]
    logger.debug(f"Population shape: {population_data.shape}")

    latitude = temperature_mean["latitude"]
    n_longitude, n_latitude = temperature_mean.shape[:2]
    filepath_initial_conditions = etl_settings["ingestion"]["initial_conditions"][
        "file_path_initial_conditions"
    ]
    initial_conditions = load_initial_conditions(
        filepath=filepath_initial_conditions,
        sizes=(n_longitude, n_latitude),
        **etl_settings,
    )

    return (
        temperature_mean,
        rainfall_data,
        population_data,
        latitude,
        initial_conditions,
    )


def load_all_data(paths: dict[str, Any], etl_settings: dict[str, Any]) -> PmodelInput:
    """Load model inputs for legacy backends with repeated sub-daily temperature."""

    temperature, rainfall, human_population = _load_and_align_datasets(
        paths, etl_settings
    )
    temperature_daily, temperature_mean = create_temperature_daily(
        temperature_dataset=temperature, **etl_settings
    )
    (
        _temperature_mean,
        rainfall_data,
        population_data,
        latitude,
        initial_conditions,
    ) = _extract_common_model_inputs(
        temperature, rainfall, human_population, etl_settings
    )

    return PmodelInput(
        initial_conditions=initial_conditions,
        latitude=latitude,
        population_density=population_data,
        rainfall=rainfall_data,
        temperature=temperature_daily,
        temperature_mean=temperature_mean,
    )


def load_all_data_daily(
    paths: dict[str, Any], etl_settings: dict[str, Any]
) -> PmodelInput:
    """Load model inputs for daily SciPy solving without repeated temperature.

    This is the low-memory input path for ``scipy_chunked`` production runs.
    It keeps the legacy ``load_all_data`` behavior unchanged and returns an
    empty ``temperature`` placeholder because the daily SciPy solver uses
    ``temperature_mean`` directly.
    """

    temperature, rainfall, human_population = _load_and_align_datasets(
        paths, etl_settings
    )
    (
        temperature_mean,
        rainfall_data,
        population_data,
        latitude,
        initial_conditions,
    ) = _extract_common_model_inputs(
        temperature, rainfall, human_population, etl_settings
    )
    temperature_daily = temperature_mean.isel(time=slice(0, 0)).rename(
        "temperature_daily"
    )

    return PmodelInput(
        initial_conditions=initial_conditions,
        latitude=latitude,
        population_density=population_data,
        rainfall=rainfall_data,
        temperature=temperature_daily,
        temperature_mean=temperature_mean,
    )


def create_model_output(
    dataset_base_shape, initial_conditions_shape, time_step
) -> PmodelOutput:

    if len(initial_conditions_shape) != 3:
        raise ValueError(
            "initial_conditions must have shape (longitude, latitude, ode_variable)."
        )

    if len(dataset_base_shape) == 0:
        raise ValueError("temperature_shape must contain at least one dimension.")

    if time_step <= 0:
        raise ValueError("time_step must be greater than 0.")

    number_longitudes, number_latitudes, _ = dataset_base_shape
    number_ode_variables = initial_conditions_shape[-1]
    number_times = int(dataset_base_shape[-1] / time_step)

    shape_output = (
        number_longitudes,
        number_latitudes,
        number_ode_variables,
        number_times,
    )
    return np.zeros(shape=shape_output, dtype=np.float64)
