"""
The purpose of this script is to run a dummy version of the Albopictus model
using a small subset of data for testing and debugging purposes.
It reads configuration settings, loads necessary datasets, and executes
the model's ODE system over a specified time range, logging key outputs
for verification.

You can verify intermediate outputs by checking the logs generated during execution.
Compare the outputs from the matlab/octave version of the model to ensure consistency.

Note: the structure mimic the structure found in the octave version of the model. With this you can easily
compare both implementations step by step.

"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr


from heiplanet_models.Pmodel.Pmodel_rates_mortality import (
    mosq_mort_e,
    mosq_mort_j,
    mosq_mort_a,
    mosq_surv_ed,
)

from heiplanet_models.Pmodel.Pmodel_rates_development import (
    mosq_dev_j,
    mosq_dev_i,
    carrying_capacity,
)

from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    water_hatching,
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
)
from heiplanet_models.Pmodel.Pmodel_initial import (
    #    assemble_filepaths,
    read_global_settings,
    check_all_paths_exist,
    load_all_data,
)

from heiplanet_models.Pmodel.Pmodel_ode import (
    albopictus_ode_system,
    albopictus_log_ode_system,
    rk4_step,
)

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

FILEPATH_ETL_SETTINGS = "./src/heiplanet_models/Pmodel/global_settings_dummy.yaml"

INITIAL_YEAR = 2024
FINAL_YEAR = 2024


# --------------------------------------------------------------#
# -----             Helper functions               ------------ #
# --------------------------------------------------------------#
def assemble_filepaths_no_year(**etl_settings) -> dict[str, Path]:
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

    path_root = Path(etl_settings["ingestion"]["path_root_datasets"])
    filename_components = etl_settings["ingestion"]["filename_components"]

    dict_paths = {
        dataset_name: path_root
        / f"{comp['prefix']}{comp['suffix'] or ''}{comp['extension']}"
        for dataset_name, comp in filename_components.items()
    }
    return dict_paths


def print_time_slices(arr):
    """
    Print each time slice of an xarray DataArray or a NumPy array.
    If 'time' dimension exists (for DataArray), print each time slice.
    Otherwise, print slices along the last dimension.
    """

    if isinstance(arr, xr.DataArray):
        if "time" in arr.dims:
            for i, t in enumerate(arr.time):
                print(f"Time slice {i}):")
                print(arr.sel(time=t).values)
                print("-" * 40)
        else:
            last_dim = arr.dims[-1]
            for i in range(arr.shape[-1]):
                print(f"Slice {i} ({last_dim}={i}):")
                print(arr.isel({last_dim: i}).values)
                print("-" * 40)
    elif isinstance(arr, np.ndarray):
        for i in range(arr.shape[-1]):
            print(f"Slice {i} (last axis={i}):")
            print(arr[..., i])
            print("-" * 40)
    else:
        logger.info("Input must be an xarray.DataArray or numpy.ndarray")


def main():
    # Main processor
    for year in range(INITIAL_YEAR, FINAL_YEAR + 1):
        logger.info(f" >>> START Processing year {year} ")

        # 1. Read ETL settings
        ETL_SETTINGS = read_global_settings(
            filepath_configuration_file=FILEPATH_ETL_SETTINGS
        )

        # 2. Assemble paths
        paths = assemble_filepaths_no_year(
            **ETL_SETTINGS
        )  # Use this with the dummy data

        # paths = assemble_filepaths( # Use this with the real sample datasets
        #     year, **ETL_SETTINGS
        # )

        # 3. Verify if all the files exist for a given year
        if check_all_paths_exist(path_dict=paths) is False:
            logger.info(f"Year {year} could not be processed.")
            logger.info(f" >>> END Processing year {year} \n")
            continue

        # 4. Load all data
        model_data = load_all_data(paths=paths, etl_settings=ETL_SETTINGS)
        logger.info(model_data)

        # ----------------------------------------------------------------------------
        # ------    Entering to the first level: `model_run` function (octave)   ------
        # ----------------------------------------------------------------------------
        # [OK] ---- Verify carrying capacity function
        CC = carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        logger.debug(f"Dim. Carrying capacity data: {CC.values.shape}")
        # logger.debug(f"Carrying capacity data: \n{print_time_slices(CC)}")

        # [OK] ---- Verify water hatching function.
        egg_active = water_hatching(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        logger.debug(f"Dim. Egg active data: {egg_active.values.shape}")
        # logger.debug(f"Egg active data: \n{print_time_slices(egg_active)}")

        # [OK] ---- Verify initial conditions. OK
        logger.debug(
            f"Dim. initial conditions: {model_data.initial_conditions.values.shape}"
        )
        # logger.debug(
        #    f"Egg active data: \n{print_time_slices(model_data.initial_conditions)}"
        # )

        # ----------------------------------------------------------------------------------
        # ------    Entering to the second level: `call_function` function (octave)   ------
        # ----------------------------------------------------------------------------------
        #  [OK] ---- Verify diapause lay.
        diapause_lay = mosq_dia_lay(
            temperature=model_data.temperature_mean,
            latitude=model_data.latitude,
        )
        logger.info(f"Dim. Diapause laying data: {diapause_lay.values.shape}")
        # logger.debug(f"Diapause laying data: \n{print_time_slices(diapause_lay)}")

        #  [OK] ---- Verify diapause hatch.
        diapause_hatch = mosq_dia_hatch(
            temperature=model_data.temperature_mean,
            latitude=model_data.latitude,
        )
        logger.debug(f"Dim. Diapause hatching data: {diapause_hatch.values.shape}")
        # logger.debug(f"Diapause hatching data: \n{print_time_slices(diapause_hatch)}")

        #  [OK] ---- Verify ed survival
        ed_survival = mosq_surv_ed(
            temperature=model_data.temperature,
            step_t=ETL_SETTINGS["ode_system"]["time_step"],
        )
        logger.debug(f"Dim. ED survival data: {ed_survival.values.shape}")
        # logger.debug(f"ED survival data: \n{print_time_slices(ed_survival)}")

        # Assign this variable to maintain compatibility with the octave code. Just a rename.
        Temp = model_data.temperature
        step_t = ETL_SETTINGS["ode_system"]["time_step"]

        # Create output array
        shape_output = (
            model_data.initial_conditions.shape[0],
            model_data.initial_conditions.shape[1],
            5,
            int(Temp.shape[2] / step_t),
        )
        v_out = np.zeros(shape=shape_output, dtype=np.float64)
        logger.debug(f"Shape v_out:{v_out.shape}")

        # Initialize state variable
        v0 = model_data.initial_conditions.compute().values
        v = v0.copy()

        # Time loop (it will iterate over all time steps)
        for t in range(model_data.temperature.shape[2]):
            # if t == 30: # just for debugging purposes
            #    break

            logger.info(f"--- Time step {t} ---")

            # Line a. Verify this slice
            T = Temp[:, :, t]
            logger.debug(f"Dim. Temperature slice at time {t}: {T.values.shape}")
            logger.debug(f"Temperature slice at time {t}:\n{T.values}")

            # Line b. Verify birth rate
            birth = mosq_birth(T)
            logger.debug(f"Dim. Birth rate at time {t}: {birth.values.shape}")
            logger.debug(f"Birth rate at time {t}:\n{birth.values}")

            # Line c. Verify dev_j rate
            dev_j = mosq_dev_j(T)
            logger.debug(f"Dim. dev_j rate at time {t}: {dev_j.values.shape}")
            logger.debug(f"dev_j rate at time {t}:\n{dev_j.values}")

            # Line d. Verify dev_i rate
            dev_i = mosq_dev_i(T)
            logger.debug(f"Dim. dev_i rate at time {t}: {dev_i.values.shape}")
            logger.debug(f"dev_i rate at time {t}:\n{dev_i.values}")

            # Line e. Verify dev_e rate
            # dev_e = mosq_dev_e(T)
            dev_e = xr.DataArray(
                np.array([1.0 / 7.1]), dims=["dev_e"]
            )  # original model
            logger.debug(f"Dim. dev_e rate at time {t}: {dev_e.shape}")
            logger.debug(f"dev_e rate at time {t}:\n{dev_e}")

            # Line f. Verify dia_lay slice
            idx_time = int(np.ceil((t + 1) / step_t)) - 1
            logger.debug(f"Idx_time at time {t}: {idx_time}")

            dia_lay = diapause_lay[:, :, idx_time]
            logger.debug(f"Dim. Diapause laying slice at time {t}: {dia_lay.shape}")
            logger.debug(f"Diapause laying slice at time {t}:\n{dia_lay.values}")

            # Line g. Verify dia_hatch slice
            dia_hatch = diapause_hatch[:, :, idx_time]
            logger.debug(f"Dim. Diapause hatching slice at time {t}: {dia_hatch.shape}")
            logger.debug(f"Diapause hatching slice at time {t}:\n{dia_hatch.values}")

            # Line h. Verify ed_surv slice
            ed_surv = ed_survival[:, :, t]
            logger.debug(f"Dim. ED survival slice at time {t}: {ed_surv.shape}")
            logger.debug(f"ED survival slice at time {t}:\n{ed_surv.values}")

            # Line i. Verify water_hatch slice
            water_hatch = egg_active[:, :, idx_time]
            logger.debug(f"Dim. Water hatching slice at time {t}: {water_hatch.shape}")
            logger.debug(f"Water hatching slice at time {t}:\n{water_hatch}")

            # Line j. Verify mort_e rate
            mort_e = mosq_mort_e(T)
            logger.debug(f"Dim. mort_e rate at time {t}: {mort_e.values.shape}")
            logger.debug(f"mort_e rate at time {t}:\n{mort_e.values}")

            # Line k. Verify mort_j rate
            mort_j = mosq_mort_j(T)
            logger.debug(f"Dim. mort_j rate at time {t}: {mort_j.values.shape}")
            logger.debug(f"mort_j rate at time {t}:\n{mort_j.values}")

            # Line l. Verify T slice
            Tmean_slice = model_data.temperature_mean[:, :, idx_time]
            logger.debug(f"Dim. Tmean slice at time {t}: {Tmean_slice.values.shape}")
            logger.debug(f"Tmean slice at time {t}:\n{Tmean_slice.values}")

            # Line m. Verify mort_a rate
            mort_a = mosq_mort_a(Tmean_slice)
            logger.debug(f"Dim. mort_a rate at time {t}: {mort_a.values.shape}")
            # logger.debug(f"mort_a rate at time {t}:\n{mort_a.values}")

            # Line n. Verify the variables that will passed to the ODE solver step (all numpy arrays)
            vars_tuple = (
                idx_time,  # Octave uses 1-based, so pass idx_time+1
                step_t,
                CC.compute().values,
                birth.compute().values,
                dia_lay.compute().values,
                dia_hatch.compute().values,
                mort_e.compute().values,
                mort_j.compute().values,
                mort_a.compute().values,
                ed_surv.compute().values,
                dev_j.compute().values,
                dev_i.compute().values,
                dev_e.compute().values,
                water_hatch.compute().values,
            )

            logger.debug(f"Vars tuple length at time {t}: {len(vars_tuple)}")
            for i, var in enumerate(vars_tuple):
                logger.debug(f"  Var {i} shape: {getattr(var, 'shape', None)}")

            # Line o. Call the ODE solver step
            v = rk4_step(
                albopictus_ode_system,
                albopictus_log_ode_system,
                v,
                vars_tuple,
                step_t,
            )

            # logger.debug(f"Shape after rk4_step at time {t}: {v.shape}")
            logger.debug(f"Array: {v}")
            # logger.debug(f"Value after rk4_step at time {t}:\n{print_time_slices(v)}")

            # # Zero compartment 2 (Python index 1) if needed
            if (t / step_t) % 365 == 200:
                v[..., 1] = 0

            if (t + 1) % step_t == 0:
                if ((idx_time) % 30) == 0:
                    logger.debug(f"MOY: {int(((t) / step_t) / 30)}")
                for j in range(5):
                    v_out[..., j, idx_time] = np.maximum(v[..., j], 0)

        # logger.debug(f" >>> END Processing year {year} \n")
        logger.info(f"Shape of final output v_out for year {year}: {v_out.shape}")
        logger.debug(
            f"Adult array in output v_out for year {year}:\n{print_time_slices(v_out[:, :, 4, :])}"
        )

        compartments = ETL_SETTINGS["ode_system"]["model_variables"]

        # Create a dict of DataArrays, one for each compartment
        data_vars = {}
        for i, name in enumerate(compartments):
            logger.debug(f"Processing compartment: {name}")
            logger.debug(f"Processing slot: {i}")
            logger.debug(f"v_out shape: {v_out.shape}")
            data_vars[name] = xr.DataArray(
                data=v_out[..., i, :],  # shape: (longitude, latitude, time)
                dims=("longitude", "latitude", "time"),
                coords={
                    "longitude": model_data.temperature_mean["longitude"],
                    "latitude": model_data.temperature_mean["latitude"],
                    "time": model_data.temperature_mean["time"],
                },
                name=name,
            )

        # Combine into a Dataset
        v_ds = xr.Dataset(data_vars)

        # Save to NetCDF if desired
        v_ds.to_netcdf(f"data/out/mosquito_population_year_{year}.nc")

        logger.info(f" >>> END Processing year {year} \n")


if __name__ == "__main__":
    # a more complete loggger
    # logging.basicConfig(
    #    format="{asctime} {name}  {levelname} - {message}",
    #    style="{",
    #    datefmt="%Y-%m-%d %H:%M",
    #    level=logging.INFO,
    # )

    # basic logger
    logging.basicConfig(level=logging.INFO)

    main()
