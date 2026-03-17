import logging


from heiplanet_models.Pmodel.Pmodel_initial import (
    read_global_settings,
    assemble_filepaths,
    check_all_paths_exist,
    load_all_data,
)

from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    water_hatching,
)

from heiplanet_models.Pmodel.Pmodel_rates_development import (
    carrying_capacity,
)

from heiplanet_models.Pmodel.Pmodel_ode import solve_system

from heiplanet_models.Pmodel.Pmodel_output import (
    build_output_dataset,
    save_output_dataset,
)

# ---- Logger
logger = logging.getLogger(__name__)

FILEPATH_ETL_SETTINGS = "./src/heiplanet_models/Pmodel/global_settings_dummy.yaml"
INITIAL_YEAR = 2024
FINAL_YEAR = 2024


def configure_logging(level: int = logging.DEBUG) -> None:
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)-5s] [%(name)-25s] > %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=level,
        force=True,
    )


def main():

    # Main processor
    for year in range(INITIAL_YEAR, FINAL_YEAR + 1):
        logger.info(f" >>> START Processing year {year} ")

        # 1. Read ETL settings
        ETL_SETTINGS = read_global_settings(
            filepath_configuration_file=FILEPATH_ETL_SETTINGS
        )

        # 2. Assemble paths
        paths = assemble_filepaths(year=None, **ETL_SETTINGS)  # OK

        # 3. Verify if all the files exist for a given year
        if check_all_paths_exist(path_dict=paths) is False:
            logger.info(f"Year {year} could not be processed.")
            logger.info(f" >>> END Processing year {year} \n")
            continue

        # 4. Load all data
        model_data = load_all_data(paths=paths, etl_settings=ETL_SETTINGS)
        logger.info(model_data)

        # 5. Calculate carrying capacity
        CC = carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )

        # 6. Calculate water hatching
        egg_active = water_hatching(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )

        # 7. Solve ODE system
        ode_solution = solve_system(
            state=model_data.initial_conditions,
            temperature=model_data.temperature,
            temperature_mean=model_data.temperature_mean,
            latitudes=model_data.latitude,
            carrying_capacity=CC,
            egg_activate=egg_active,
            time_step=ETL_SETTINGS["ode_system"]["time_step"],
        )
        logger.info("Model output shape: %s", ode_solution.shape)
        logger.info(ode_solution[:, :, 4, 3])

        # 8. Build output dataset
        output_dataset = build_output_dataset(
            state=ode_solution,
            model_data=model_data,
            compartments=ETL_SETTINGS["ode_system"]["model_variables"],
        )

        # 9. Save to NetCDF if desired
        output_path = save_output_dataset(
            dataset=output_dataset,
            year=year,
            **ETL_SETTINGS["serving"],
        )
        logger.info("Saved output dataset to: %s", output_path.resolve())

        # 10. END processing year
        logger.info(f" >>> END Processing year {year} \n")


if __name__ == "__main__":
    configure_logging(level=logging.INFO)

    main()
