import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_output import save_output_dataset


@pytest.mark.integration
def test_run_model_pipeline_end_to_end(
    integration_etl_settings: dict, run_pipeline
) -> None:
    """Run the same pipeline stages as scripts/run_model.py on dummy resources."""
    output_dataset, temperature_mean = run_pipeline(integration_etl_settings)

    compartments = integration_etl_settings["ode_system"]["model_variables"]

    assert isinstance(output_dataset, xr.Dataset)
    assert set(output_dataset.data_vars) == set(compartments)

    n_lon = temperature_mean.sizes["longitude"]
    n_lat = temperature_mean.sizes["latitude"]
    n_time = temperature_mean.sizes["time"]

    for variable in compartments:
        data_array = output_dataset[variable]
        assert data_array.dims == ("longitude", "latitude", "time")
        assert data_array.shape == (n_lon, n_lat, n_time)
        assert np.isfinite(data_array.values).all()

    saved_path = save_output_dataset(
        dataset=output_dataset,
        year=2024,
        **integration_etl_settings["serving"],
    )

    assert saved_path.exists()

    with xr.open_dataset(saved_path) as reloaded:
        assert set(reloaded.data_vars) == set(compartments)
        assert reloaded.sizes["longitude"] == n_lon
        assert reloaded.sizes["latitude"] == n_lat
        assert reloaded.sizes["time"] == n_time


@pytest.mark.integration
@pytest.mark.regression
def test_run_model_pipeline_matches_golden_dataset(
    integration_etl_settings: dict,
    repo_root,
    run_pipeline,
    assert_allclose_arrays,
    transpose_to_match_dims,
) -> None:
    """Compare generated output to golden reference data for stable regression checks."""
    golden_path = repo_root / "test" / "test_resources" / "output_dataset_dummy.nc"

    output_dataset, _ = run_pipeline(integration_etl_settings)

    assert golden_path.exists()
    with xr.open_dataset(golden_path) as golden_dataset:
        # Golden data currently stores a single reference variable.
        assert "adults" in golden_dataset.data_vars
        assert "adults" in output_dataset.data_vars

        assert_allclose_arrays(
            output_dataset["longitude"].values,
            golden_dataset["longitude"].values,
            rtol=0.0,
            atol=0.0,
        )
        assert_allclose_arrays(
            output_dataset["latitude"].values,
            golden_dataset["latitude"].values,
            rtol=0.0,
            atol=0.0,
        )
        assert_allclose_arrays(
            output_dataset["time"].values,
            golden_dataset["time"].values,
            rtol=0.0,
            atol=0.0,
        )
        golden_adults = transpose_to_match_dims(
            reference=output_dataset["adults"],
            candidate=golden_dataset["adults"],
        )
        assert_allclose_arrays(
            output_dataset["adults"].values,
            golden_adults.values,
            rtol=1e-6,
            atol=1e-8,
        )
