import pytest
import xarray as xr


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

    output_path = run_pipeline(integration_etl_settings)

    assert golden_path.exists()
    with (
        xr.open_dataset(output_path) as output_dataset,
        xr.open_dataset(golden_path) as golden_dataset,
    ):
        compartments = integration_etl_settings["ode_system"]["model_variables"]
        assert set(output_dataset.data_vars) == set(compartments)
        for variable in compartments:
            assert output_dataset[variable].dims == (
                "longitude",
                "latitude",
                "time",
            )
            assert output_dataset[variable].notnull().all()

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
