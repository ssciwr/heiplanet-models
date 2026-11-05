import argparse
import copy
import shutil
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from rich.console import Console

from heiplanet_models.Pmodel.Pmodel_output import (
    assemble_output_filepath,
    final_state_filepath,
)
from scripts.pipeline_utils import StageRunner
from scripts.run_pmodel_pipeline import (
    DEFAULT_SCIPY_ATOL,
    DEFAULT_SCIPY_METHOD,
    DEFAULT_SCIPY_RTOL,
    resolve_initial_conditions,
    run_year,
)


def _settings(base: dict, **initial_conditions) -> dict:
    settings = copy.deepcopy(base)
    settings["ingestion"]["initial_conditions"] = {
        "file_path_initial_conditions": None,
        **initial_conditions,
    }
    # float64 output so that a file round trip does not change the state
    settings["serving"]["dtype"] = "f8"
    return settings


def _run(settings: dict, backend: str) -> Path:
    args = argparse.Namespace(
        backend=backend,
        chunk_lon=2,
        chunk_lat=2,
        scipy_method=DEFAULT_SCIPY_METHOD,
        scipy_rtol=DEFAULT_SCIPY_RTOL,
        scipy_atol=DEFAULT_SCIPY_ATOL,
    )
    return run_year(
        year=None,
        args=args,
        etl_settings=settings,
        stage=StageRunner(Console(file=StringIO())),
    )


def _write_file(path: Path, aggregation: str | None = None) -> Path:
    attrs = {} if aggregation is None else {"time_aggregation": aggregation}
    xr.Dataset({"adults": ("time", [1.0])}, attrs=attrs).to_netcdf(path)
    return path


def test_resolve_initial_conditions_prefers_final_state_file(
    integration_etl_settings, tmp_path
):
    configured = _write_file(tmp_path / "configured.nc")
    settings = _settings(
        integration_etl_settings,
        chain_previous_year=True,
        file_path_initial_conditions=str(configured),
    )
    serving = settings["serving"]

    assert resolve_initial_conditions(2015, settings) == configured

    # a previous daily output (no attribute = legacy daily file) is used
    previous_output = _write_file(assemble_output_filepath(year=2014, **serving))
    assert resolve_initial_conditions(2015, settings) == previous_output

    # the final state file takes precedence over the output
    previous_state = _write_file(final_state_filepath(year=2014, **serving))
    assert resolve_initial_conditions(2015, settings) == previous_state
    assert resolve_initial_conditions(None, settings) == configured


def test_resolve_initial_conditions_skips_monthly_output_without_state(
    integration_etl_settings,
):
    settings = _settings(integration_etl_settings, chain_previous_year=True)
    _write_file(
        assemble_output_filepath(year=2014, **settings["serving"]), "monthly_mean"
    )
    assert resolve_initial_conditions(2015, settings) is None


def test_resolve_initial_conditions_without_chaining(
    integration_etl_settings, tmp_path
):
    settings = _settings(integration_etl_settings, chain_previous_year=False)
    _write_file(final_state_filepath(year=2014, **settings["serving"]))
    assert resolve_initial_conditions(2015, settings) is None

    settings["ingestion"]["initial_conditions"]["file_path_initial_conditions"] = str(
        tmp_path / "missing.nc"
    )
    assert resolve_initial_conditions(2015, settings) is None


@pytest.fixture
def positive_rainfall_settings(integration_etl_settings, tmp_path):
    """Dummy inputs with positive rainfall, so carrying capacity is valid.

    The shared dummy rainfall is negative, which makes ``scipy_chunked`` skip
    every cell and would leave nothing to compare.
    """
    source = Path(integration_etl_settings["ingestion"]["path_root_datasets"])
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    for path in source.glob("dataset_*_dummy.nc"):
        shutil.copy(path, data_dir / path.name)
    rainfall_path = data_dir / "dataset_rainfall_dummy.nc"
    with xr.open_dataset(rainfall_path) as rainfall:
        positive = abs(rainfall).load()
    positive.to_netcdf(rainfall_path)

    settings = copy.deepcopy(integration_etl_settings)
    settings["ingestion"]["path_root_datasets"] = str(data_dir)
    return settings


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["scipy_chunked", "legacy_optimized"])
def test_spin_up_equals_restart_from_previous_output(
    positive_rainfall_settings, tmp_path, backend
):
    """One spin-up year must equal running twice with the first output as IC."""

    base = positive_rainfall_settings
    first = _run(_settings(base), backend)
    first_copy = tmp_path / "first_run.nc"
    shutil.copy(first, first_copy)

    restarted = _run(
        _settings(base, file_path_initial_conditions=str(first_copy)), backend
    )
    with xr.open_dataset(restarted) as ds:
        restarted_values = {name: ds[name].values.copy() for name in ds.data_vars}

    spun_up = _run(_settings(base, spinup_years=1), backend)
    with xr.open_dataset(spun_up) as ds, xr.open_dataset(first_copy) as cold:
        # the spun-up run must differ from the cold start in some compartment
        assert any(
            not np.allclose(ds[name].values, cold[name].values) for name in ds.data_vars
        )
        for name, values in restarted_values.items():
            np.testing.assert_allclose(ds[name].values, values, rtol=1e-12, atol=0)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["scipy_chunked", "legacy_optimized"])
def test_monthly_output_and_final_state(positive_rainfall_settings, tmp_path, backend):
    """Monthly output is the mean of the daily output; the state file holds
    the last day; restarting from the state file equals restarting from the
    daily output."""

    daily_settings = _settings(positive_rainfall_settings)
    daily_settings["serving"]["time_aggregation"] = "daily"
    daily_path = _run(daily_settings, backend)
    daily_copy = tmp_path / "daily.nc"
    shutil.copy(daily_path, daily_copy)

    monthly_settings = _settings(positive_rainfall_settings)
    monthly_settings["serving"]["time_aggregation"] = "monthly_mean"
    monthly_path = _run(monthly_settings, backend)
    state_path = final_state_filepath(year=None, **monthly_settings["serving"])

    with (
        xr.open_dataset(daily_copy) as daily,
        xr.open_dataset(monthly_path) as monthly,
        xr.open_dataset(state_path) as state,
    ):
        assert monthly.attrs["time_aggregation"] == "monthly_mean"
        expected = daily.resample(time="1MS").mean()
        assert monthly.sizes["time"] == expected.sizes["time"]
        np.testing.assert_array_equal(monthly.time.values, expected.time.values)
        assert any(np.any(daily[name].values > 0) for name in daily.data_vars)
        for name in daily.data_vars:
            assert monthly[name].attrs["cell_methods"] == "time: mean"
            np.testing.assert_allclose(
                monthly[name].transpose(*expected[name].dims).values,
                expected[name].values,
                rtol=1e-12,
            )
            np.testing.assert_array_equal(
                state[name].transpose(*daily[name].dims).values,
                daily[name].isel(time=[-1]).values,
            )

    from_state = _run(
        _settings(
            positive_rainfall_settings, file_path_initial_conditions=str(state_path)
        ),
        backend,
    )
    with xr.open_dataset(from_state) as ds:
        from_state_values = {name: ds[name].values.copy() for name in ds.data_vars}
    from_daily = _run(
        _settings(
            positive_rainfall_settings, file_path_initial_conditions=str(daily_copy)
        ),
        backend,
    )
    with xr.open_dataset(from_daily) as ds:
        for name, values in from_state_values.items():
            np.testing.assert_allclose(ds[name].values, values, rtol=1e-12, atol=0)
