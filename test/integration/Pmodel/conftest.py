import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from rich.console import Console

from heiplanet_models.Pmodel.Pmodel_initial import read_global_settings
from scripts.pipeline_utils import StageRunner
from scripts.run_pmodel_pipeline import (
    DEFAULT_CHUNK_LAT,
    DEFAULT_CHUNK_LON,
    DEFAULT_SCIPY_ATOL,
    DEFAULT_SCIPY_METHOD,
    DEFAULT_SCIPY_RTOL,
    run_year,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def integration_etl_settings(tmp_path: Path, repo_root: Path) -> dict:
    """Load ETL settings and redirect output to a temporary directory."""
    settings_path = repo_root / "test" / "test_resources" / "global_settings_dummy.yaml"
    etl_settings = read_global_settings(str(settings_path))

    # Make ingestion path absolute to keep tests independent from CWD.
    etl_settings["ingestion"]["path_root_datasets"] = str(
        repo_root / "test" / "test_resources"
    )

    # Write outputs into pytest temp directory.
    etl_settings["serving"]["path_output_datasets"] = str(tmp_path)
    etl_settings["serving"]["filename_components"]["prefix"] = "integration_output_"

    return etl_settings


@pytest.fixture
def run_pipeline():
    """Execute the current Pmodel script pipeline with legacy-compatible settings."""

    def _run(etl_settings: dict) -> Path:
        args = argparse.Namespace(
            backend="legacy_optimized",
            chunk_lon=DEFAULT_CHUNK_LON,
            chunk_lat=DEFAULT_CHUNK_LAT,
            scipy_method=DEFAULT_SCIPY_METHOD,
            scipy_rtol=DEFAULT_SCIPY_RTOL,
            scipy_atol=DEFAULT_SCIPY_ATOL,
        )
        return run_year(
            year=None,
            args=args,
            etl_settings=etl_settings,
            stage=StageRunner(Console(file=StringIO())),
        )

    return _run


@pytest.fixture
def assert_allclose_arrays():
    def _to_numeric_if_datetime(values: np.ndarray) -> np.ndarray:
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]").astype(np.int64)
        return values

    def _assert(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> None:
        assert a.shape == b.shape
        np.testing.assert_allclose(
            _to_numeric_if_datetime(np.asarray(a)),
            _to_numeric_if_datetime(np.asarray(b)),
            rtol=rtol,
            atol=atol,
        )

    return _assert


@pytest.fixture
def transpose_to_match_dims():
    def _transpose(reference: xr.DataArray, candidate: xr.DataArray) -> xr.DataArray:
        """Golden data stores dimensions in a different order; transpose to match output."""
        return candidate.transpose(*reference.dims)

    return _transpose
