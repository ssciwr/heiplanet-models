"""Integration test for `script_Jmodel` against real Hugging Face data.

This downloads the ERA5-Land "2t" fixture data published by heiplanet-data's
own integration tests to the `iulusoy/heiplanet-data-silver` Hugging Face
dataset repo, runs the Jmodel DAG over it end-to-end via
`script_Jmodel.main()`, and publishes the resulting output to the
`iulusoy/heiplanet-models-dataset` Hugging Face dataset repo, so that
downstream consumers can pull fixed, real output data instead of
recomputing it.

Marked `hf_integration` (not the local-only `integration` marker used
elsewhere in this suite) since it needs network access; it is excluded from
the default `main.yml` CI run and only run by the `integration.yml`
workflow. Uploading needs an `HF_TOKEN` with write access to
`iulusoy/heiplanet-models-dataset`; it is skipped (not failed) when no
token is available, e.g. for a contributor running integration tests
locally without one.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import xarray as xr
from huggingface_hub import HfApi

from heiplanet_models import utils

pytestmark = pytest.mark.hf_integration

# Downstream consumers pull the model output fixture data from here.
HF_OUTPUT_REPO = "iulusoy/heiplanet-models-dataset"


@pytest.fixture
def jmodel_config(
    hf_input_data: Path, repo_root: Path, tmp_path: Path
) -> dict[str, Any]:
    config = utils.load_config()
    config["run"] = {
        "data_folder": str(hf_input_data),
        "file_match": "2t",
        "r0_path": str(repo_root / "test" / "test_r0.csv"),
        "output_folder": str(tmp_path / "gold"),
        "run_mode": "parallelized",
        "grid_data_baseurl": None,
        "year": None,
        "skip_existing": True,
    }
    config["execution"]["scheduler"] = "synchronous"
    return config


def test_main_runs_jmodel_on_hf_silver_data(
    script_jmodel, jmodel_config: dict[str, Any]
) -> None:
    script_jmodel.main(jmodel_config)

    output_folder = Path(jmodel_config["run"]["output_folder"])
    output_files = list(output_folder.glob("*_output_JModel_global.nc"))
    assert len(output_files) >= 1, (
        "Jmodel should have produced at least one output file"
    )

    for output_file in output_files:
        with xr.open_dataset(output_file) as ds:
            assert "R0" in ds.data_vars
            assert ds.R0.dims == ("time", "latitude", "longitude")
            # the HF fixture area is a slice of open ocean (ERA5-Land has no
            # data there), so t2m and R0 are legitimately all-NaN here; just
            # check the model didn't produce anything out of range
            assert bool(((ds.R0 >= 0) | ds.R0.isnull()).all())

    # publish the computed output for downstream consumers' own integration
    # tests; skip if no token is configured (e.g. running locally without
    # HF credentials)
    token = get_token()
    if not token:
        pytest.skip("HF_TOKEN not set; skipping upload to Hugging Face")

    upload_to_huggingface(
        folder_path=output_folder,
        repo_id=HF_OUTPUT_REPO,
        token=token,
    )


def upload_to_huggingface(
    folder_path: Path,
    repo_id: str,
    token: str | None = None,
) -> str:
    """Upload a local folder to a Hugging Face Hub dataset repo.

    Args:
        folder_path (Path): Path to the local folder to upload.
        repo_id (str): Target dataset repo, as "<namespace>/<name>"
            (e.g. "iulusoy/heiplanet-models-dataset").
        token (str | None): Hugging Face access token with write access to
            `repo_id`. Defaults to None, which uses the `HF_TOKEN`
            environment variable or a cached `huggingface-cli login`.

    Returns:
        str: URL of the uploaded folder on the Hugging Face Hub.
    """
    if not folder_path or not Path(folder_path).exists():
        raise ValueError(f"Folder {folder_path} must exist to be uploaded.")

    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("Repo id must be a non-empty string.")

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}/blob/main"
    print(f"Uploaded {folder_path} to {url}")
    return url


def get_token() -> str | None:
    """Get a Hugging Face access token from the environment or cached login.

    Returns:
        str | None: The token, or None if not found.
    """
    return os.environ.get("HF_TOKEN")
