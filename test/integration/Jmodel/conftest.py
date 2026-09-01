from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download, list_repo_files

from heiplanet_models import utils

# Fixture data published by heiplanet-data's own integration tests.
HF_INPUT_REPO = "iulusoy/heiplanet-data-silver"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def script_jmodel(repo_root):
    """The `script_Jmodel` driver script, loaded by path since it lives
    outside the installed `heiplanet_models` package (see
    `src/script_Jmodel.py`)."""
    return utils.load_module(
        "script_Jmodel", str(repo_root / "src" / "script_Jmodel.py")
    )


@pytest.fixture
def hf_input_data(tmp_path: Path) -> Path:
    """Download the "2t" fixture file(s) from the `heiplanet-data-silver`
    Hugging Face dataset into a local directory, mirroring what
    `run.data_folder` points at in production.
    """
    data_folder = tmp_path / "silver"
    data_folder.mkdir()

    files = [
        f for f in list_repo_files(HF_INPUT_REPO, repo_type="dataset") if "2t" in f
    ]
    if not files:
        pytest.fail(
            f"No '2t' files found in the {HF_INPUT_REPO} Hugging Face dataset; "
            "the upstream fixture data may have moved or been renamed."
        )

    for filename in files:
        downloaded = hf_hub_download(
            repo_id=HF_INPUT_REPO, repo_type="dataset", filename=filename
        )
        (data_folder / filename).write_bytes(Path(downloaded).read_bytes())

    return data_folder
