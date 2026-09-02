import json
import logging
from pathlib import Path
from typing import Any

import heiplanet_models as mb
from heiplanet_models import utils

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)-5s] [%(name)-25s] > %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=level,
        force=True,
    )


def main(config: dict[str, Any] | None = None) -> None:
    """Run the Jmodel DAG for every matching input file.

    Args:
        config (dict[str, Any] | None): The config, in the shape of
            `config_Jmodel.json` (a "run" block with paths/run settings, plus
            "graph"/"execution" for `ComputationGraph`). Defaults to None,
            which loads the packaged `config_Jmodel.json`.
    """
    if config is None:
        config = utils.load_config()

    run_cfg = config["run"]
    data_path = Path(run_cfg["data_folder"])
    files = [f for f in data_path.glob("*.nc") if run_cfg["file_match"] in f.name]

    outpath = Path(run_cfg["output_folder"])
    outpath.mkdir(parents=True, exist_ok=True)

    if not files:
        logger.warning(
            "No input files found in %s matching %r", data_path, run_cfg["file_match"]
        )

    for file in files:
        stem = file.stem
        output = outpath / f"{stem}_output_JModel_global.nc"

        if run_cfg.get("skip_existing") and output.exists():
            logger.info("Skipping %s, output already exists at %s", file, output)
            continue

        logger.info("Processing %s", file)

        # deep copy so each run gets its own config, isolated from the next
        # iteration (and safe to hand to concurrent tasks later)
        graph_config = json.loads(json.dumps(config))
        graph_config["graph"]["setup_modeldata"]["kwargs"].update(
            input=str(file),
            output=str(output),
            r0_path=run_cfg["r0_path"],
            run_mode=run_cfg["run_mode"],
            grid_data_baseurl=run_cfg["grid_data_baseurl"],
            year=run_cfg["year"],
        )

        with open(outpath / f"{stem}_config.json", "w") as f:
            json.dump(graph_config, f)

        try:
            computation = mb.computation_graph.ComputationGraph(graph_config)
            computation.execute()
        except Exception:
            logger.exception("Failed processing %s", file)
            continue


if __name__ == "__main__":
    configure_logging()
    main()
