import heiplanet_models as mb
from pathlib import Path
from datetime import datetime
import json

data_path = Path("../heiplanet-data/.data_heiplanet_db/silver/")
ro_file = "./data/in/R0_pip_stats.csv"


if __name__ == "__main__":
    # get all files in the data_path
    files = list(data_path.glob("*.nc"))
    # keep only the ones with t2m in the file name
    files = [f for f in files if "2t" in f.name]
    outpath = Path("data") / "out" / datetime.today().strftime("%Y-%m-%d")
    outpath.mkdir(parents=True, exist_ok=True)
    with open("./src/heiplanet_models/config_Jmodel.json", "r") as f:
        global_config = json.load(f)
    run_mode = "parallelized"
    grid_data_baseurl = None
    nuts_level = None
    resolution = None
    year = None
    global_config["graph"]["setup_modeldata"]["kwargs"]["r0_path"] = str(ro_file)
    global_config["graph"]["setup_modeldata"]["kwargs"]["run_mode"] = run_mode
    global_config["graph"]["setup_modeldata"]["kwargs"]["grid_data_baseurl"] = (
        grid_data_baseurl
    )
    global_config["graph"]["setup_modeldata"]["kwargs"]["year"] = year

    for file in files:
        print(file)
        # get the base file name without the ending
        data_file = file.name
        data_file = data_file.rsplit(".", 1)[0]
        output = outpath / (data_file + "_output_JModel_global.nc")
        global_config["graph"]["setup_modeldata"]["kwargs"]["output"] = str(output)
        global_config["graph"]["setup_modeldata"]["kwargs"]["input"] = str(file)
        with open(outpath / (data_file + "_config.json"), "w") as f:
            json.dump(global_config, f)
        computation_global = mb.computation_graph.ComputationGraph(global_config)
        computation_global.execute()
