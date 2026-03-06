from heiplanet_models import computation_graph as cg
from pathlib import Path
import pytest
import pandas as pd
import json
from dask.distributed import Client


def make_configfile_graph(tmp_path, computation_graph_working):
    with open(tmp_path / "computation_graph.json", "w") as f:
        json.dump(computation_graph_working, f)

    computation = cg.ComputationGraph.from_config(tmp_path / "computation_graph.json")
    return computation


def make_dict_graph(_, computation_graph_working):
    computation = cg.ComputationGraph(computation_graph_working)
    return computation


def test_computation_graph_invalid_config(computation_graph_invalid_highlevel):
    # test with an invalid config file.
    with pytest.raises(ValueError, match="Configuration verification failed:"):
        cg.ComputationGraph(computation_graph_invalid_highlevel)


def test_computation_graph_invalid_modules(computation_graph_invalid_modules):
    with pytest.raises(
        ValueError,
        match=f"Configuration verification failed: Module ./non_existent_module for node invalid_module at path {Path('./non_existent_module').resolve().absolute()} does not exist.",
    ):
        cg.ComputationGraph(computation_graph_invalid_modules)


def test_computation_graph_invalid_functions(computation_graph_invalid_func):
    with pytest.raises(
        RuntimeError,
        match="Failed to load function 'non_existent_function' from module 'computation_module': module 'computation_module' has no attribute 'non_existent_function'",
    ):
        cg.ComputationGraph(computation_graph_invalid_func)


def test_computation_graph_multiple_sink_nodes(computation_graph_multiple_sink_nodes):
    # test with a config file that has multiple sink nodes.
    # this should raise an error.
    with pytest.raises(
        ValueError,
        match="Multiple sink nodes found in the computational graph: add, multiply.",
    ):
        cg.ComputationGraph(computation_graph_multiple_sink_nodes)


def test_computation_misc_config_errors(computation_graph_working):
    del computation_graph_working["execution"]["scheduler"]

    with pytest.raises(
        ValueError,
        match="Configuration verification failed: Execution configuration is missing 'scheduler' key.",
    ):
        cg.ComputationGraph(computation_graph_working)

    computation_graph_working["execution"]["scheduler"] = "wrong_scheduler"
    with pytest.raises(
        ValueError,
        match="Configuration verification failed: Unsupported scheduler: wrong_scheduler. Supported schedulers are 'synchronous', 'threads', 'multiprocessing', or 'distributed'.",
    ):
        cg.ComputationGraph(computation_graph_working)

    # undo last change
    computation_graph_working["execution"]["scheduler"] = "synchronous"
    del computation_graph_working["graph"]["load_data"]["input"]

    with pytest.raises(
        ValueError,
        match="Configuration verification failed: Node load_data is missing required keys. Required keys are 'function', 'input', 'args', 'kwargs', and 'module'.",
    ):
        cg.ComputationGraph(computation_graph_working)


@pytest.mark.parametrize(
    "factory_function",
    [
        make_configfile_graph,
        make_dict_graph,
    ],
    ids=["from_configfile", "from_dict"],
)
def test_computation_graph_working_initialization(
    factory_function, tmp_path, computation_graph_working
):
    computation = factory_function(tmp_path, computation_graph_working)

    # check the properties of the computation graph

    # 1) has the correct sink node and nodes in general
    assert len(computation.task_graph) == 6
    assert computation.sink_node_name == "save"
    assert "load_data" in computation.task_graph
    assert "add" in computation.task_graph
    assert "multiply" in computation.task_graph
    assert "subtract" in computation.task_graph
    assert "affine" in computation.task_graph
    assert "save" in computation.task_graph

    # 2) knows about the correct module
    assert "computation_module" in computation.module_functions

    # 3) has known functions specified in the config
    assert "load_data" in computation.module_functions["computation_module"]
    assert "add" in computation.module_functions["computation_module"]
    assert "multiply" in computation.module_functions["computation_module"]
    assert "subtract" in computation.module_functions["computation_module"]
    assert "affine" in computation.module_functions["computation_module"]
    assert "save_data" in computation.module_functions["computation_module"]

    # 4) the task graph has the correct nodes and edges
    assert "load_data" in computation.task_graph
    assert "add" in computation.task_graph
    assert "multiply" in computation.task_graph
    assert "subtract" in computation.task_graph
    assert "affine" in computation.task_graph
    assert "save" in computation.task_graph

    # 5) loaded functions are callable
    f_dict = computation.module_functions["computation_module"]
    assert f_dict["add"](3, 5) == 8
    assert f_dict["multiply"](2, 4) == 8
    assert f_dict["subtract"](10, 2) == 8
    assert f_dict["affine"](8, b=5, a=2) == 21

    loaded_data = f_dict["load_data"](Path("./test") / "computation_test_data.csv")
    assert all(loaded_data == [1, 2, 3])

    f_dict["save_data"](
        pd.DataFrame({"idx": [0, 1, 2], "value": [1, 2, 3]}),
        Path(tmp_path / "output.csv"),
    )
    assert Path(tmp_path / "output.csv").exists()

    # 6) knows about the Jmodel and the utils modules
    assert "Jmodel" in computation.module_functions
    assert "utils" in computation.module_functions

    # 7) knows about the utils functions
    assert "load_module" in computation.module_functions["utils"]
    assert "load_name_from_module" in computation.module_functions["utils"]

    # 8) knows about the Jmodel functions
    assert "setup_modeldata" in computation.module_functions["Jmodel"]
    assert "read_input_data" in computation.module_functions["Jmodel"]
    assert "run_model" in computation.module_functions["Jmodel"]
    assert "store_output_data" in computation.module_functions["Jmodel"]


def test_computation_graph_visualization(computation_graph_working, tmp_path):
    computation = cg.ComputationGraph(computation_graph_working)
    computation.visualize(tmp_path / "computation_graph.svg")
    assert (tmp_path / "computation_graph.svg").exists()

    computation.sink_node = None

    with pytest.raises(
        ValueError, match="Sink node is not defined. Cannot visualize the graph."
    ):
        computation.visualize(tmp_path / "computation_graph.svg")


@pytest.mark.parametrize(
    "scheduler",
    [
        "synchronous",
        "threads",
        "multiprocessing",
    ],
)
def test_computation_graph_execution_schedulers(
    scheduler, computation_graph_working, tmp_path
):
    computation_graph_working["execution"]["scheduler"] = scheduler
    computation = cg.ComputationGraph(computation_graph_working)
    computation.execute()
    path = computation_graph_working["graph"]["save"]["args"][0]
    assert Path(path).exists()
    result = pd.read_csv(Path(path))
    assert all(result["value"] == [5, -3, -15])


def test_computation_graph_execution_distributed(computation_graph_working, tmp_path):
    computation_graph_working["execution"]["scheduler"] = "distributed"
    computation_graph_working["graph"]["save"]["args"] = [
        str(tmp_path / "output_distributed.csv"),
    ]
    client = Client(n_workers=2)
    computation = cg.ComputationGraph(computation_graph_working)
    computation.execute(client=client)
    path = computation_graph_working["graph"]["save"]["args"][0]
    assert Path(path).exists()
    result = pd.read_csv(Path(path))
    assert all(result["value"] == [5, -3, -15])
