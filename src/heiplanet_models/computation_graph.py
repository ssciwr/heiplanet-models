"""
This module defines the ComputationGraph class. This class represents a model as a directed acyclic graph (DAG) that executes a series of interdependent tasks which together represent the run of a given heiplanet model and manages the setup and execution of such a graph.
"""

# compatability with python 3.10+
from __future__ import annotations
import importlib

# dask needed for parallel execution and lazy evaluation
import dask
from dask.delayed import Delayed

# stdlib imports
import distributed
import json
from typing import Callable, Any
from pathlib import Path
import logging

from pydantic import validate_call

# local imports
from .verify import ComputationGraphConfig


class ComputationGraph:
    """A class to represent a computation DAG that executes a series of tasks which together represent the run of a given heiplanet model. These models are defined as combinations of functions known to the class.
    Modules are a loose collection of functions that are registered with the class and combined into a computational graph to create a functional system. Therefore, functions are registered as either part of a module or as utility functions, e.g., if they are used by multiple modules. The computational graph is built from these functions and executed in via dask tasks to allow for parallel, lazy execution and efficient resource management. Computations can be combined freely from the functions registered with different modules.

    Attributes:
        modules (dict[str, Any]): A dictionary of modules, where each module is a module object imported from a given path.
        module_functions (dict[str, dict[str, Callable]]): A dictionary mapping module names to dictionaries of function names and their corresponding callable objects.
        task_graph (dict[str, Delayed]): A dictionary representing the Dask computational graph, where each node is a dask.delayed object.
        config (dict[str, Any]): A configuration dictionary for the computation, the computational graph structure.
        sink_node (Delayed | None): The sink node of the computational graph, which is the final node that triggers the execution of the entire computation.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the computation graph from the given configuration.
        This method verifies the configuration, loads the necessary modules, retrieves the functions from the modules, builds the computational graph, and sets the Dask scheduler.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
        """
        conf = self._verify_config(config)
        self.config = conf

        self.logger = logging.getLogger("ComputationGraph")
        self.logger.setLevel(conf.execution.log_level.upper())
        # load needed code.
        self.module_functions = self._get_functions_from_module(config)

        # build the computational graph and find the sink node which we use to execute the graph
        self.sink_node_name = self._find_sink_node(config)
        self.task_graph = self._build_dag(config)
        self.sink_node = self.task_graph[self.sink_node_name]

        # set the dask scheduler
        self.scheduler = conf.execution.scheduler

    def _get_functions_from_module(
        self, config: dict[str, Any]
    ) -> dict[str, dict[str, Callable]]:
        """Find all functions in the given modules and return a dictionary mapping module names to dictionaries of function names and their corresponding callable objects.

        Args:
            config (dict[str, Any]): A configuration dictionary containing the computational graph structure in which function names and modules they live in are defined.

        Returns:
            dict[str, dict[str, Callable]]: A dictionary mapping module names to dictionaries of function names and their corresponding callable objects.
        """
        functions = {}

        for _, spec in config["graph"].items():
            module_path = Path(spec["module"]).resolve().absolute()
            modulename = module_path.stem
            funcname = spec["function"]
            try:
                module = importlib.import_module(modulename)
            except Exception as e:
                raise ValueError(f"Importing module {modulename} unsuccessful") from e

            try:
                func = getattr(module, funcname)
            except Exception as e:
                raise ValueError(
                    f"Could not load name {funcname} from {modulename}"
                ) from e

            if func not in functions:
                functions[f"{modulename}.{funcname}"] = func

        return functions

    def _find_sink_node(self, config: dict[str, Any]) -> str:
        """Find the sink node in the computational graph.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the computational graph structure.

        Raises:
            ValueError: If multiple sink nodes are found.
            ValueError: If no sink node is found.

        Returns:
            str: The name of the sink node.
        """
        all_inputs = []

        for node in config["graph"].values():
            if "input" in node and isinstance(node["input"], list):
                all_inputs.extend(node["input"])
        all_inputs = set(all_inputs)

        sink_node = None
        for node_name, _ in config["graph"].items():
            if node_name not in all_inputs:
                if sink_node is not None:
                    raise ValueError(
                        f"Multiple sink nodes found in the computational graph: {sink_node}, {node_name}."
                    )
                sink_node = node_name

        if sink_node is None:
            raise ValueError("No sink node found in the computational graph.")

        self.logger.debug(f"Sink node found: {sink_node}")
        return sink_node

    def _build_dag(self, config: dict[str, Any]) -> dict[str, Delayed]:
        """Build the Dask computational graph from the configuration.
            The returned executable graph is a dictionary mapping node names to Dask delayed objects, and the returned sink node is the final node in the graph that triggers the execution of the entire computation. There must only be one sink node in the graph.
            If there are multiple sink nodes, an error is raised.
        Args:
            config (dict[str, Any]): Configuration dictionary containing the computational graph structure as name: {
                "function": function_name,
                "input": [input_node_name1, input_node_name2, ...],
                "args": [args],
                "kwargs": {kwargs},
                "module": module_name
            }

        Raises:
            ValueError: A module is not found in the registered modules.
            ValueError:  A function is not found in the module.

        Returns:
            tuple[dict[str, Delayed], Delayed]: A tuple containing the Dask computational graph and the sink node.

        """

        # recursive function that progressively works it's way up the computational graph from the sink node until it reaches source nodes that have no dependencies, and then down again to build the computation tasks in the graph from sources to sink.
        def build_node(node_name, delayed_tasks):
            """Build a single node in the Dask computational graph."""
            self.logger.debug(f" Building input node: {node_name}")

            if node_name in delayed_tasks:
                self.logger.debug(f" Node already defined: {node_name}")
                return

            node_spec = config["graph"][node_name]
            input_nodes = node_spec["input"]
            args = node_spec["args"] if node_spec["args"] is not None else []
            kwargs = node_spec["kwargs"] if node_spec["kwargs"] is not None else {}

            # move up the DAG to build the input nodes first. Since none are defined initially, this will eventually reach the source node.
            for input_node in input_nodes:
                build_node(input_node, delayed_tasks)

            # when the input nodes are all defined, we can create the task
            # for the source node there are no input nodes, so we can create the task immediately
            # Check if all input nodes are already defined
            if (
                all(input_node in delayed_tasks for input_node in input_nodes)
                or len(input_nodes) == 0
            ):
                self.logger.debug(f" Creating task for node: {node_name}")
                # All input nodes are already defined, so we can create the task
                module_name = str(Path(node_spec["module"]).stem)
                func = self.module_functions[module_name][node_spec["function"]]
                delayed_tasks[node_name] = dask.delayed(func)(
                    *[delayed_tasks[input_node] for input_node in input_nodes],
                    *args,
                    **kwargs,
                )
            # no else because on the way down the graph we will never encounter this case.

        delayed_tasks = {}
        build_node(self.sink_node_name, delayed_tasks)
        self.logger.debug(f"build_graph: {delayed_tasks.keys()}")
        return delayed_tasks

    @validate_call
    def _verify_config(self, cfg: dict[str, Any]) -> ComputationGraphConfig:
        return ComputationGraphConfig(**cfg)

    def execute(self, client: distributed.client.Client = None):
        """Executes the computational graph.

        Args:
            client (distributed.client.Client, optional): The client to use for execution if the computation should be executed on a cluster. If None, will use the local machine. Defaults to None. For more on how to use the client, see https://distributed.dask.org/en/stable/client.html.

        Raises:
            ValueError: If the sink node is not defined.
            ValueError: If the scheduler is not defined.

        Returns:
            Any: The result of the computation.
        """

        return self.sink_node.compute(scheduler=self.scheduler, client=client)

    def visualize(self, filename: str):
        """Visualizes the computational graph.

        Raises:
            ValueError: If the sink node is not defined.

        Returns:
            Any: The visualization of the sink node as returned by the Delayed.visualize method.
        """
        if self.sink_node is None:
            raise ValueError("Sink node is not defined. Cannot visualize the graph.")
        return self.sink_node.visualize(
            filename=str(filename),
            optimize_graph=False,  # Shows the full graph structure
            rankdir="TB",  # Top to bottom layout
        )

    @classmethod
    def from_config(cls, path_to_config: str | Path) -> "ComputationGraph":
        """Creates a `ComputationGraph` instance from a configuration dictionary read from a json file."""
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return cls(config)
