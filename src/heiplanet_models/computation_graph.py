"""
This module defines the ComputationGraph class. This class represents a model as a directed acyclic graph (DAG) that executes a series of interdependent tasks which together represent the run of a given heiplanet model and manages the setup and execution of such a graph.
"""

# compatability with python 3.10+
from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

# dask needed for parallel execution and lazy evaluation
import dask

# stdlib imports
import distributed
from dask.delayed import Delayed

# internals
from . import Jmodel, utils


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

    module_functions: dict[str, dict[str, Callable]]
    config: dict[str, Any] = None  # Configuration for the computation
    sink_node: Delayed | None = None  # The sink node of the computational graph
    task_graph: dict[str, Delayed] | None = None
    sink_node_name: str | None = None
    scheduler: str | None = None  # The Dask scheduler to use for execution
    default_modules: ClassVar[set[str]] = {
        "utils",
        "Jmodel",
    }  # README: we need a better way to manage default modules

    def __init__(self, config: dict[str, Any]):
        """Initialize the computation graph from the given configuration.
        This method verifies the configuration, loads the necessary modules, retrieves the functions from the modules, builds the computational graph, and sets the Dask scheduler.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
        """
        config_valid, msg = self._verify_config(config)

        if not config_valid:
            raise ValueError(f"Configuration verification failed: {msg}")

        self.config = config

        self.logger = logging.getLogger("ComputationGraph")
        self.logger.setLevel(config["execution"].get("log_level", logging.DEBUG))
        # load needed code.
        self.module_functions = self._get_functions_from_module(config)

        # build the computational graph and find the sink node which we use to execute the graph
        self.sink_node_name = self._find_sink_node(config)
        self.task_graph = self._build_dag(config)
        self.sink_node = self.task_graph[self.sink_node_name]

        # set the dask scheduler
        self.scheduler = config["execution"]["scheduler"]

    def _get_functions_from_module(
        self, config: dict[str, Any]
    ) -> dict[str, dict[str, Callable]]:
        """Find all functions in the given modules and return a dictionary mapping module names to dictionaries of function names and their corresponding callable objects.

        Args:
            config (dict[str, Any]): A configuration dictionary containing the computational graph structure in which function names and modules they live in are defined.

        Returns:
            dict[str, dict[str, Callable]]: A dictionary mapping module names to dictionaries of function names and their corresponding callable objects.
        """
        module_functions = {}
        for name, spec in config["graph"].items():
            module_path = Path(spec["module"]).resolve().absolute()
            module_name = module_path.stem
            if module_name in self.default_modules:
                continue
            function_name = spec["function"]
            try:
                func = utils.load_name_from_module(
                    module_name=module_name,
                    file_path=module_path,
                    name=function_name,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load function '{function_name}' from module '{module_name}': {e}"
                ) from e

            if module_name not in module_functions:
                module_functions[module_name] = {}
            module_functions[module_name][function_name] = func

        # add the default modules and utility functions needed
        # README: this needs to be generalized later when we have a more stable
        # way of handling model code
        module_functions["Jmodel"] = {}
        module_functions["utils"] = {}

        for module in [utils, Jmodel]:
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__ and name[0] != "_":
                    module_functions[module.__name__.split(".")[-1]][name] = obj

        return module_functions

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
        for node_name in config["graph"]:
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

    def _verify_computation_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Verify the configuration of the computational graph.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Returns:
            bool, str: A tuple containing a boolean indicating whether the configuration is valid and an error message if it is not.
        """
        # verify the computation structure.
        for node, value in config.items():
            # verify that the node is a dict
            if value is None or not isinstance(value, dict):
                return False, f"Node {node} is not a dict."

            # all nodes that define a computation node must have the name of the function to call, a list of nodes that need to run before this one and that are used as inputs, as well as additional arguments and keyword arguments that are passed to the function. The latter two can be empty or None, but the keys must be present to make this choice explicit and distinguish it from having forgotten to specify them.
            if any(
                key not in value
                for key in ["function", "input", "args", "kwargs", "module"]
            ):
                return (
                    False,
                    f"Node {node} is missing required keys. Required keys are 'function', 'input', 'args', 'kwargs', and 'module'.",
                )

            # check that the module path exists and is a valid file
            if (
                str(Path(value["module"]).stem) not in self.default_modules
                and Path.exists(Path(value["module"]).resolve().absolute()) is False
            ):
                module_name = value["module"]
                return (
                    False,
                    f"Module {module_name} for node {node} at path {Path(value['module']).resolve().absolute()} does not exist.",
                )

            # the input nodes must be a list of names of other nodes
            if not isinstance(value["input"], list):
                return False, f"input nodes for node {node} must be a list"

            # the input nodes must be explicitly specified and must be present # in the graph somewhere, otherwise we cannot resolve them.
            for input_node in value["input"]:
                if input_node not in config:
                    return (
                        False,
                        f"input node {input_node} of node {node} not found in graph",
                    )

            # the positional arguments and keyword arguments must be a list and a dict, respectively, or None
            if not isinstance(value["args"], list) and value["args"] is not None:
                return False, f"arguments for node {node} must be a list"

            if not isinstance(value["kwargs"], dict) and value["kwargs"] is not None:
                return False, f"keyword arguments for node {node} must be a dict"

        return True, "Configuration is valid."

    def _verify_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Verify the configuration dictionary.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Returns:
            bool, str: A tuple containing a boolean indicating whether the configuration is valid and an error message if it is not.
        """

        # verify the structure of the configuration file. Checks that all needed nodes are present and of the right type and within allowed parameters

        # verify the high-level structure of the configuration
        needed_high_level_keys = ["graph", "execution"]
        if not all(key in config for key in needed_high_level_keys):
            return (
                False,
                f"Configuration is missing required keys. Required keys are {needed_high_level_keys}.",
            )

        # we need to have a dask scheduler defined in the execution section...
        if "scheduler" not in config["execution"]:
            return False, "Execution configuration is missing 'scheduler' key."

        # ... and it must be one of those that are supported by dask
        if config["execution"]["scheduler"] not in [
            "synchronous",
            "threads",
            "multiprocessing",
            "distributed",
        ]:
            scheduler = config["execution"]["scheduler"]
            return (
                False,
                f"Unsupported scheduler: {scheduler}. Supported schedulers are 'synchronous', 'threads', 'multiprocessing', or 'distributed'.",
            )

        return self._verify_computation_config(config["graph"])

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
    def from_config(cls, path_to_config: str | Path) -> ComputationGraph:
        """Creates a `ComputationGraph` instance from a configuration dictionary read from a json file."""
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return cls(config)
