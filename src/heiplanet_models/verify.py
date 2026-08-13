from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any


class ComputationGraphNode(BaseModel):
    function: str
    module: str
    input: list[str] = Field(default_factory=list)
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("function", "module")
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Value cannot be empty.")
        return value

    @field_validator("input", mode="after")
    def validate_input(cls, values: list[str]) -> list[str]:
        for input_node in values:
            if not input_node:
                raise ValueError("Input node names cannot be empty.")
        return values


class ComputationGraphExecution(BaseModel):
    scheduler: str
    log_level: str

    @field_validator("log_level")
    def validate_log_levels(cls, value: str) -> str:
        if not value:
            raise ValueError("log level name cannot be empty.")

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in valid_log_levels:
            raise ValueError(
                f"Invalid log level '{value}'. Valid options are: {', '.join(valid_log_levels)}."
            )
        return value

    @field_validator("scheduler")
    def validate_scheduler(cls, value: str) -> str:

        if not value:
            raise ValueError("scheduler name cannot be empty.")

        valid_schedulers = {"synchronous", "threads", "multiprocessing", "distributed"}

        if value not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{value}'. Valid options are: {', '.join(valid_schedulers)}."
            )

        return value


class ComputationGraphConfig(BaseModel):
    graph: dict[str, ComputationGraphNode]
    execution: ComputationGraphExecution

    def get_sink_nodes(self) -> list[str]:

        # logic: sink node must not be dependency in any other node
        all_nodes = set(self.graph.keys())
        upstream_nodes = set()

        for node in self.graph.values():
            upstream_nodes.update(node.input)

        sink_nodes = all_nodes - upstream_nodes

        if not sink_nodes:
            raise ValueError(
                "The graph must have at least one sink node (node with no downstream dependencies)."
            )

        if len(sink_nodes) > 1:
            raise ValueError(
                "The graph must not have more than one sink node (node with no downstream dependencies)"
            )

        return list(sink_nodes)

    def check_for_cycles(self, sink_node: str) -> bool:
        # cycle: node is in the input of an upstream node, and we know we only have one sink node
        # => we use a recursive reverse depth first search to check for cycles

        def dfs_reverse(node_name, visited: list[str]) -> bool:
            if node_name in visited:
                return True  # cycle detected
            visited.append(node_name)

            for upstream in self.graph[node_name].input:
                result = dfs_reverse(upstream, visited)
                if result:
                    return True

                visited.pop()  # when call stack unspools here this will remove all added nodes of a path
                # from the current node downwards, so that we can check other paths for cycles without interference from already visited nodes in other paths

            return False

        found_cycle = dfs_reverse(sink_node, [])
        return found_cycle

    @model_validator(mode="after")
    def validate_graph(self) -> "ComputationGraphConfig":

        for name, content in self.graph.items():
            for upstream in content.input:
                if upstream not in self.graph:
                    raise ValueError(
                        f"Node '{name}' has an upstream node '{upstream}' that is not defined in the graph."
                    )

        sink_nodes = self.get_sink_nodes()

        cycles_present = self.check_for_cycles(sink_nodes[0])
        if cycles_present:
            raise ValueError(
                "The graph contains cycles, which is not allowed. Please ensure the graph is acyclic."
            )

        return self
