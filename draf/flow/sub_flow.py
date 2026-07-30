"""SubFlow — a node that executes a nested graph."""

from draf.node.node import Node
from draf.graph import Graph


class SubFlow(Node):
    """A node that executes a sub-graph with optional key mapping.

    Args:
        graph: Compiled sub-graph to run.
        input_map: Parent key → sub-graph key (default: passthrough).
        output_map: Sub-graph key → parent key (default: passthrough).
    """

    type = "subflow"

    def __init__(
        self,
        graph: Graph,
        input_map: dict[str, str] | None = None,
        output_map: dict[str, str] | None = None,
    ):
        super().__init__({})
        self._graph = graph
        self._input_map = input_map or {}
        self._output_map = output_map or {}

    async def execute(self, ctx, state: dict) -> dict:
        sub_state = {}
        if self._input_map:
            for parent_key, sub_key in self._input_map.items():
                sub_state[sub_key] = state.get(parent_key)
        else:
            sub_state = dict(state)

        result = await self._graph.run(sub_state, tools=list(ctx.tools.values()))

        out = {}
        if self._output_map:
            for sub_key, parent_key in self._output_map.items():
                out[parent_key] = result.get(sub_key)
        else:
            out = result
        return out
