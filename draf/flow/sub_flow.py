"""SubFlow — a node that executes a nested graph."""

from typing import Awaitable, Callable

from draf.node.node import Node
from draf.graph import Graph
from draf.stream import StreamEvent


class SubFlow(Node):
    """A node that executes a sub-graph with optional key mapping.

    Args:
        graph: Compiled sub-graph to run.
        input_map: Parent key → sub-graph key (default: passthrough).
        output_map: Sub-graph key → parent key (default: passthrough).
        max_iterations: Max node executions inside the sub-graph
            (passed to ``graph.run()``).  ``None`` means unlimited.
    """

    type = "subflow"

    def __init__(
        self,
        graph: Graph,
        input_map: dict[str, str] | None = None,
        output_map: dict[str, str] | None = None,
        max_iterations: int | None = None,
    ):
        super().__init__(
            input_map=input_map or {},
            output_map=output_map or {},
            max_iterations=max_iterations,
        )
        self._graph = graph
        self._input_map = input_map or {}
        self._output_map = output_map or {}
        self._max_iterations = max_iterations

    async def execute(self, ctx, state: dict) -> dict:
        sub_state = {}
        if self._input_map:
            for parent_key, sub_key in self._input_map.items():
                sub_state[sub_key] = state.get(parent_key)
        else:
            sub_state = dict(state)

        result = await self._graph.run(
            sub_state,
            tools=list(ctx.tools.values()),
            max_iterations=self._max_iterations,
            emit=self._forward(ctx.emit),
        )

        out = {}
        if self._output_map:
            for sub_key, parent_key in self._output_map.items():
                out[parent_key] = result.get(sub_key)
        else:
            out = result
        return out

    @staticmethod
    def _forward(
        emit: "Callable[[StreamEvent], Awaitable[None]] | None",
    ) -> "Callable[[StreamEvent], Awaitable[None]] | None":
        """Wrap an outer emit sink, dropping the nested run's bookkeeping.

        The inner run emits its own ``run_start``/``run_end`` lifecycle
        events; those belong to the top-level stream, so they are
        stripped while node/token/llm/edge events stream through.
        """
        if emit is None:
            return None

        async def forward(event: StreamEvent) -> None:
            if event.type in ("run_start", "run_end"):
                return
            await emit(event)

        return forward
