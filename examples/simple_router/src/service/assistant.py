"""Application service: one conversation turn against the router graph.

The CLI entry point (``cli.py``) goes through :class:`Assistant`, so a turn
behaves identically on the terminal and in tests — session hydration, tool
set, reducers and checkpointing are defined exactly once.
"""

from __future__ import annotations

from typing import AsyncIterator

from draf.graph import Graph
from draf.stream import StreamEvent

from src.graphs.state import STATE_REDUCERS
from src.storage import load_or_seed


class Assistant:
    """Runs durable conversation turns against a compiled graph."""

    def __init__(
        self,
        graph: Graph,
        checkpointer,
        *,
        max_iterations: int = 80,
    ):
        self.graph = graph
        self.checkpointer = checkpointer
        self.max_iterations = max_iterations

    async def run_turn(self, session_id: str, message: str) -> dict:
        """Run one turn and return the final state."""
        state, run_kwargs = await load_or_seed(self.graph, self.checkpointer, session_id, message)
        return await self.graph.run(
            state,
            tools=[],
            reducers=STATE_REDUCERS,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            max_iterations=self.max_iterations,
            **run_kwargs,
        )

    async def stream_turn(
        self, session_id: str, message: str
    ) -> AsyncIterator[StreamEvent]:
        """Stream the events of one turn (``run_start`` .. ``run_end``)."""
        state, run_kwargs = await load_or_seed(self.graph, self.checkpointer, session_id, message)
        async for event in self.graph.stream(
            state,
            tools=[],
            reducers=STATE_REDUCERS,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            max_iterations=self.max_iterations,
            **run_kwargs,
        ):
            yield event
