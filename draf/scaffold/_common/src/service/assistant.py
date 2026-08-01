"""Application service: one conversation turn against the supervisor graph.

Both the FastAPI server (the project-root ``app.py``) and the debug CLI (``cli.py``)
go through :class:`Assistant`, so a turn behaves identically over HTTP and
on the terminal — session hydration, tool set, reducers and checkpointing
are defined exactly once.
"""

from __future__ import annotations

from typing import AsyncIterator

from draf.checkpoint import DEFAULT_OWNER
from draf.graph import Graph
from draf.stream import StreamEvent
from draf.tool import Tool

from src.graphs.state import STATE_REDUCERS
from src.storage import load_or_seed


class Assistant:
    """Runs durable conversation turns against a compiled graph."""

    def __init__(
        self,
        graph: Graph,
        tools: list[Tool],
        checkpointer,
        *,
        max_iterations: int = 80,
    ):
        self.graph = graph
        self.tools = tools
        self.checkpointer = checkpointer
        self.max_iterations = max_iterations

    async def run_turn(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
    ) -> dict:
        """Run one turn and return the final state."""
        state, run_kwargs = await load_or_seed(
            self.graph, self.checkpointer, session_id, message, owner=owner
        )
        return await self.graph.run(
            state,
            tools=self.tools,
            reducers=STATE_REDUCERS,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            **run_kwargs,
        )

    async def stream_turn(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the events of one turn (``run_start`` .. ``run_end``)."""
        state, run_kwargs = await load_or_seed(
            self.graph, self.checkpointer, session_id, message, owner=owner
        )
        async for event in self.graph.stream(
            state,
            tools=self.tools,
            reducers=STATE_REDUCERS,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            **run_kwargs,
        ):
            yield event
