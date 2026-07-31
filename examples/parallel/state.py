"""Run a graph with parallel branches and a typed State (no API key).

Shows the ``State``/TypedDict reducers working inside parallel branches:
three concurrent producers append to the shared ``messages`` key, and
``Parallel`` merges their deltas through the ``append`` reducer instead
of overwriting one another.

Usage:
    python examples/parallel/state.py
"""

import asyncio
from typing import Annotated, TypedDict

from draf.flow import Flow
from draf.node import Node
from draf.state import State


class ChatState(TypedDict):
    messages: Annotated[list, "append"]
    status: str


class Produce(Node):
    """Append a single message to the shared ``messages`` key."""

    type = "produce"

    async def execute(self, ctx, state):
        return {"messages": [self.config["msg"]]}


class MarkDone(Node):
    type = "mark_done"

    async def execute(self, ctx, state):
        return {"status": "finished"}


async def main():
    flow = (
        Flow("parallel-typed-state")
        .parallel(
            [Produce({"msg": "alpha: done"})],
            [Produce({"msg": "beta: done"})],
            [Produce({"msg": "gamma: done"})],
        )
        .converge(MarkDone())
    )

    graph = flow.compile()
    result = await graph.run(state=State(ChatState, {}))

    print("messages:", sorted(result["messages"]))
    print("status:", result["status"])


if __name__ == "__main__":
    asyncio.run(main())
