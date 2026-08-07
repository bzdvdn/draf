"""Command — combine state updates with dynamic routing (raw Graph API).

The same content-moderation gate as ``flow.py``, but assembled with the raw
:class:`teff.graph.Graph` and explicit :class:`teff.graph.Edge` objects
instead of the :class:`teff.flow.Flow` builder — it shows that ``goto`` can
target *any* node even when there is no edge between them.

The graph has an ordinary edge ``moderate -> review``, yet a
``goto="deliver"`` from ``moderate`` to ``deliver`` works because the routing
comes from the returned :class:`teff.node.Command`, not the topology:

    start -> moderate --(no goto, plain dict)--> review -> deliver
                     --goto="deliver"-----------> deliver (bypasses edges)
                     --goto=Command.STOP--------> end immediately

No LLM, no API key, no Ollama — plain node classes only.

Usage:
    python examples/command_routing/graph.py
"""

from __future__ import annotations

import asyncio

from teff.graph import Edge, Graph
from teff.logging import configure_logging
from teff.node import Command, Transform
from teff.node.node import Node


class Moderate(Node):
    """The decider: route with an explicit ``goto`` instead of edges."""

    type = "moderate"

    async def execute(self, ctx, state: dict) -> dict | Command:
        text = str(state.get("text", "")).lower()

        if "bad" in text:
            # Stop the run immediately — nothing after this node executes.
            return Command(update={"blocked": True}, goto=Command.STOP)

        if "trusted" in text:
            # There is no ``moderate -> deliver`` edge, but ``goto`` jumps.
            return Command(update={"cleared": True}, goto="deliver")

        if text.rstrip().endswith("?"):
            # Update state; keep the normal ``moderate -> review`` edge.
            return Command(update={"cleared": True, "needs_review": True})

        # A plain dict — route along the real ``moderate -> review`` edge.
        return {"cleared": True}


class Review(Node):
    type = "review"

    async def execute(self, ctx, state: dict) -> dict:
        return {"reviewed": True}


class Deliver(Node):
    type = "deliver"

    async def execute(self, ctx, state: dict) -> dict:
        return {"delivered": state.get("cleared", False)}


def build_graph() -> Graph:
    return Graph(
        nodes={
            "start": Transform(
                action="uppercase", input_key="text", output_key="text", id="start"
            ),
            "moderate": Moderate(),
            "review": Review(),
            "deliver": Deliver(),
        },
        edges=[
            Edge("start", "moderate"),
            Edge("moderate", "review"),
            Edge("review", "deliver"),
        ],
        entry_point="start",
    )


async def main() -> None:
    graph = build_graph()
    print("=== Command routing (raw Graph API) ===\n")
    for text in ["this is bad", "trusted user", "could you fix the bug?", "hello"]:
        state = {"text": text}
        result = await graph.run(state, max_iterations=10)
        summary = {
            "cleared": result.get("cleared"),
            "needs_review": result.get("needs_review", False),
            "reviewed": result.get("reviewed", False),
            "blocked": result.get("blocked", False),
            "delivered": result.get("delivered"),
        }
        print(f"input : {text!r}")
        print(f"route : {summary}\n")


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
