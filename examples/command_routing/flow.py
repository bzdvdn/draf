"""Command — combine state updates with dynamic routing (Flow API).

Normally a node returns a plain dict of state updates and the graph routes
along its outgoing edges (``branch`` / ``loop`` / string conditions).
Returning a :class:`draf.node.Command` lets the node **also** pick the next
node — LangGraph-style dynamic routing:

    Command(update={...}, goto="node_id")   # jump to any node, bypassing edges
    Command(update={...})                    # update, keep normal edge routing
    Command(goto=Command.STOP)                # end the run immediately

This example is a content-moderation gate built from plain function nodes —
no LLM, no API key, no Ollama — so it runs offline.  The flow is linear:

    start -> moderate -> review -> deliver

but ``moderate`` routes with ``Command`` depending on the incoming text:

  * text contains "bad"     -> ``Command(update={blocked}, goto=STOP)``  (end)
  * text contains "trusted" -> ``Command(update=True, goto="deliver")``  (skip review)
  * text ends with "?"      -> ``Command(update=True)``                   (normal edge -> review)
  * anything else             -> plain dict, normal edge -> review -> deliver

Usage:
    python examples/command_routing/flow.py
"""

from __future__ import annotations

import asyncio

from draf.flow import Flow
from draf.graph import Graph
from draf.logging import configure_logging
from draf.node import Command, Transform


def moderate(ctx, state: dict) -> dict | Command:
    """The decider: route with an explicit ``goto`` instead of edges."""
    text = str(state.get("text", "")).lower()

    if "bad" in text:
        # Block and stop the run — nothing after this node runs.
        return Command(update={"blocked": True}, goto=Command.STOP)

    if "trusted" in text:
        # Jump straight to the last node, skipping the review below.
        return Command(update={"cleared": True}, goto="deliver")

    if text.rstrip().endswith("?"):
        # Needs a review: update only, keep the normal review edge.
        return Command(update={"cleared": True, "needs_review": True})

    # A plain dict routes normally through review -> deliver.
    return {"cleared": True}


def review(ctx, state: dict) -> dict | Command:
    """Human-ish review; returns a Command without a ``goto``."""
    return Command(update={"reviewed": True})


def deliver(ctx, state: dict) -> dict:
    """Terminal node: its return lands back in ``state``."""
    return {"delivered": state.get("cleared", False)}


def build_flow() -> Graph:
    return (
        Flow("command_routing")
        .step(
            Transform(action="uppercase", input_key="text", output_key="text"),
            id="start",
        )
        .step(moderate, id="moderate")
        .step(review, id="review")
        .step(deliver, id="deliver")
    ).compile()


async def main() -> None:
    flow = build_flow()
    print("=== Command routing (Flow API) ===\n")
    for text in ["this is bad", "trusted user", "could you fix the bug?", "hello"]:
        state = {"text": text}
        result = await flow.run(state, max_iterations=10)
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
