"""Durable graph execution: crash a node, resume from a checkpoint.

A graph of three nodes where the second node crashes on the first run.
The run raises, but the checkpoint before node 2 was already persisted.
Re-running with the same checkpoint_id skips the crashed node and
continues, proving the workflow is durable across failures.

Usage:
    python examples/checkpoint_resume/main.py
"""

import asyncio
import os

from draf.checkpoint import SQLiteCheckpointer
from draf.flow import Flow
from draf.node import Node, Transform

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "checkpoints.db")

# Simulates a transient external failure (network blip, timeout): the node
# fails once, then succeeds.  Must live outside the state, because state is
# restored to the pre-node checkpoint on resume.
_crash_once = {"armed": True}


class FailingNode(Node):
    """Raises on the first execution, succeeds afterwards."""

    type = "failing"

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__(**(config or {}), **kwargs)

    async def execute(self, ctx, state):
        if _crash_once["armed"]:
            _crash_once["armed"] = False
            raise RuntimeError("simulated transient failure")
        state["done"] = True
        return state


async def main():
    flow = Flow("durable")
    flow.step(
        Transform({"action": "uppercase", "input_key": "text", "output_key": "shout"})
    )
    flow.step(FailingNode({}))
    flow.step(
        Transform(
            {"action": "count_lines", "input_key": "shout", "output_key": "lines"}
        )
    )
    graph = flow.compile()

    checkpointer = SQLiteCheckpointer(DB_PATH)
    try:
        state = {"text": "durable\nworkflows"}

        for attempt in (1, 2):
            try:
                result = await graph.run(
                    state=state,
                    checkpointer=checkpointer,
                    checkpoint_id="durable-run",
                )
                print(f"Run {attempt}: success -> {result}")
            except RuntimeError as e:
                print(f"Run {attempt}: crashed ({e}), checkpoint saved")
                continue

        # show the persisted checkpoint
        cp = await checkpointer.load("durable-run")
        print("Saved checkpoint:", cp)
    finally:
        checkpointer.close()
        os.remove(DB_PATH)


if __name__ == "__main__":
    asyncio.run(main())
