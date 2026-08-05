"""Time travel: rewind a finished run to any checkpoint and replay.

A story-writing graph runs to completion, saving a checkpoint *before each
node* via a checkpointer.  Time travel lets you:

1. Inspect the checkpoint history (which node runs next at each step).
2. Jump back to any earlier checkpoint.
3. Edit the state at that moment.
4. Replay from there — the past (chapters 1-2) stays identical, the
   future (chapters 3-4) is rewritten.

This mirrors LangGraph's "time travel" (rewind a thread to a checkpoint,
edit state, and resume), built on DRAFTFLOW's ``SQLiteHistoryCheckpointer``:
every save (one per node, keyed by iteration) is appended to a history
table, so an earlier checkpoint survives the overwrite of the "current"
one.  The nodes are plain ``Transform`` appends — no LLM, fully offline.

Usage:
    python examples/time_travel/graph.py
"""

from __future__ import annotations

import asyncio
import os

from draf.checkpoint import SQLiteHistoryCheckpointer
from draf.graph import Edge, Graph
from draf.node import Transform

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "time_travel.db")


def build_graph() -> Graph:
    setup = Transform(
        config={
            "action": "append",
            "template": "Chapter 1: A hero named {hero} sets out.",
            "output_key": "chapters",
        }
    )
    conflict = Transform(
        config={
            "action": "append",
            "template": "Chapter 2: They come to the town of {setting}.",
            "output_key": "chapters",
        }
    )
    twist = Transform(
        config={
            "action": "append",
            "template": "Chapter 3: The twist: {twist}!",
            "output_key": "chapters",
        }
    )
    ending = Transform(
        config={
            "action": "append",
            "template": "Chapter 4: And so the tale of {hero} ends.",
            "output_key": "chapters",
        }
    )
    return Graph(
        nodes={
            "setup": setup,
            "conflict": conflict,
            "twist": twist,
            "ending": ending,
        },
        edges=[
            Edge("setup", "conflict"),
            Edge("conflict", "twist"),
            Edge("twist", "ending"),
        ],
        entry_point="setup",
    )


async def main() -> None:
    graph = build_graph()
    cp = SQLiteHistoryCheckpointer(DB_PATH)
    try:
        # --- run 1: write the original story, checkpointing every step ---
        result = await graph.run(
            state={
                "hero": "Ada",
                "setting": "Bellmore",
                "twist": "the butler did it",
                "chapters": [],
            },
            checkpointer=cp,
            checkpoint_id="story",
        )
        print("=== Original story ===")
        for ch in result["chapters"]:
            print("  ", ch)

        # --- inspect the timeline: iteration -> next node ---
        timeline = await cp.history("story")
        print("\n=== Checkpoint timeline ===")
        for iteration, next_node in timeline:
            print(f"  iter {iteration}: next node = {next_node}")

        # --- time travel: rewind to the moment just before 'twist' ---
        target = next(i for i, n in timeline if n == "twist")
        past = await cp.load_at("story", target)
        assert past is not None
        past.state["twist"] = "it was all a dream"

        await cp.save("story-branch", past)
        print(
            f"\n=== Rewound to iter {target}, changed twist -> {past.state['twist']} ==="
        )

        # --- replay from the past: chapters 1-2 are identical, 3-4 are new ---
        branch = await graph.run(
            state={"chapters": []},
            checkpointer=cp,
            checkpoint_id="story-branch",
        )
        print("=== Rewritten story ===")
        for ch in branch["chapters"]:
            print("  ", ch)

        same_past = branch["chapters"][:2] == result["chapters"][:2]
        changed_future = branch["chapters"][2:] != result["chapters"][2:]
        print(f"\npast preserved (chapters 1-2): {same_past}")
        print(f"future rewritten (chapters 3-4): {changed_future}")
    finally:
        cp.close()
        os.remove(DB_PATH)


if __name__ == "__main__":
    asyncio.run(main())
