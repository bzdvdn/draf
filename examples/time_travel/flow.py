"""Time travel, built with the Flow builder instead of raw Graph edges.

The same story-writing graph as ``graph.py`` (setup → conflict → twist →
ending), but assembled with :class:`teff.flow.Flow` — four ``flow.step()``
calls and no hand-written ``Edge`` objects:

    flow.step(setup).step(conflict).step(twist).step(ending)

The checkpointing/time-travel logic is identical to ``graph.py`` and comes
straight from the framework's ``SQLiteHistoryCheckpointer``:

1. Inspect the checkpoint history (which node runs next at each step).
2. Jump back to any earlier checkpoint.
3. Edit the state at that moment.
4. Replay from there — the past (chapters 1-2) stays identical, the
   future (chapters 3-4) is rewritten.

Every save (one per node, keyed by iteration) is appended to the history
table, so an earlier checkpoint survives the overwrite of the "current"
one.  The nodes are plain ``Transform`` appends — no LLM, fully offline.

Usage:
    python examples/time_travel/flow.py
"""

from __future__ import annotations

import asyncio
import os

from teff.checkpoint import SQLiteHistoryCheckpointer
from teff.flow import Flow
from teff.graph import Graph
from teff.node import Transform

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "time_travel.db")


def build_flow() -> Graph:
    flow = Flow("story")
    flow.step(
        Transform(
            config={
                "action": "append",
                "template": "Chapter 1: A hero named {hero} sets out.",
                "output_key": "chapters",
            }
        ),
        id="setup",
    )
    flow.step(
        Transform(
            config={
                "action": "append",
                "template": "Chapter 2: They come to the town of {setting}.",
                "output_key": "chapters",
            }
        ),
        id="conflict",
    )
    flow.step(
        Transform(
            config={
                "action": "append",
                "template": "Chapter 3: The twist: {twist}!",
                "output_key": "chapters",
            }
        ),
        id="twist",
    )
    flow.step(
        Transform(
            config={
                "action": "append",
                "template": "Chapter 4: And so the tale of {hero} ends.",
                "output_key": "chapters",
            }
        ),
        id="ending",
    )
    return flow.compile()


async def main() -> None:
    graph = build_flow()
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
