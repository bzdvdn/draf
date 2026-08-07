"""Same workflow as workflow.yaml, built with the low-level Graph API.

Low-level gives explicit control: every node and every edge (with an
optional ``condition``) is declared by hand.

Usage:
    python examples/hello_workflow/low_level.py
"""

import asyncio

from teff.graph import Edge, Graph
from teff.node import Transform


def build() -> Graph:
    return Graph(
        nodes={
            "count": Transform(
                action="count_lines",
                input_key="text",
                output_key="lines",
            ),
            "single": Transform(
                action="value",
                value="single-line note",
                output_key="note",
            ),
            "multi": Transform(
                action="value",
                value="multi-line note",
                output_key="note",
            ),
            "status": Transform(
                action="value",
                value="done",
                output_key="status",
            ),
        },
        edges=[
            Edge("count", "single", "lines=1"),
            Edge("count", "multi", "lines!=1"),
            Edge("single", "status"),
            Edge("multi", "status"),
        ],
        entry_point="count",
    )


async def main() -> None:
    graph = build()
    result = await graph.run(state={"text": "two\nlines"})
    print("lines:", result.get("lines"))
    print("note:", result.get("note"))
    print("status:", result.get("status"))


if __name__ == "__main__":
    asyncio.run(main())
