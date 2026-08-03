"""Streaming LLM tokens and graph events via the low-level ``Graph`` API.

The same workflow built with the declarative ``Flow`` API lives in
``run.py``.  ``flow.compile()`` returns exactly this ``Graph``, so both
versions stream identically.

Requires Ollama running locally with llama3.1:8b.

Usage:
    python examples/streaming/graph.py
"""

import asyncio

from draf import set_defaults
from draf.graph import Edge, Graph
from draf.node import LLM

set_defaults(provider="ollama")


async def main():
    graph = Graph(
        nodes={
            "draft": LLM(
                model="llama3.1:8b",
                system="Ты инженер по ремонту.",
                prompt=(
                    "Составь краткий план ремонта санузла "
                    "на сумму 80000 рублей, 3-5 пунктов."
                ),
                output_key="draft",
            ),
            "final": LLM(
                model="llama3.1:8b",
                system="Ты инженер по ремонту.",
                prompt="Отформатируй план красиво в 5-7 строк:\n{draft}",
                output_key="final",
            ),
        },
        edges=[Edge("draft", "final")],
        entry_point="draft",
    )

    async for event in graph.stream({}):
        if event.type == "node_start":
            print(f"\n-- {event.node_id} [{event.node_type}] --")
        elif event.type == "token":
            print(event.data["token"], end="", flush=True)
        elif event.type == "edge":
            print(f"\n  -> routed to {event.data['target_id']}")
        elif event.type == "run_end":
            status = event.data["status"]
            print(f"\n== run_end: {status} ==")
        elif event.type == "node_error":
            print(f"\n  !! {event.data['error']}")


if __name__ == "__main__":
    asyncio.run(main())
