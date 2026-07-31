"""Schema-validated LLM output via the low-level ``Graph`` API.

The LLM is asked to return a JSON object matching the ``RepairPlan``
``TypedDict``.  ``graph.run()`` parses the response, validates it against
the derived JSON Schema, and returns the parsed dict in ``state["plan"]``.
If the model returns malformed JSON, the node feeds the validation error
back and re-asks (up to ``max_retries``); after that a
``StructuredOutputError`` routes to the ``__error__`` edge.

The same workflow built with the declarative ``Flow`` API (without the
``__error__`` edge, the error propagates to the caller) lives in
``run.py``.

Requires Ollama running locally with llama3.1:8b.

Usage:
    python examples/structured_output/graph.py
"""

import asyncio
import json
from typing import TypedDict

from draf import set_defaults
from draf.graph import Graph, Edge
from draf.node import LLM

set_defaults(provider="ollama")


class RepairPlan(TypedDict):
    title: str
    items: list[str]
    cost: int


async def main():
    graph = Graph(
        nodes={
            "plan": LLM(
                model="llama3.1:8b",
                prompt=(
                    "Составь план ремонта санузла в виде JSON: "
                    "title (строка), items (3 пункта), cost (целое число)."
                ),
                output_key="plan",
                output_type=RepairPlan,
                max_retries=2,
            ),
            "fallback": LLM(
                model="llama3.1:8b",
                prompt="Ответь простым текстом: не получилось распарсить JSON.",
                output_key="error_msg",
            ),
        },
        edges=[Edge("plan", "fallback", "__error__")],
        entry_point="plan",
    )

    result = await graph.run({})
    if "error_msg" in result:
        print("structured output failed; fallback:", result["error_msg"])
        return

    plan = result["plan"]
    print("parsed dict:", type(plan).__name__)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
