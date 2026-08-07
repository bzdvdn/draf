"""Schema-validated LLM output via the Flow API.

``Flow.step()`` builds the chain declaratively and ``flow.compile()``
returns a ``Graph``.  The ``LLM`` node validates the model's JSON
response against the ``RepairPlan`` ``TypedDict`` (via the derived
JSON Schema) and returns the parsed dict in ``state["plan"]``.  On
malformed output the node feeds the validation error back and retries
up to ``max_retries`` times; if every attempt fails,
``StructuredOutputError`` propagates out of ``graph.run()``.

The same graph built with the low-level ``Graph`` API (plus an
``__error__`` fallback edge that handles the failure inside the graph)
lives in ``graph.py``.

Requires Ollama running locally with llama3.1:8b.

Usage:
    python examples/structured_output/run.py
"""

import asyncio
import json
from typing import TypedDict

from teff.flow import Flow
from teff.node.llm import StructuredOutputError
from teff.provider import ProviderRegistry


class RepairPlan(TypedDict):
    title: str
    items: list[str]
    cost: int


async def main():
    flow = Flow(
        "structured-output",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.llm(
        model="llama3.1:8b",
        prompt=(
            "Составь план ремонта санузла в виде JSON: "
            "title (строка), items (3 пункта), cost (целое число)."
        ),
        output_key="plan",
        output_type=RepairPlan,
        max_retries=2,
    )
    graph = flow.compile()

    try:
        result = await graph.run({})
    except StructuredOutputError as exc:
        print(f"structured output failed after {exc.attempts} attempts")
        return

    plan = result["plan"]
    print("parsed dict:", type(plan).__name__)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
