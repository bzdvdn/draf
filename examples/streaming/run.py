"""Streaming LLM tokens and graph events via the Flow API.

``Flow.step()`` builds the chain declaratively, ``flow.compile()`` turns
it into a ``Graph``, and ``graph.stream()`` yields a
:class:`StreamEvent` for every observable step of a run — node start/end,
routing edges, checkpoints, and LLM tokens as they are generated.  Tokens
render immediately instead of waiting for the whole response.

The same graph built with the low-level ``Graph`` API lives in
``graph.py``.

Requires Ollama running locally with llama3.1:8b.

Usage:
    python examples/streaming/run.py
"""

import asyncio

from teff.flow import Flow
from teff.provider import ProviderRegistry

SYSTEM = "Ты инженер по ремонту."


async def main():
    flow = Flow(
        "streaming",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.llm(
        model="llama3.1:8b",
        system=SYSTEM,
        prompt=(
            "Составь краткий план ремонта санузла на сумму 80000 рублей, 3-5 пунктов."
        ),
        output_key="draft",
    ).llm(
        model="llama3.1:8b",
        system=SYSTEM,
        prompt="Отформатируй план красиво в 5-7 строк:\n{draft}",
        output_key="final",
    )

    graph = flow.compile()

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
