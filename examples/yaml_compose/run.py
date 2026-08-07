"""Run the pure-YAML composition example (no API key needed).

Usage:
    python examples/yaml_compose/run.py
"""

import asyncio

from teff.yaml import load_workflow


async def main():
    graph, tools, state, reducers = load_workflow("examples/yaml_compose/workflow.yaml")
    result = await graph.run(state, tools=tools, reducers=reducers)
    for k in (
        "items",
        "has_urgent",
        "routed",
        "joined",
        "generated_at",
    ):
        print(f"{k}: {result.get(k)!r}")


if __name__ == "__main__":
    asyncio.run(main())
