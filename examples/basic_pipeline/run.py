"""Run the basic pipeline (no API key needed).

Usage:
    python examples/basic_pipeline/run.py
"""

import asyncio

from teff.yaml import load_workflow


async def main():
    graph, tools, state, reducers = load_workflow(
        "examples/basic_pipeline/workflow.yaml"
    )
    result = await graph.run(state, tools=tools, reducers=reducers)
    for k in ("title", "trimmed", "loud", "line_count"):
        print(f"{k}: {result.get(k)!r}")


if __name__ == "__main__":
    asyncio.run(main())
