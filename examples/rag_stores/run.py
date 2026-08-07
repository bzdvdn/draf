"""Run any RAG workflow from a YAML file — CLI emulation.

Usage:
    uv run python examples/rag_stores/run.py <path/to/workflow.yaml>

Each store example has its own README describing what to install
(dependency extra, and a server where required).
"""

import asyncio
import sys

from teff.yaml import load_workflow


async def main(path: str):
    graph, tools, state, reducers = load_workflow(path)
    result = await graph.run(state, tools=tools, reducers=reducers, max_iterations=10)
    tool_calls = sum(1 for m in result.get("messages", []) if m.get("role") == "tool")
    print("Query:", result.get("query"))
    print("Answer:", result.get("answer"))
    print("Tool calls:", tool_calls)


if __name__ == "__main__":
    default = "in_memory/workflow.yaml"
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if len(sys.argv) > 1:
        path = sys.argv[1]
    asyncio.run(main(path))
