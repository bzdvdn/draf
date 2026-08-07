"""Run the RAG workflow from workflow.yaml — CLI emulation.

Usage:
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    python examples/rag_search/run.py

The documents are declared in workflow.yaml and embedded lazily on the
first search, so this script is just: load workflow → run graph.
"""

import asyncio

from teff.yaml import load_workflow


async def main():
    graph, tools, state, reducers = load_workflow("examples/rag_search/workflow.yaml")
    result = await graph.run(state, tools=tools, reducers=reducers, max_iterations=10)
    tool_calls = sum(1 for m in result.get("messages", []) if m.get("role") == "tool")
    print("Query:", result.get("query"))
    print("Answer:", result.get("answer"))
    print("Tool calls:", tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
