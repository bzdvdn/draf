"""Run the branching workflow with Ollama.

Usage:
    # Make sure Ollama is running and has the model
    ollama pull llama3.1:8b
    python examples/branching/run.py
"""

import asyncio

from teff.yaml import load_workflow


async def main():
    graph, tools, state, reducers = load_workflow("examples/branching/workflow.yaml")
    result = await graph.run(state, tools=tools, reducers=reducers)
    print("Result:", result.get("result"))
    print("Sentiment:", result.get("sentiment"))


if __name__ == "__main__":
    asyncio.run(main())
