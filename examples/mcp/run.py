"""ReAct agent calling tools over MCP.

Connects the graph to the local MCP server in ``server.py`` over stdio
and exposes its tools to a :class:`~teff.node.agent.ReActAgent` loop.
Any MCP server works the same way — point ``mcp_tools`` at a Streamable
HTTP endpoint (``url=...``) or a stdio command (``command=...``).

Requires Ollama running locally with llama3.1:8b (the `mcp` SDK ships with
the core package).

Usage:
    python examples/mcp/run.py
"""

import asyncio
import sys
from pathlib import Path

from teff.flow import Flow
from teff.provider import ProviderRegistry
from teff.tool import mcp_tools

SERVER = Path(__file__).resolve().parent / "server.py"


async def main():
    flow = Flow(
        "mcp_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.react(
        model="llama3.1:8b",
        system=(
            "You are a helpful assistant with access to a weather tool "
            "and a calculator tool. Use them when the question asks."
        ),
        input_key="query",
        output_key="answer",
    )
    graph = flow.compile()

    async with mcp_tools(command=[sys.executable, str(SERVER)]) as tools:
        print("MCP tools:", [t.name for t in tools])
        result = await graph.run(
            state={"query": "What is the weather in Tokyo?"},
            tools=tools,
            max_iterations=10,
        )
    print("Answer:", result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
