"""ReAct agent: LLM + calculator tool in a visible graph cycle.

Requires Ollama running locally with llama3.1:8b.

Usage:
    # Make sure Ollama is running and has the model
    ollama pull llama3.1:8b
    python examples/react_agent/main.py
"""

import asyncio

from draf.flow import Flow
from draf.provider import ProviderRegistry
from draf.tool import Tool


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate a mathematical expression"

    def run(self, expression: str = "") -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error: {e}"


async def main():
    flow = Flow(
        "react_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.react(
        model="llama3.1:8b",
        system="You are a helpful assistant with calculator access. "
        "Use the calculator tool when you need to compute something.",
        input_key="query",
        output_key="answer",
        # Stream the final text token-by-token (tool calls still run normally).
        stream=True,
        on_token=lambda token: print(token, end="", flush=True),
    )

    graph = flow.compile()

    result = await graph.run(
        state={"query": "What is 2 + 2? Then multiply by 5."},
        tools=[Calculator()],
        max_iterations=10,
    )
    print("\n---")
    print("Answer:", result["answer"])
    print("Messages:", len(result["messages"]))


if __name__ == "__main__":
    asyncio.run(main())
