"""Same workflow as workflow.yaml, built with high-level Flow API.

Demonstrates Flow().step().branch().Case().converge() pattern.

Usage:
    ollama pull llama3.1:8b
    python examples/branching/flow.py
"""

import asyncio

from draf.flow import Case, Flow
from draf.node import Transform
from draf.provider import ProviderRegistry
from draf.state.state import reducers_from_yaml_schema


async def main():
    on_positive = Transform(
        action="value",
        value="Glad you liked it! ",
        output_key="reply",
    )
    on_negative = Transform(
        action="value",
        value="Sorry to hear that. ",
        output_key="reply",
    )
    shout = Transform(
        action="uppercase",
        input_key="reply",
        output_key="result",
    )

    flow = (
        Flow(
            "sentiment-router",
            providers=ProviderRegistry.from_presets("ollama"),
            default_provider="ollama",
        )
        .llm(
            model="llama3.1:8b",
            system=(
                "Classify the sentiment of the user's text. Respond with"
                ' exactly one word: "positive" or "negative".'
            ),
            input_key="text",
            output_key="sentiment",
        )
        .branch(
            "sentiment",
            Case("positive").add(on_positive),
            Case("negative").add(on_negative),
        )
        .converge(shout)
    )

    graph = flow.compile()
    reducers = reducers_from_yaml_schema(
        {
            "messages": {"reducer": "append", "type": "list"},
        }
    )

    result = await graph.run(
        state={"text": "I love this product!"},
        reducers=reducers,
    )
    print("Result:", result.get("result"))
    print("Sentiment:", result.get("sentiment"))


if __name__ == "__main__":
    asyncio.run(main())
