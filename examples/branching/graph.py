"""Same workflow as workflow.yaml, built with low-level Graph API.

Usage:
    ollama pull llama3.1:8b
    python examples/branching/graph.py
"""

import asyncio
from draf import set_defaults
from draf.graph import Graph, Edge
from draf.node import LLM, Transform
from draf.state.state import reducers_from_yaml_schema

set_defaults(provider="ollama")


async def main():
    g = Graph(
        nodes={
            "classify": LLM(
                model="llama3.1:8b",
                system=(
                    "Classify the sentiment of the user's text. Respond with"
                    ' exactly one word: "positive" or "negative".'
                ),
                input_key="text",
                output_key="sentiment",
            ),
            "on_positive": Transform(
                action="value",
                value="Glad you liked it! ",
                output_key="reply",
            ),
            "on_negative": Transform(
                action="value",
                value="Sorry to hear that. ",
                output_key="reply",
            ),
            "shout": Transform(
                action="uppercase",
                input_key="reply",
                output_key="result",
            ),
        },
        edges=[
            Edge("classify", "on_positive", "sentiment=positive"),
            Edge("classify", "on_negative", "sentiment=negative"),
            Edge("on_positive", "shout"),
            Edge("on_negative", "shout"),
        ],
        entry_point="classify",
    )

    reducers = reducers_from_yaml_schema(
        {
            "messages": {"reducer": "append", "type": "list"},
        }
    )

    result = await g.run(
        state={"text": "I love this product!"},
        reducers=reducers,
    )
    print("Result:", result.get("result"))
    print("Sentiment:", result.get("sentiment"))


if __name__ == "__main__":
    asyncio.run(main())
