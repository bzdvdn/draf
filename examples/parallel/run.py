"""Run a graph with parallel branches (no API key needed).

Three transforms run concurrently on independent keys, then a final
transform converges the results.

Usage:
    python examples/parallel/run.py
"""

import asyncio

from teff.flow import Flow
from teff.node import Transform


async def main():
    flow = (
        Flow("parallel-demo")
        .step(
            Transform(
                action="uppercase",
                input_key="title",
                output_key="title",
            )
        )
        .parallel(
            [Transform(action="count_lines", input_key="title", output_key="lines")],
            [Transform(action="uppercase", input_key="body", output_key="body")],
        )
        .converge(
            Transform(
                action="value",
                value="done",
                output_key="status",
            )
        )
    )

    graph = flow.compile()
    result = await graph.run(
        state={
            "title": "hello world\nsecond line",
            "body": "some text",
        }
    )
    for k in ("title", "lines", "body", "status"):
        print(f"{k}: {result.get(k)!r}")


if __name__ == "__main__":
    asyncio.run(main())
