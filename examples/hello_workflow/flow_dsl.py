"""Same workflow as workflow.yaml, built with the Flow builder.

Flow DSL chains nodes: ``transform`` → ``branch`` → ``converge``.  Each
method returns the flow, so you read the graph top-to-bottom like a script.

Usage:
    python examples/hello_workflow/flow_dsl.py
"""

import asyncio

from teff.flow import Flow
from teff.flow.case import Case
from teff.node import Transform


def build() -> Flow:
    flow = (
        Flow("hello-offline")
        .transform(
            action="count_lines",
            input_key="text",
            output_key="lines",
            id="count",
        )
        .branch(
            "lines",
            Case("1").add(
                Transform(
                    action="value",
                    value="single-line note",
                    output_key="note",
                ),
                id="single",
            ),
            default=Transform(
                action="value",
                value="multi-line note",
                output_key="note",
            ),
        )
        .converge(
            Transform(action="value", value="done", output_key="status"),
            id="status",
        )
    )
    return flow


async def main() -> None:
    graph = build().compile()
    result = await graph.run(state={"text": "two\nlines"})
    print("lines:", result.get("lines"))
    print("note:", result.get("note"))
    print("status:", result.get("status"))


if __name__ == "__main__":
    asyncio.run(main())
