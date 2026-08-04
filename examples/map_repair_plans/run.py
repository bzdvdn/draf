"""Map + prompt templates + typed State.

Fans out a list of repair jobs into parallel LLM calls.  Each LLM call
reads *two* state keys (``type``, ``summ``) into one prompt via the
``{key}`` template, and the results are collected into the ``plans``
list.  State is a typed :class:`~draf.state.State` (TypedDict), not a
bare dict.

Requires Ollama running locally with llama3.1:8b.

Usage:
    python examples/map_repair_plans/run.py
    # or the YAML workflow (plain dict + schema reducers):
    draf -f examples/map_repair_plans/workflow.yaml
"""

import asyncio
from typing import Annotated, TypedDict

from draf.flow import Flow
from draf.node import LLM
from draf.provider import ProviderRegistry
from draf.state import State
from draf.trace import RunTracer

SYSTEM = (
    "Ты инженер по ремонту. Составь краткий план работ по заданию "
    "клиента. Отвечай по-русски, 3-5 пунктов."
)

REPAIRS = [
    ("кухни", 150000),
    ("санузла", 80000),
    ("балкона", 45000),
]


class RepairState(TypedDict):
    type: list[str]
    summ: list[int]
    plans: Annotated[list, "append"]


async def main():
    flow = Flow(
        "repair-plans",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    ).map(
        LLM(
            model="llama3.1:8b",
            system=SYSTEM,
            prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
            output_key="plan",
        ),
        input_keys=["type", "summ"],
        output_key="plans",
        max_concurrency=2,
    )

    graph = flow.compile()
    tracer = RunTracer()
    state = State(
        RepairState,
        {
            "type": [r[0] for r in REPAIRS],
            "summ": [r[1] for r in REPAIRS],
        },
    )
    result = await graph.run(state=state, tracer=tracer)

    print("=== Plans ===")
    for i, plan in enumerate(result["plans"], start=1):
        print(f"--- Plan {i} ---")
        print(plan if plan else "(empty)")
        print()

    summary = tracer.summary()
    print(
        f"\nstatus: {summary.status}  "
        f"llm_calls: {summary.llm_calls}  tokens: {summary.tokens}"
    )


if __name__ == "__main__":
    asyncio.run(main())
