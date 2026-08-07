"""Human-in-the-loop with a revision loop.

The LLM drafts a repair plan, then an ``Interrupt`` node pauses the run.
If the operator approves (``да``), a final LLM formats the result and the
graph ends.  If they reject or ask for changes (anything else), an
editing LLM rewrites the draft with their feedback and ``flow.loop()``
sends execution back to the same ``Interrupt`` node for another round.

``loop()`` wires the cycle as a conditional edge on ``approved`` — the
resume path is decided after the operator answers, and ``max_iterations``
caps the number of revision rounds.

Requires Ollama running locally with llama3.1:8b.

Usage:
    python examples/human_in_loop/run.py
"""

import asyncio
from typing import TypedDict

from teff.flow import Flow
from teff.node import LLM
from teff.node.interrupt import GraphInterrupt
from teff.provider import ProviderRegistry
from teff.state import State

SYSTEM = "Ты инженер по ремонту. Составляй краткие планы работ по-русски, 3-5 пунктов."


class ApprovalState(TypedDict):
    summ: int
    draft: str
    approved: str
    final: str


async def main():
    flow = (
        Flow(
            "human-in-the-loop",
            providers=ProviderRegistry.from_presets("ollama"),
            default_provider="ollama",
        )
        .llm(
            model="llama3.1:8b",
            system=SYSTEM,
            prompt="Составь план ремонта санузла на сумму {summ} рублей.",
            output_key="draft",
        )
        .interrupt(key="approved", prompt="Одобрить? (да / правки)")
        .loop(
            key="approved",
            until="да",
            done=LLM(
                model="llama3.1:8b",
                system=SYSTEM,
                prompt="Клиент одобрил план: {draft}\nВерни финальную версию плана.",
                output_key="final",
            ),
            body=LLM(
                model="llama3.1:8b",
                system=SYSTEM,
                prompt=(
                    "Клиент отклонил план с правками: {approved}\n\n"
                    "План: {draft}\n\n"
                    "Переработай план с учётом правок клиента и верни новый план."
                ),
                output_key="draft",
            ),
        )
    )

    graph = flow.compile()
    from teff.checkpoint import JSONFileCheckpointer

    cp = JSONFileCheckpointer("checkpoints")

    state = State(ApprovalState, {"summ": 80000})
    resume = None
    while True:
        try:
            result = await graph.run(
                state=state,
                checkpointer=cp,
                checkpoint_id="approval-run",
                max_iterations=10,
                resume=resume,
            )
            break
        except GraphInterrupt as interrupt:
            print(f"\n{interrupt.prompt}\n")
            print(f"--- Draft ---\n{state['draft']}\n")
            answer = input("> ") or "да"
            resume = {"approved": answer}

    print(f"\n--- Final ---\n{result['final']}")
    print(f"\napproved: {result['approved']}")


if __name__ == "__main__":
    asyncio.run(main())
