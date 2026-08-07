"""Support-triage example — run one ticket against a local Ollama.

The supervisor routes the ticket to a billing or technical specialist; a
difficult ticket goes to a human and **pauses** until the operator answers
(durable via the file checkpointer — resume works across processes).

Usage::

    python examples/recipes/support_triage/main.py
"""

from __future__ import annotations

import asyncio

from support_triage.graphs.build import build_flow
from support_triage.graphs.state import TriageState, initial_state

from teff.checkpoint import JSONFileCheckpointer
from teff.node.interrupt import GraphInterrupt
from teff.state import State


async def main() -> None:
    graph = build_flow().compile()
    cp = JSONFileCheckpointer("data/checkpoints")

    state = State(TriageState, dict(initial_state()))
    state["ticket"] = (
        "Здравствуйте, с меня списали деньги за услугу, которой я не пользовался. "
        "Верните, пожалуйста!"
    )
    state["messages"] = [{"role": "user", "content": state["ticket"]}]

    resume = None
    while True:
        try:
            result = await graph.run(
                state=state,
                checkpointer=cp,
                checkpoint_id="ticket-1",
                max_iterations=50,
                resume=resume,
            )
            break
        except GraphInterrupt as interrupt:
            print(f"\n[оператор] {interrupt.prompt}")
            answer = input("> ") or "передать старшему оператору"
            resume = {"escalate": answer}

    reply = [m for m in result["messages"] if m.get("role") == "assistant"][-1]
    print(f"\n--- Ответ ---\n{reply['content']}")


if __name__ == "__main__":
    asyncio.run(main())
