"""Run the complex supervisor example offline (no API key, no Ollama).

A scripted httpx transport plays every role in the team — the team lead
routes planner → coder, the reviewer first rejects the code, the Map fixes
the issues, the loop re-reviews until "pass", and the operator approves the
interrupt gate with "отлично" (any_of [approve, отлично, ok]).

Usage:
    python examples/supervisor_complex/run.py
"""

import asyncio
import json

import httpx

from teff.checkpoint import JSONFileCheckpointer
from teff.node.interrupt import GraphInterrupt
from teff.yaml import load_workflow

WORKFLOW = "examples/supervisor_complex/workflow.yaml"
CHECKPOINT = "/tmp/supervisor-complex-cp"


def _scripted_transport() -> None:
    """Scripted Ollama replies dispatched on the system-prompt text.

    The reviewer rejects the first review ("needs_work") and accepts the
    re-review ("pass"), so exactly one refinement round runs.
    """

    supervisor_answers = iter(["planner", "coder"])
    reviewer_calls = {"n": 0}

    async def mock_post(*args, **kwargs):
        body = kwargs.get("json") or {}
        system = "".join(
            str(m.get("content", ""))
            for m in body.get("messages") or []
            if m.get("role") == "system"
        )
        if "team lead" in system:
            content = next(supervisor_answers)
        elif "senior planner" in system:
            content = (
                "1. Разбить задачу на функции\n2. Определить сигнатуры\n3. Написать код"
            )
        elif "senior developer" in system:
            content = "def calculate_total(items):\n    return sum(items)"
        elif "fix exactly one code issue" in system:
            content = "# исправлено\nreturn sum(float(x) for x in items)"
        elif "apply the reviewer's fixes" in system:
            content = (
                "def calculate_total(items):\n    return sum(float(x) for x in items)"
            )
        elif "strict code reviewer" in system:
            reviewer_calls["n"] += 1
            if reviewer_calls["n"] == 1:
                content = json.dumps(
                    {"verdict": "needs_work", "issues": ["добавить проверку типов"]}
                )
            else:
                content = json.dumps({"verdict": "pass", "issues": []})
        else:
            content = ""

        class _Reply:
            def raise_for_status(self):
                return None

            def json(self):
                return {"message": {"role": "assistant", "content": content}}

        return _Reply()

    httpx.AsyncClient.post = mock_post  # type: ignore[method-assign]


async def main() -> None:
    _scripted_transport()
    graph, tools, state, reducers = load_workflow(WORKFLOW)

    cp = JSONFileCheckpointer(CHECKPOINT)
    try:
        await graph.run(
            state,
            tools=tools,
            reducers=reducers,
            checkpointer=cp,
            checkpoint_id="offline",
        )
    except GraphInterrupt as interrupt:
        print(f"  {interrupt.prompt}")
        state = await graph.run(
            state,
            tools=tools,
            reducers=reducers,
            checkpointer=cp,
            checkpoint_id="offline",
            resume={"approved": "отлично"},
        )

    print(f"  plan:      {state['plan']!r}")
    print(f"  code:      {state['code']!r}")
    print(f"  verdict:   {state['verdict']!r}")
    print(f"  decision:  {state['decision']!r}")
    print(f"  delivered: {state['delivered_at']!r}")


if __name__ == "__main__":
    asyncio.run(main())
