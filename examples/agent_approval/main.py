"""Tool approval (human-in-the-loop) for a ReAct agent.

The ``tool_approval`` gate decides what the agent may run.  A callable can
``approve`` / ``deny`` a tool call; returning ``"pause"`` raises a
:class:`GraphInterrupt` so a human can inspect the request and resume the
run with an answer.

Three variants of the same workflow:

1. **Flow API** (recommended) — ``flow.harness(tool_approval=...)`` wires the
   agent ↔ tool loop for you (one-shot pause).
2. **Low-level Graph** — the same loop built by hand with ``ReActAgent`` +
   ``ToolExec`` nodes and explicit edges, exactly what ``flow.harness()``
   expands to.
3. **Revision loop** — ``flow.loop()`` with an ``Interrupt`` decider: a
   ``deny`` feeds the agent feedback and **returns to the prompt** until the
   call is approved.

Usage:
    # Make sure Ollama is running and has the model
    ollama pull llama3.1:8b
    python examples/agent_approval/main.py
"""

import asyncio
import os
import tempfile

from draf.checkpoint import JSONFileCheckpointer
from draf.flow import Flow
from draf.node import Node
from draf.node.interrupt import GraphInterrupt
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


async def human_approval(name, args):
    """Always pause and ask a human for sign-off."""
    return "pause"


def build_flow_graph():
    """Idiomatic version — ``flow.harness()`` wires the agent loop."""
    flow = Flow(
        "approval_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.harness(
        model="llama3.1:8b",
        input_key="input",
        output_key="answer",
        tool_approval=human_approval,
    )
    return flow.compile()


def build_low_level_graph():
    """Low-level equivalent — the exact wiring ``flow.harness()`` builds."""
    from draf.graph import Edge, Graph
    from draf.node.agent import ReActAgent, ToolExec

    return Graph(
        nodes={
            "agent": ReActAgent(
                {"model": "llama3.1:8b", "input_key": "input", "output_key": "answer"}
            ),
            "tool": ToolExec({"tool_approval": human_approval}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        default_provider="ollama",
    )


async def run_with_approval(graph, label: str) -> None:
    """Run *graph*, pausing for approval, then resume with the answer."""
    cp = JSONFileCheckpointer(os.path.join(tempfile.mkdtemp(), "checkpoints.json"))
    state = {"input": "What is 123 * 456?"}
    run_id = "approval-demo"

    try:
        await graph.run(
            state,
            tools=[Calculator()],
            max_iterations=10,
            checkpointer=cp,
            checkpoint_id=run_id,
        )
    except GraphInterrupt as interrupt:
        print(f"\n--- paused: {interrupt.prompt} ---")
        decision = input("Approve? [y/N] ").strip().lower()
        value = "approve" if decision in ("y", "yes") else "deny"
        # the decision is written into the state under the interrupt key
        result = await graph.run(
            state,
            tools=[Calculator()],
            max_iterations=10,
            checkpointer=cp,
            checkpoint_id=run_id,
            resume={"tool_approval": value},
        )
        print(f"[{label}] Answer:", result.get("answer"))


class DenyFeedback(Node):
    """Tell the agent its tool call was denied so it re-plans."""

    type = "deny_feedback"

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__(**(config or {}), **kwargs)

    async def execute(self, ctx, state):
        calls = state.get("_tool_calls") or []
        messages = list(state.get("messages", []))
        for call in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": (
                        f"Tool call '{call.get('name')}' was not approved. "
                        "Rework your plan and try again."
                    ),
                }
            )
        state["messages"] = messages
        state["_tool_calls"] = []
        state["_tool_call_name"] = ""
        return state


def build_loop_graph():
    """`flow.loop()` decider: deny loops back to the prompt until approved."""
    from draf.node import LLM
    from draf.node.agent import ReActAgent, ToolExec

    agent_cfg = {"model": "llama3.1:8b", "input_key": "input", "output_key": "answer"}
    answer_cfg = {
        "model": "llama3.1:8b",
        "messages_key": "messages",
        "output_key": "answer",
        "use_tools": False,
    }

    flow = Flow(
        "approval_loop",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.step(ReActAgent(agent_cfg))  # emits a tool call
    flow.interrupt("tool_approval", prompt="Approve the tool call?")
    flow.loop(
        key="tool_approval",
        until="approve",
        # approved → run the tool, then a plain LLM answers from the result
        done=[ToolExec({}), LLM(answer_cfg)],
        # denied → feedback, agent re-plans, then the loop asks again
        body=[DenyFeedback(), ReActAgent(agent_cfg)],
    )
    return flow.compile()


async def run_with_loop() -> None:
    """Pause/re-ask in a loop: `deny` returns to the prompt until approved."""
    cp = JSONFileCheckpointer(os.path.join(tempfile.mkdtemp(), "checkpoints.json"))
    graph = build_loop_graph()
    state = {"input": "What is 123 * 456?"}
    run_id = "approval-loop"
    resume = None
    result = None

    while True:
        try:
            result = await graph.run(
                state,
                tools=[Calculator()],
                max_iterations=10,
                checkpointer=cp,
                checkpoint_id=run_id,
                resume=resume,
            )
            break
        except GraphInterrupt as interrupt:
            print(f"\n--- paused: {interrupt.prompt} ---")
            decision = input("Approve? [y/N] ").strip().lower()
            resume = {
                "tool_approval": "approve" if decision in ("y", "yes") else "deny"
            }

    print("[loop] Answer:", result.get("answer"))


async def main():
    print("== Flow API (one-shot pause) ==")
    await run_with_approval(build_flow_graph(), "flow")

    print("\n== Low-level Graph (what flow.harness() expands to) ==")
    await run_with_approval(build_low_level_graph(), "graph")

    print("\n== flow.loop() decider (deny returns to the prompt) ==")
    await run_with_loop()


if __name__ == "__main__":
    asyncio.run(main())
