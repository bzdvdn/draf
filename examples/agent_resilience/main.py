"""Agent resilience: retries, failover, context management, token budget.

Shows the production harness knobs on a ReAct agent:

- ``max_retries`` / ``retry_on`` — retry failed HTTP requests (429/5xx/
  timeouts) with exponential backoff.
- ``fallbacks`` — fail over to backup models when the primary is down.
- ``tool_timeout`` / ``tool_retries`` — bound and re-attempt tool calls.
- ``max_context_tokens`` — trim the conversation before each model call.
- ``max_total_tokens`` — stop the loop once the token budget is spent.

The run goes against a mocked HTTP transport, so it needs no real API key:
the primary model is always down (503) and the run recovers by retrying and
then failing over to the backup model.

Two variants of the same workflow:

1. **Flow API** (recommended) — ``flow.harness(...)`` wires the agent loop.
2. **Low-level Graph** — the same loop with explicit ``ReActAgent`` +
   ``ToolExec`` nodes, exactly what ``flow.harness()`` expands to.

Set ``DRAF_LIVE=1`` to hit a real OpenAI-compatible endpoint instead (e.g.
Ollama on ``LLM_BASE_URL``); that path uses the Flow API against a real
model.

Usage:
    python examples/agent_resilience/main.py
"""

import asyncio
import os

from draf.flow import Flow
from draf.provider import ProviderRegistry
from draf.tool import Tool


class Echo(Tool):
    name = "echo"
    description = "Return the input text"

    def run(self, text: str = "") -> str:
        return text


def patch_httpx(calls: dict) -> None:
    """Mock transport: the primary model is down; the backup answers."""

    import httpx

    class MockResponse:
        def __init__(self, *, down: bool):
            self._down = down

        def raise_for_status(self):
            calls["n"] += 1
            if self._down:
                req = httpx.Request("POST", "http://x")
                raise httpx.HTTPStatusError(
                    "503", request=req, response=httpx.Response(503, request=req)
                )

        def json(self):
            return {"choices": [{"message": {"content": "resilient answer"}}]}

    async def mock_post(*a, **kw):
        model = (kw.get("json") or {}).get("model", "")
        return MockResponse(down=(model == "gpt-4"))

    httpx.AsyncClient.post = mock_post  # type: ignore[method-assign]


RESILIENCE_KNOBS = dict(
    max_tool_rounds=5,
    max_retries=3,
    retry_on=[429, 500, 502, 503, 504],
    fallbacks=["gpt-4-backup"],
    tool_timeout=30,
    tool_retries=1,
    max_context_tokens=8000,
    max_total_tokens=40000,
)


def run_flow() -> None:
    """Idiomatic version — ``flow.harness()`` wires the agent loop."""
    os.environ["OPENAI_API_KEY"] = "sk-mock"
    calls = {"n": 0}
    patch_httpx(calls)

    flow = Flow(
        "resilient_agent",
        providers=ProviderRegistry.from_presets("openai"),
        default_provider="openai",
    )
    flow.harness(
        model="gpt-4",
        input_key="input",
        output_key="answer",
        **RESILIENCE_KNOBS,
    )
    graph = flow.compile()

    result = asyncio.run(
        graph.run(state={"input": "retry me"}, tools=[Echo()], max_iterations=10)
    )
    print("Answer:", result["answer"])
    print(
        "HTTP attempts (4 retries on the down primary + 1 on the backup):", calls["n"]
    )


def run_graph() -> None:
    """Low-level equivalent — the exact wiring ``flow.harness()`` builds."""
    from draf.graph import Edge, Graph
    from draf.node.agent import ReActAgent, ToolExec

    os.environ["OPENAI_API_KEY"] = "sk-mock"
    calls = {"n": 0}
    patch_httpx(calls)

    agent_cfg = {"model": "gpt-4", "input_key": "input", "output_key": "answer"}
    agent_cfg.update(RESILIENCE_KNOBS)

    graph = Graph(
        nodes={
            "agent": ReActAgent(agent_cfg),
            "tool": ToolExec({"tool_timeout": 30, "tool_retries": 1}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        provider="openai",
    )

    result = asyncio.run(
        graph.run(state={"input": "retry me"}, tools=[Echo()], max_iterations=10)
    )
    print("Answer:", result["answer"])
    print(
        "HTTP attempts (4 retries on the down primary + 1 on the backup):", calls["n"]
    )


def run_live() -> None:
    """Real model on an OpenAI-compatible endpoint (e.g. Ollama)."""
    flow = Flow(
        "resilient_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.harness(
        model="llama3.1:8b",
        input_key="query",
        output_key="answer",
        **{**RESILIENCE_KNOBS, "fallbacks": ["qwen2.5:1.5b"]},
    )
    graph = flow.compile()

    result = asyncio.run(
        graph.run(state={"query": "Summarise the last commit in two lines."})
    )
    print("Answer:", result["answer"])


def main():
    if os.environ.get("DRAF_LIVE"):
        run_live()
        return

    print("== Flow API ==")
    run_flow()
    print("\n== Low-level Graph (what flow.harness() expands to) ==")
    run_graph()


if __name__ == "__main__":
    main()
