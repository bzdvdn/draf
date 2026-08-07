"""Agent harness: parallel tool calls in one round + error-edge fallback.

``flow.harness()`` builds the LLM↔tool loop as visible graph topology.  The
agent can request several tools in a single turn — they run **concurrently**
(``asyncio.gather``) and all results come back before the model continues.

This example also shows ``tool_error_mode="raise"``: instead of turning a
failed tool into a ``tool`` message, the failure propagates and routes into
an ``__error__`` edge pointing at a fallback node.

Requires Ollama running locally with llama3.1:8b.

Usage:
    ollama pull llama3.1:8b
    python examples/harness_agent/main.py
"""

import asyncio
import time

from teff.flow import Flow
from teff.graph import Edge, Graph
from teff.node import Transform
from teff.node.agent import ReActAgent, ToolExec
from teff.provider import ProviderRegistry
from teff.tool import Tool

# Tiny fictional "upstream" database. The chat model has never seen it, so
# it MUST call the tools to answer.
CITIES = {
    "oslo": {"weather": "5°C, light rain", "population": "700,000"},
    "tokyo": {"weather": "31°C, sunny", "population": "14,000,000"},
}


class CityWeather(Tool):
    name = "city_weather"
    description = "Get the current weather for a city (e.g. 'oslo')"

    async def arun(self, city: str = "") -> str:  # type: ignore[override]
        await asyncio.sleep(0.5)  # simulate a slow upstream API
        data = CITIES.get(city.strip().lower())
        if data is None:
            raise KeyError(f"unknown city: {city!r}")
        return f"{city}: {data['weather']}"


class CityPopulation(Tool):
    name = "city_population"
    description = "Get the population for a city (e.g. 'tokyo')"

    async def arun(self, city: str = "") -> str:  # type: ignore[override]
        await asyncio.sleep(0.5)
        data = CITIES.get(city.strip().lower())
        if data is None:
            raise KeyError(f"unknown city: {city!r}")
        return f"{city}: {data['population']}"


async def run_parallel() -> None:
    """Two tools requested in one turn run side by side (single round)."""
    flow = Flow(
        "harness_parallel",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.harness(
        model="llama3.1:8b",
        system=(
            "You compare cities. When asked to compare, call BOTH "
            "city_weather and city_population in the SAME turn, then "
            "answer from the results. Never call them one at a time."
        ),
        input_key="query",
        output_key="answer",
    )
    graph = flow.compile()

    t0 = time.monotonic()
    result = await graph.run(
        state={"query": "Compare Oslo and Tokyo: which is hotter and which is bigger?"},
        tools=[CityWeather(), CityPopulation()],
        max_iterations=10,
    )
    elapsed = time.monotonic() - t0

    print("=== Parallel tool calls ===")
    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print(f"Total run: {elapsed:.2f}s (two 0.5s tools finish in one round)")
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    print(f"Tool results: {len(tool_msgs)}")


async def run_error_fallback() -> None:
    """A failing tool raises and routes to an __error__ fallback node."""
    graph = Graph(
        nodes={
            "agent": ReActAgent(
                {
                    "model": "llama3.1:8b",
                    "system": "You answer geography questions. Use the tools you have.",
                    "input_key": "query",
                    "output_key": "answer",
                }
            ),
            "tool": ToolExec({"tool_error_mode": "raise"}),
            "fallback": Transform(
                {
                    "action": "value",
                    "value": "Could not answer — tool failed.",
                    "output_key": "answer",
                }
            ),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
            Edge("tool", "fallback", "__error__"),
        ],
        entry_point="agent",
        default_provider="ollama",
    )

    print("\n=== tool_error_mode='raise' + __error__ edge ===")
    result = await graph.run(
        state={"query": "What is the weather in Atlantis?"},
        tools=[CityWeather()],
        max_iterations=5,
    )
    print("Query:", result["query"])
    print("Answer:", result["answer"])  # 'Atlantis' is not in CITIES -> fallback


async def main() -> None:
    await run_parallel()
    await run_error_fallback()


if __name__ == "__main__":
    asyncio.run(main())
