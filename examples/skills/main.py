"""Skills: reusable instructions + tool scoping loaded from a folder.

A skill is a folder with a ``SKILL.md`` (Agent Skills layout).  Mounting
it on a harness agent does two things:

- its instructions are merged into the system prompt;
- its ``allowed-tools`` / ``disallowed-tools`` narrow which tools the
  model can see — here ``secret_tool`` is registered but the model never
  learns about it, because the skill forbids it.

Layout::

    examples/skills/
    ├── skills/
    │   └── city-guide/SKILL.md
    └── main.py

Requires Ollama running locally with llama3.1:8b.

Usage:
    ollama pull llama3.1:8b
    python examples/skills/main.py
"""

import asyncio
from pathlib import Path

from teff.flow import Flow
from teff.provider import ProviderRegistry
from teff.tool import Tool

SKILL_DIR = Path(__file__).resolve().parent / "skills"

CITIES = {
    "oslo": {"weather": "5°C, light rain", "population": "700,000"},
    "tokyo": {"weather": "31°C, sunny", "population": "14,000,000"},
}


class CityWeather(Tool):
    name = "city_weather"
    description = "Get the current weather for a city (e.g. 'oslo')"

    def run(self, city: str = "") -> str:  # type: ignore[override]
        data = CITIES.get(city.strip().lower())
        if data is None:
            return f"Error: unknown city {city!r}"
        return f"{city}: {data['weather']}"


class CityPopulation(Tool):
    name = "city_population"
    description = "Get the population for a city (e.g. 'tokyo')"

    def run(self, city: str = "") -> str:  # type: ignore[override]
        data = CITIES.get(city.strip().lower())
        if data is None:
            return f"Error: unknown city {city!r}"
        return f"{city}: {data['population']}"


class SecretTool(Tool):
    name = "secret_tool"
    description = "Internal tool the agent must never use"

    def run(self, **kwargs) -> str:  # type: ignore[override]
        return "classified"


async def main() -> None:
    flow = Flow(
        "skills_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.harness(
        model="llama3.1:8b",
        input_key="query",
        output_key="answer",
        skills=["city-guide"],
        skill_dir=str(SKILL_DIR),
    )
    graph = flow.compile()

    result = await graph.run(
        state={"query": "Which city is hotter: Oslo or Tokyo?"},
        tools=[CityWeather(), CityPopulation(), SecretTool()],
        max_iterations=10,
    )
    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print("Skill mounted: city-guide (secret_tool scoped out)")


if __name__ == "__main__":
    asyncio.run(main())
