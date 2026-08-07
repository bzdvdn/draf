"""Deep research, built with the Flow builder instead of raw Graph edges.

The same workflow as ``graph.py`` — planner → ReAct web research →
writer → reviewer → extract, looping through ``fix`` until the reviewer
accepts — but assembled with :class:`teff.flow.Flow`.  ``flow.react()``
replaces the hand-wired ``ReActAgent <-> ToolExec`` cycle and
``flow.loop()`` replaces the review/fix back edge:

    flow.step(planner)          # writes "topics_obj" (JSON object)
    flow.step(extract_topics)   # raw json_get -> state["topics"] (list)
    flow.react(model=..., input_key="topics", output_key="notes",
               use_tools=["web_search", "fetch_url"])
    flow.step(writer)           # writes "report"
    flow.step(reviewer)         # writes "critique"
    flow.step(extract)          # writes "verdict"          <- decider
    flow.loop(
        key="verdict", until="pass",
        done=final, body=[fix, reviewer],
    )

``loop`` wires ``extract --verdict=pass--> final`` and
``extract --verdict!=pass--> fix -> reviewer -> extract``.

The default run is fully offline: the LLM transport is mocked and the
``web_search`` / ``fetch_url`` tools are stand-ins returning canned
snippets, so it needs no API key, no Ollama, and no network.  Set
``TEFF_LIVE=1`` to use a real Ollama instance and the real DuckDuckGo
search + URL fetch tools.

This file is fully self-contained (duplicates the mock from ``graph.py``)
so it can be read and run on its own.

Usage:
    python examples/deep_research/flow.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TypedDict

import httpx

from teff.flow import Flow
from teff.graph import Graph
from teff.logging import configure_logging
from teff.node import LLM, Transform
from teff.provider import ProviderRegistry
from teff.tool import Tool
from teff.trace import RunTracer

MODEL = "llama3.1:8b"
TOPIC = "launching a personal blog in 2026"

STEPS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}

PLANNER_SYSTEM = (
    "You are a research planner. Respond with a single JSON object of the "
    'form {{"topics": ["question 1", "question 2", ...]}} with 2-3 specific '
    "research questions to answer. No prose, only the JSON object."
)
RESEARCH_SYSTEM = (
    "You are a web research agent. Use the web_search tool to search and the "
    "fetch_url tool to read pages until you can answer the user's questions, "
    "then reply with a short notes summary of your findings."
)
WRITER_SYSTEM = (
    "You are a research writer. Turn the research notes into a clear, "
    "well-structured report with sections and concrete facts."
)
REVIEWER_SYSTEM = (
    "You are a strict research editor. Review the report and respond with a "
    'single JSON object: {{"verdict": "pass" or "needs_work", "feedback": '
    '"one short sentence telling the writer what to fix"}}.'
)
FINAL_SYSTEM = "You are the research lead. Produce the final polished research summary."


class Review(TypedDict):
    verdict: str
    feedback: str


class FakeWebSearch(Tool):
    """Offline stand-in for the real ``web_search`` tool."""

    name = "web_search"
    description = "Search the web (offline stub)"

    async def arun(self, query: str = "", num_results: int = 5) -> str:  # type: ignore[override]
        return (
            f"Top results for {query!r}:\\n"
            "- 2026 blogging guide: start with a niche, post weekly, use SEO\\n"
            "- platform comparison: static sites vs managed hosting\\n"
            "- monetization: newsletter + ads + products"
        )


class FakeWebFetch(Tool):
    """Offline stand-in for the real ``fetch_url`` tool."""

    name = "fetch_url"
    description = "Fetch a URL and extract its text (offline stub)"

    async def arun(self, url: str = "", max_chars: int = 10000) -> str:  # type: ignore[override]
        return (
            f"Content of {url}: a practical 2026 guide to launching a blog "
            "covering hosting, writing cadence, and growing an audience."
        )


def build_tools(live: bool) -> list[Tool]:
    if live:
        return [
            __import__(
                "teff.tool.builtin.web_search", fromlist=["WebSearchTool"]
            ).WebSearchTool(),
            __import__(
                "teff.tool.builtin.web_fetch", fromlist=["WebFetchTool"]
            ).WebFetchTool(),
        ]
    return [FakeWebSearch(), FakeWebFetch()]


def patch_transport(calls: dict) -> None:
    """Scripted mock LLM transport: research loop + one report revision."""

    async def mock_post(*_args, **_kwargs):
        body = _kwargs.get("json") or {}
        messages = body.get("messages") or []
        user_text = next(
            (
                str(m.get("content", ""))
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )

        if body.get("tools"):
            if any(m.get("role") == "tool" for m in messages):
                content = (
                    "Research notes: a successful 2026 blog starts with a "
                    "focused niche, a weekly posting cadence, and a simple "
                    "static-site stack. Monetization follows audience."
                )
            else:
                return _Reply(
                    "",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps(
                                    {"query": "2026 blogging guide"}
                                ),
                            },
                        }
                    ],
                )
        elif "Break this into 2-3 research questions" in user_text:
            content = json.dumps(
                {
                    "topics": [
                        "What stack should a 2026 blog use?",
                        "How often should I post?",
                        "How do blogs make money?",
                    ]
                }
            )
        elif "Review the report" in user_text:
            calls["reviews"] += 1
            if calls["reviews"] == 1:
                content = json.dumps(
                    {
                        "verdict": "needs_work",
                        "feedback": "add a monetization section",
                    }
                )
            else:
                content = json.dumps({"verdict": "pass", "feedback": ""})
        elif "Rewrite the report" in user_text:
            content = (
                "Report v2: A 2026 blog plan — pick a niche, post weekly on a "
                "static-site stack, and monetize via newsletter and ads."
            )
        elif "final polished research summary" in user_text:
            content = (
                "Final summary: launch a niche blog, post weekly, keep the "
                "stack simple, and monetize once the audience grows."
            )
        else:
            content = (
                "Report v1: Start a blog in 2026 by picking a niche and a "
                "hosting platform. Write consistently and promote on social."
            )
        return _Reply(content)

    class _Reply:
        def __init__(self, content: str, tool_calls: list | None = None):
            self._content = content
            self._tool_calls = tool_calls or []

        def raise_for_status(self):
            return None

        def json(self):
            message = {"role": "assistant", "content": self._content}
            if self._tool_calls:
                message["tool_calls"] = self._tool_calls  # type: ignore
            return {
                "message": message,
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 15},
            }

    httpx.AsyncClient.post = mock_post  # type: ignore[method-assign]


def build_flow(model: str, live: bool) -> Graph:
    tool_names = [t.name for t in build_tools(live)]

    planner = LLM(
        model=model,
        system=PLANNER_SYSTEM,
        prompt="Break this into 2-3 research questions: {topic}",
        output_key="topics_obj",
        json_schema=STEPS_SCHEMA,
    )
    extract_topics = Transform(
        action="json_get",
        input_key="topics_obj",
        field="topics",
        output_key="topics",
        raw=True,
    )
    writer = LLM(
        model=model,
        system=WRITER_SYSTEM,
        prompt="Write a report from these research notes:\n\n{notes}",
        output_key="report",
    )
    reviewer = LLM(
        model=model,
        system=REVIEWER_SYSTEM,
        prompt="Review the report:\n\n{report}\n\nRespond with JSON.",
        output_key="critique",
        output_type=Review,
    )
    extract = Transform(
        action="json_get",
        input_key="critique",
        field="verdict",
        output_key="verdict",
    )
    fix = LLM(
        model=model,
        system=WRITER_SYSTEM,
        prompt=(
            "Rewrite the report addressing the editor's feedback: {critique}\n\n"
            "Report:\n{report}"
        ),
        output_key="report",
    )
    final = LLM(
        model=model,
        system=FINAL_SYSTEM,
        prompt="Produce the final polished research summary from this report:\n{report}",
        output_key="final",
    )
    flow = (
        Flow(
            "deep_research",
            providers=ProviderRegistry.from_presets("ollama"),
            default_provider="ollama",
        )
        .llm(node=planner, id="planner")
        .transform(node=extract_topics, id="extract_topics")
        .react(
            model=model,
            system=RESEARCH_SYSTEM,
            input_key="topics",
            output_key="notes",
            use_tools=tool_names,
            max_tool_rounds=6,
            id="research",
        )
        .llm(node=writer, id="writer")
        .llm(node=reviewer, id="reviewer")
        .transform(node=extract, id="extract")
        .loop(
            key="verdict",
            until="pass",
            done=final,
            body=[fix, reviewer],
        )
    )
    return flow.compile()


async def main() -> None:
    live = os.environ.get("TEFF_LIVE") == "1"
    calls = {"reviews": 0}
    if not live:
        patch_transport(calls)

    graph = build_flow(MODEL, live)
    tracer = RunTracer()
    result = await graph.run(
        state={"topic": TOPIC},
        tools=build_tools(live),
        tracer=tracer,
        max_iterations=40,
    )

    print("=== Deep research (Flow API) ===")
    print("Questions :", result["topics"])
    print("Notes     :", result["notes"])
    print("Report    :", result["report"])
    print("Review    :", result["critique"])
    print("Verdict   :", result["verdict"])
    print("Final     :", result["final"])
    print(f"\nreview rounds: {calls['reviews']}")
    summary = tracer.summary()
    print(
        f"status: {summary.status}  llm_calls: {summary.llm_calls}  "
        f"tokens: {summary.tokens}"
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
