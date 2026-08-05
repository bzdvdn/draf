"""Plan-and-execute: an LLM plans, a Map executes, a reviewer decides.

The classic LangGraph "plan-and-execute" pattern.  An LLM first produces a
JSON object of steps; the steps fan out to a ``Map`` that executes each one
(here a per-step LLM call); a reviewer validates the results and either
accepts them or sends control back to the planner for a revised plan:

    planner -> extract_steps -> [Map] executor -> reviewer -> extract
                                                                 |
                                              verdict=pass ------> final
                                              verdict!=pass ------> replan -+
                                                                             |
                                                                             v
                                                                   extract_steps <-+

- ``planner`` returns a JSON object ``{"steps": [...]}`` via an explicit
  ``json_schema`` (a bare top-level array is unreliable with local models).
- ``extract_steps`` pulls the array out of that object into ``state["steps"]``
  (``json_get`` with ``raw=True`` keeps the list as a list).
- ``Map`` fans the list out to parallel executor calls
  (``input_keys=["steps"]``, results collected into ``state["results"]``).
- ``reviewer`` returns ``{"verdict": "pass"|"needs_work", "feedback": "..."}``.
- On ``needs_work`` the ``replan`` node rewrites the plan from the
  reviewer's feedback and the same ``Map`` runs again.

By default the run goes against a mocked HTTP transport (planner's first
plan is rejected once, then accepted) — no API key, no Ollama.  Set
``DRAF_LIVE=1`` to hit a real Ollama instance.

Usage:
    python examples/plan_and_execute/graph.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TypedDict

import httpx

from draf.graph import Edge, Graph
from draf.logging import configure_logging
from draf.node import LLM, Map, Transform
from draf.provider import ProviderRegistry
from draf.trace import RunTracer

MODEL = "llama3.1:8b"
GOAL = "launch a blog about Python"

# A bare top-level JSON array is unreliable with local models (Ollama's
# ``format: json`` forces an object, so the model wraps the array).  Ask
# for an object with a ``steps`` key instead, then extract the array with
# a ``json_get`` transform using ``raw=True`` (so the list stays a list).
STEPS_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

PLANNER_SYSTEM = (
    "You are a project planner. Respond with a single JSON object of the form "
    '{{"steps": ["step 1", "step 2", ...]}} with 2-3 concise step strings. '
    "No prose, only the JSON object."
)
EXECUTOR_SYSTEM = (
    "You are a step executor. Execute the single step and report what you did."
)
REVIEWER_SYSTEM = (
    "You are a quality reviewer. Review the step results and respond with a "
    'single JSON object: {{"verdict": "pass" or "needs_work", "feedback": '
    '"one short sentence telling the planner what to fix"}}.'
)
FINAL_SYSTEM = (
    "You are the delivery lead. Summarize the completed work into a short report."
)


class Review(TypedDict):
    verdict: str
    feedback: str


def patch_transport(calls: dict) -> None:
    """Scripted mock transport: plan rejected once, then accepted."""

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
        if "Plan a short list of steps" in user_text:
            calls["plans"] += 1
            if calls["plans"] == 1:
                content = json.dumps(
                    {
                        "steps": [
                            "research the topic",
                            "draft the first post",
                            "publish",
                        ]
                    }
                )
            else:
                content = json.dumps(
                    {"steps": ["research the topic", "write and publish"]}
                )
        elif "Execute this step" in user_text:
            step = user_text.split("Execute this step: ", 1)[-1].strip()
            content = f"done: {step}"
        elif "Review the results" in user_text:
            calls["reviews"] += 1
            if calls["reviews"] == 1:
                content = json.dumps(
                    {"verdict": "needs_work", "feedback": "split the plan in two"}
                )
            else:
                content = json.dumps({"verdict": "pass", "feedback": ""})
        else:
            content = "Report: all planned steps completed."
        return _Reply(content)

    class _Reply:
        def __init__(self, content: str):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            content = self._content
            return {
                "message": {"role": "assistant", "content": content},
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12},
            }

    httpx.AsyncClient.post = mock_post  # type: ignore[method-assign]


def build_graph(model: str) -> Graph:
    planner = LLM(
        model=model,
        system=PLANNER_SYSTEM,
        prompt="Plan a short list of steps to achieve: {goal}",
        output_key="steps_obj",
        json_schema=STEPS_SCHEMA,
        id="planner",
    )
    extract_steps = Transform(
        action="json_get",
        input_key="steps_obj",
        field="steps",
        output_key="steps",
        raw=True,
    )
    execute = Map(
        LLM(
            model=model,
            system=EXECUTOR_SYSTEM,
            prompt="Execute this step: {steps}",
            output_key="result",
        ),
        input_keys=["steps"],
        output_key="results",
        max_concurrency=2,
        id="execute",
    )
    reviewer = LLM(
        model=model,
        system=REVIEWER_SYSTEM,
        prompt="Review the results:\n\n{results}\n\nRespond with JSON.",
        output_key="review",
        output_type=Review,
        id="reviewer",
    )
    extract = Transform(
        action="json_get",
        input_key="review",
        field="verdict",
        output_key="verdict",
    )
    replan = LLM(
        model=model,
        system=PLANNER_SYSTEM,
        prompt=(
            "Plan a short list of steps to achieve: {goal}\n\n"
            "The reviewer said: {review}\n\nRespond with a revised JSON array."
        ),
        output_key="steps_obj",
        json_schema=STEPS_SCHEMA,
        id="replan",
    )
    final = LLM(
        model=model,
        system=FINAL_SYSTEM,
        prompt="Summarize the outcome:\n\nResults: {results}",
        output_key="final",
        id="final",
    )

    return Graph(
        nodes={
            "planner": planner,
            "extract_steps": extract_steps,
            "execute": execute,
            "reviewer": reviewer,
            "extract": extract,
            "replan": replan,
            "final": final,
        },
        edges=[
            Edge("planner", "extract_steps"),
            Edge("extract_steps", "execute"),
            Edge("execute", "reviewer"),
            Edge("reviewer", "extract"),
            Edge("extract", "final", condition="verdict=pass"),
            Edge("extract", "replan", condition="verdict!=pass"),
            Edge("replan", "extract_steps"),
        ],
        entry_point="planner",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )


async def main() -> None:
    live = os.environ.get("DRAF_LIVE") == "1"
    calls = {"plans": 0, "reviews": 0}
    if not live:
        patch_transport(calls)

    graph = build_graph(MODEL)
    tracer = RunTracer()
    result = await graph.run(
        state={"goal": GOAL},
        tracer=tracer,
        max_iterations=30,
    )

    print("=== Plan-and-execute ===")
    print("Final plan :", result["steps"])
    print("Results    :", result["results"])
    print("Review     :", result["review"])
    print("Verdict    :", result["verdict"])
    print("Report     :", result["final"])
    print(f"\nplanning rounds: {calls['plans']}  review rounds: {calls['reviews']}")
    summary = tracer.summary()
    print(
        f"status: {summary.status}  llm_calls: {summary.llm_calls}  "
        f"tokens: {summary.tokens}"
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
