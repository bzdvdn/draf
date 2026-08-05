"""Self-refine loop: generate, critique, fix, repeat until accepted.

The classic LangGraph "reflection" pattern, built on the low-level Graph
API so the loop is fully visible:

    draft --[draft]--> critic --[critique]--> extract
                                               |
                            verdict=pass ------> final
                            verdict!=pass ------> fix -> critic

- ``draft`` writes a first version.
- ``critic`` returns a structured JSON review (``verdict`` + ``issues``)
  via ``output_type`` — schema-validated, re-asked on malformed JSON.
- ``extract`` turns the review object into a plain string key the edge
  conditions can read (``verdict=pass`` / ``verdict!=pass``).
- ``fix`` rewrites the draft addressing the issues and loops back to the
  critic; ``final`` runs once the critic approves.

By default the run goes against a mocked HTTP transport with a scripted
model: the critic rejects the first draft (``needs_work``) and accepts
the second (``pass``), so exactly one revision happens — no API key and
no Ollama needed.  Set ``DRAF_LIVE=1`` to hit a real Ollama instance.

Usage:
    python examples/self_refine/graph.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TypedDict

import httpx

from draf.graph import Edge, Graph
from draf.logging import configure_logging
from draf.node import LLM, Transform
from draf.provider import ProviderRegistry
from draf.trace import RunTracer

MODEL = "llama3.1:8b"

DRAFT_SYSTEM = "You are a concise, punchy blog writer."
CRITIC_SYSTEM = (
    "You are a strict blog editor. Review the draft and respond with a single "
    'JSON object: {{"verdict": "pass" or "needs_work", "issues": [list of '
    "short problems to fix]}}."
)
FIX_SYSTEM = (
    "You are a concise blog writer. Rewrite the draft addressing every issue "
    "the editor listed. Keep the same topic, fix the problems."
)
FINAL_SYSTEM = (
    "You are a concise blog writer. Produce the final, polished version of the text."
)


class Critique(TypedDict):
    verdict: str
    issues: list[str]


def patch_transport(calls: dict) -> None:
    """Scripted mock transport: draft -> needs_work -> pass (one revision)."""

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
        if "Review the draft" in user_text:
            calls["reviews"] += 1
            if calls["reviews"] == 1:
                content = json.dumps(
                    {
                        "verdict": "needs_work",
                        "issues": ["too long", "no punchline"],
                    }
                )
            else:
                content = json.dumps({"verdict": "pass", "issues": []})
        elif "Rewrite the draft" in user_text:
            content = "Python is a joy. (Draft v2, tight and punchy.)"
        elif "final, polished" in user_text:
            content = "Python is a joy — final version."
        else:
            content = (
                "Python is a great programming language and it is very good "
                "and also really useful for many things."
            )
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
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }

    httpx.AsyncClient.post = mock_post  # type: ignore[method-assign]


def build_graph(model: str) -> Graph:
    draft = LLM(
        model=model,
        system=DRAFT_SYSTEM,
        prompt="Write a short blog post about {topic}.",
        output_key="draft",
        id="draft",
    )
    critic = LLM(
        model=model,
        system=CRITIC_SYSTEM,
        prompt="Review the draft:\n\n{draft}\n\nRespond with JSON.",
        output_key="critique",
        output_type=Critique,
        id="critic",
    )
    extract = Transform(
        action="json_get",
        input_key="critique",
        field="verdict",
        output_key="verdict",
    )
    fix = LLM(
        model=model,
        system=FIX_SYSTEM,
        prompt=(
            "Rewrite the draft addressing these issues: {critique}\n\nDraft:\n{draft}"
        ),
        output_key="draft",
        id="fix",
    )
    final = LLM(
        model=model,
        system=FINAL_SYSTEM,
        prompt="Produce the final, polished version of this draft:\n{draft}",
        output_key="final",
        id="final",
    )

    return Graph(
        nodes={
            "draft": draft,
            "critic": critic,
            "extract": extract,
            "fix": fix,
            "final": final,
        },
        edges=[
            Edge("draft", "critic"),
            Edge("critic", "extract"),
            Edge("extract", "final", condition="verdict=pass"),
            Edge("extract", "fix", condition="verdict!=pass"),
            Edge("fix", "critic"),
        ],
        entry_point="draft",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )


async def main() -> None:
    live = os.environ.get("DRAF_LIVE") == "1"
    calls = {"reviews": 0}
    if not live:
        patch_transport(calls)

    graph = build_graph(MODEL)
    tracer = RunTracer()
    result = await graph.run(
        state={"topic": "why I love Python"},
        tracer=tracer,
        max_iterations=20,
    )

    print("=== Self-refine ===")
    print("Draft  :", result["draft"])
    print("Review :", result["critique"])
    print("Verdict:", result["verdict"])
    print("Final  :", result["final"])
    print(f"\nrevision rounds (critic calls): {calls['reviews']}")
    summary = tracer.summary()
    print(
        f"status: {summary.status}  llm_calls: {summary.llm_calls}  "
        f"tokens: {summary.tokens}"
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
