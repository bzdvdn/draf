"""Supervisor node — routes the conversation to the right agent.

The supervisor loop is **bounded**: every call increments a counter in state
(``supervisor_rounds``), and once it reaches ``max_rounds`` the supervisor
forces ``finish`` without another LLM call.  A model that never picks
``finish`` therefore cannot loop forever — the graph always terminates.
The counter resets on each new user message (it lives in ``_TRANSIENT_KEYS``).
"""

from __future__ import annotations

from draf.harness import Harness
from draf.node.llm import LLM
from draf.node.node import Node

from src.graphs.prompts import SUPERVISOR_PROMPT
from src.nodes.context import _last_user_message

_AGENTS = frozenset({"direct", "planner", "estimator", "materials", "qa", "finish"})


class Supervisor(Node):
    """Decide which agent handles the latest user message.

    Reads the last user message, asks the model which agent fits it best
    (returning a single word), and writes ``next_agent``.  When there is
    no user message to route, or the round counter reached ``max_rounds``,
    the conversation is finished.
    """

    type = "supervisor"

    def __init__(
        self,
        config: dict | None = None,
        *,
        system: str = SUPERVISOR_PROMPT,
        messages_key: str = "messages",
        output_key: str = "next_agent",
        rounds_key: str = "supervisor_rounds",
        max_rounds: int = 6,
        **kwargs,
    ):
        merged = {
            "system": system,
            "messages_key": messages_key,
            "output_key": output_key,
            "rounds_key": rounds_key,
            "max_rounds": max_rounds,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    def _parse_agent(self, text: str) -> str:
        for word in reversed(
            text.strip().lower().replace(",", " ").replace(":", " ").split()
        ):
            w = word.strip(".*\"'»«-")
            if w in _AGENTS:
                return w
        return "finish"

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        messages_key = cfg.get("messages_key", "messages")
        output_key = cfg.get("output_key", "next_agent")
        rounds_key = cfg.get("rounds_key", "supervisor_rounds")
        max_rounds = int(cfg.get("max_rounds", 6))

        last_user = _last_user_message(state.get(messages_key, []))
        if not last_user:
            return {output_key: "finish"}

        # Bounded loop: count supervisor calls and force finish once the
        # budget is spent, so a model that never says "finish" cannot hang.
        rounds = int(state.get(rounds_key) or 0) + 1
        if rounds >= max_rounds:
            return {output_key: "finish", rounds_key: rounds}

        harness = Harness.from_config(cfg, default_provider=LLM.DEFAULT_PROVIDER)
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:

            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        reply = await harness.call(
            [
                {"role": "system", "content": cfg.get("system", "")},
                {"role": "user", "content": last_user},
            ]
        )
        return {output_key: self._parse_agent(reply.content), rounds_key: rounds}
