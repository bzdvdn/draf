"""Supervisor node — routes the conversation to the right agent.

HOW TO EXTEND
    Keep this node as-is for the standard pattern.  If you change the agent
    names in ``graphs/build.py``, mirror them in the ``_AGENTS`` set below
    (and in the supervisor prompt), and pass the matching ``route_keys`` /
    ``done_keys`` so the decider can finish deterministically.

The supervisor loop is **bounded**: every call increments a counter in state
(``supervisor_rounds``), and once it reaches ``max_rounds`` the supervisor
forces ``finish`` without another LLM call.  A model that never picks
``finish`` therefore cannot loop forever — the graph always terminates.
The counter resets on each new user message (it lives in ``_TRANSIENT_KEYS``).

Two deterministic guards keep the loop from burning that budget:

* **done_keys** — once the listed output slots are filled (``done_mode`` is
  ``"all"`` or ``"any"``), the supervisor returns ``finish`` *without*
  another LLM call: the answer already exists.
* **route_keys** — a map of route value -> output slot.  If the model picks
  an agent whose slot already has content, that would just overwrite finished
  work, so the pick is ignored and the loop finishes instead.

The decider's user message now carries the accumulated progress (the
``sections``), the current round and the latest user message, so the model
can see what already exists and route — or finish — accordingly.
"""

from __future__ import annotations

from draf.harness import Harness
from draf.node.llm import LLM
from draf.node.node import Node
from src.graphs.prompts import SUPERVISOR_PROMPT
from src.nodes.context import last_user_message

#: The only route values this decider may produce.
_AGENTS = frozenset({"coder", "talk", "finish"})


class Supervisor(Node):
    """Decide which agent handles the latest user message.

    Reads the last user message (plus any work already produced), asks the
    model which agent fits it best (a single word), and writes
    ``next_agent``.  When there is no user message to route, the round
    counter reached ``max_rounds``, or the ``done_keys`` are already filled,
    the conversation is finished without another LLM call.
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
        sections: dict[str, str] | None = None,
        route_keys: dict[str, str] | None = None,
        done_keys: set[str] | None = None,
        done_mode: str = "all",
        fallback_agent: str = "",
        **kwargs,
    ):
        merged = {
            "system": system,
            "messages_key": messages_key,
            "output_key": output_key,
            "rounds_key": rounds_key,
            "max_rounds": max_rounds,
            "sections": sections or {},
            "route_keys": route_keys or {},
            "done_keys": set(done_keys or ()),
            "done_mode": done_mode,
            "fallback_agent": fallback_agent,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    def _parse_agent(self, text: str) -> str:
        for word in reversed(text.strip().lower().split()):
            w = word.strip(".*\"'»«-")
            if w in _AGENTS:
                return w
        return "finish"

    def _progress_text(self, state: dict) -> str:
        parts = []
        for key, label in self.config.get("sections", {}).items():
            value = state.get(key)
            if value:
                parts.append(f"{label}:\n{value}")
        return "\n\n".join(parts)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        messages_key = cfg.get("messages_key", "messages")
        output_key = cfg.get("output_key", "next_agent")
        rounds_key = cfg.get("rounds_key", "supervisor_rounds")
        max_rounds = int(cfg.get("max_rounds", 6))

        last_user = last_user_message(state.get(messages_key, []))
        if not last_user:
            return {output_key: "finish"}

        # Bounded loop: count supervisor calls and force finish once the
        # budget is spent, so a model that never says "finish" cannot hang.
        rounds = int(state.get(rounds_key) or 0) + 1
        if rounds >= max_rounds:
            return {output_key: "finish", rounds_key: rounds}

        # Deterministic finish: the required output slots are filled, so an
        # answer already exists — no further LLM call is needed.
        done_keys = set(cfg.get("done_keys") or ())
        if done_keys:
            filled = [k for k in done_keys if state.get(k)]
            if cfg.get("done_mode", "all") == "any":
                done = bool(filled)
            else:
                done = len(filled) == len(done_keys)
            if done:
                return {output_key: "finish", rounds_key: rounds}

        user_parts = []
        progress = self._progress_text(state)
        if progress:
            user_parts.append(progress)
        user_parts.append(f"Round: {rounds}/{max_rounds}")
        user_parts.append(f"User: {last_user}")

        harness = Harness.from_config(cfg, default_provider=LLM.DEFAULT_PROVIDER)
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:

            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        reply = await harness.call(
            [
                {"role": "system", "content": cfg.get("system", "")},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ]
        )
        agent = self._parse_agent(reply.content)

        # Don't end a turn that produced nothing: if the model picks "finish"
        # (or returns text that fails to parse to a known agent, which also
        # resolves to "finish") before any output slot is filled, route to
        # *fallback_agent* instead so the user still gets a real answer.
        fallback_agent = cfg.get("fallback_agent") or ""
        if (
            agent == "finish"
            and fallback_agent
            and not any(state.get(k) for k in done_keys)
        ):
            agent = fallback_agent

        # No-rerun guard: an agent whose output slot already has content must
        # not be re-routed — that would just overwrite finished work.
        route_keys = cfg.get("route_keys") or {}
        if agent in route_keys and state.get(route_keys[agent]):
            return {output_key: "finish", rounds_key: rounds}

        return {output_key: agent, rounds_key: rounds}
