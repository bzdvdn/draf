"""Supervisor node — decide which routed agent runs next.

A single node owns the routing decision in a
:meth:`draf.flow.Flow.route` loop: it renders the shared ``sections`` into
the model context, asks the model for a single-word *proposal*, and writes
the chosen route to ``output_key``.  Deterministic guards keep the loop safe
even when the model never says ``finish``:

* **bounded loop** — every call increments ``rounds_key``; once it reaches
  ``max_rounds`` the node returns ``finish`` without another model call;
* **done_keys** — once the listed output slots are filled (``done_mode`` is
  ``"all"`` or ``"any"``) the node returns ``finish`` with no model call:
  the answer already exists;
* **route_keys** — a map route value → output slot.  Picking an agent whose
  slot already has content would just overwrite finished work, so the pick
  is ignored and the loop finishes instead;
* **fallback_agent** — ``finish`` before anything was produced routes to
  *fallback_agent* so the user still gets a real answer.

The decider's user message carries the accumulated progress (``sections``),
the current round and the latest user message, so the model can see what
already exists and route — or finish — accordingly.

Extensibility: subclasses override :meth:`decide` for a deterministic,
state-driven policy and :meth:`_needs_model` to control when the model is
consulted (see ``examples/release_coordinator`` for a fill-order policy).
"""

from __future__ import annotations

from typing import AbstractSet

from draf.harness import Harness
from draf.node.context import last_user_message
from draf.node.llm import LLM
from draf.node.node import Node


class Supervisor(Node):
    """Decide which agent handles the latest user message.

    Reads the last user message (plus any work already produced), asks the
    model which agent fits it best (a single word), and writes the chosen
    route to ``output_key``.  When the round counter reached ``max_rounds``
    or the ``done_keys`` are already filled, the conversation is finished
    without another model call.
    """

    type = "supervisor"

    def __init__(
        self,
        config: dict | None = None,
        *,
        system: str = "",
        model: str = "",
        provider: str = "",
        messages_key: str = "messages",
        output_key: str = "next_agent",
        rounds_key: str = "supervisor_rounds",
        max_rounds: int = 6,
        sections: dict[str, str] | None = None,
        agents: AbstractSet[str] | None = None,
        route_keys: dict[str, str] | None = None,
        done_keys: set[str] | None = None,
        done_mode: str = "all",
        fallback_agent: str = "",
        **kwargs,
    ):
        merged = {
            "system": system,
            "model": model,
            "provider": provider,
            "messages_key": messages_key,
            "output_key": output_key,
            "rounds_key": rounds_key,
            "max_rounds": max_rounds,
            "sections": sections or {},
            "agents": agents,
            "route_keys": route_keys or {},
            "done_keys": set(done_keys or ()),
            "done_mode": done_mode,
            "fallback_agent": fallback_agent,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    def _agents(self) -> set[str]:
        """The single-word vocabulary the model may answer with."""
        agents = self.config.get("agents")
        if agents:
            return set(agents)
        route_keys = self.config.get("route_keys") or {}
        fallback = self.config.get("fallback_agent") or ""
        return set(route_keys) | {"finish"} | ({fallback} if fallback else set())

    def _parse_agent(self, text: str) -> str:
        """Return the last word of *text* that is in the agent vocabulary."""
        agents = self._agents()
        for word in reversed(
            text.strip().lower().replace(",", " ").replace(":", " ").split()
        ):
            w = word.strip(" .*\"'»«-")
            if w in agents:
                return w
        return ""

    def _progress_text(self, state: dict) -> str:
        """Render the non-empty ``sections`` as ``Label:\\n<value>`` blocks."""
        parts = []
        for key, label in (self.config.get("sections") or {}).items():
            value = state.get(key)
            if value:
                parts.append(f"{label}:\n{value}")
        return "\n\n".join(parts)

    def _needs_model(self, state: dict) -> bool:
        """Whether the model must be consulted this round.

        Default (chat routing): the model is needed only when there is a user
        message to route and no ``done_keys`` are already filled.  Set
        ``messages_key=""`` to always consult the model, or override this in
        a subclass whose :meth:`decide` resolves some states deterministically.
        """
        cfg = self.config
        done_keys = set(cfg.get("done_keys") or ())
        if done_keys:
            filled = [k for k in done_keys if state.get(k)]
            if cfg.get("done_mode", "all") == "any":
                done = bool(filled)
            else:
                done = len(filled) == len(done_keys)
            if done:
                return False
        messages_key = cfg.get("messages_key")
        if not messages_key:
            return True
        return bool(last_user_message(state.get(messages_key, [])))

    def decide(self, state: dict, proposal: str) -> str:
        """Resolve the route from the parsed *proposal* plus the guards.

        Default implements the chat guards on top of the model's single word:
        a filled ``done_keys`` set short-circuits to ``finish``, a premature
        ``finish`` falls back to *fallback_agent*, and a ``route_keys`` agent
        whose slot is already filled is not re-routed.  Subclasses override
        this for a deterministic policy; *proposal* is ``""`` when the model
        was not consulted.
        """
        proposal = proposal or "finish"
        cfg = self.config
        done_keys = set(cfg.get("done_keys") or ())
        if done_keys:
            filled = [k for k in done_keys if state.get(k)]
            if cfg.get("done_mode", "all") == "any":
                done = bool(filled)
            else:
                done = len(filled) == len(done_keys)
            if done:
                return "finish"
        fallback = cfg.get("fallback_agent") or ""
        if (
            proposal == "finish"
            and fallback
            and not any(state.get(k) for k in done_keys)
        ):
            return fallback
        route_keys = cfg.get("route_keys") or {}
        if proposal in route_keys and state.get(route_keys[proposal]):
            return "finish"
        return proposal

    async def _ask_model(
        self, ctx, state: dict, *, rounds: int, max_rounds: int
    ) -> str:
        """Render the context, call the model, return the parsed proposal."""
        cfg = self.config
        harness = Harness.from_config(cfg, default_provider=LLM.DEFAULT_PROVIDER)
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:

            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        messages_key = cfg.get("messages_key", "messages")
        user = (
            f"User: {last_user_message(state.get(messages_key, []))}"
            if messages_key
            else ""
        )
        user_parts = [
            part
            for part in (
                self._progress_text(state),
                f"Round: {rounds}/{max_rounds}",
                user,
            )
            if part
        ]
        reply = await harness.call(
            [
                {"role": "system", "content": cfg.get("system", "")},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ]
        )
        return self._parse_agent(reply.content)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        rounds_key = cfg.get("rounds_key", "supervisor_rounds")
        output_key = cfg.get("output_key", "next_agent")
        max_rounds = int(cfg.get("max_rounds", 6))
        rounds = int(state.get(rounds_key) or 0) + 1

        # Bounded loop: a model that never says "finish" cannot hang.
        if rounds >= max_rounds:
            return {output_key: "finish", rounds_key: rounds}

        if not self._needs_model(state):
            return {output_key: self.decide(state, ""), rounds_key: rounds}

        proposal = await self._ask_model(
            ctx, state, rounds=rounds, max_rounds=max_rounds
        )
        return {output_key: self.decide(state, proposal), rounds_key: rounds}
