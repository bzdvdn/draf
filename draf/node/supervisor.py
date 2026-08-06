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
  *fallback_agent* so the user still gets a real answer;
* **fill_order** — a ``[(agent, slot), ...]`` pipeline.  The model picks the
  entry agent once, then the chain runs in order (each missing slot → its
  agent) and finishes when every slot is full; a mid-chain agent picked
  directly runs once and finishes.  No subclass needed for the common
  "run the agents in order, then finish" pattern.

The finish token is configurable via ``finish`` (default ``"finish"``) — for
prompts that spell out their own terminator such as ``<end>``.  The parser
normalizes enclosing punctuation and ``<>`` on both sides, so a model replying
``<finish>`` still matches the default token.

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
from draf.node.node import Node


class Supervisor(Node):
    """Decide which agent handles the latest user message.

    Reads the last user message (plus any work already produced), asks the
    model which agent fits it best (a single word), and writes the chosen
    route to ``output_key``.  When the round counter reached ``max_rounds``
    or the ``done_keys`` are already filled, the conversation is finished
    without another model call.

    ``fill_order`` turns the supervisor into a deterministic pipeline
    without a subclass: the model picks only the *entry* agent, then every
    mid-pipeline round runs the chain in order (``planner`` → ``estimator``
    → ... → ``finish``) with no further model calls.  A mid-chain agent
    picked directly (a targeted question) runs once and finishes.  See
    ``examples/applications/repair-ai-chat`` for a chat that routes a
    ``direct`` branch through ``done_keys`` while chaining the repair
    agents through ``fill_order``.

    ``finish`` renames the terminator token the model answers with (default
    ``"finish"``); the same value is written to ``output_key`` for the
    ``finish`` route branch.  Set it to whatever your system prompt tells
    the model to reply, e.g. ``finish="<end>"``.
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
        finish: str = "finish",
        fill_order: list[tuple[str, str]] | None = None,
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
            "finish": finish,
            "fill_order": [(agent, slot) for agent, slot in (fill_order or [])],
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
        chain = [agent for agent, _ in (self.config.get("fill_order") or [])]
        fallback = self.config.get("fallback_agent") or ""
        return (
            set(route_keys)
            | set(chain)
            | {self._finish()}
            | ({fallback} if fallback else set())
        )

    def _finish(self) -> str:
        """The token that ends the conversation (``finish`` by default)."""
        return self.config.get("finish") or "finish"

    def _parse_agent(self, text: str) -> str:
        """Return the last word of *text* that matches the agent vocabulary.

        Both sides are normalized by stripping enclosing punctuation and
        ``<>``, so a model that wraps its reply as ``<finish>`` matches the
        configured ``finish`` token and vice versa.  The canonical spelling
        from the vocabulary is returned.
        """
        canonical = {}
        for agent in self._agents():
            canonical[agent.strip(" .*\"'»«-<>").lower()] = agent
        for word in reversed(
            text.strip().lower().replace(",", " ").replace(":", " ").split()
        ):
            w = word.strip(" .*\"'»«-<>")
            if w in canonical:
                return canonical[w]
        return ""

    def _progress_text(self, state: dict) -> str:
        """Render the non-empty ``sections`` as ``Label:\\n<value>`` blocks."""
        parts = []
        for key, label in (self.config.get("sections") or {}).items():
            value = state.get(key)
            if value:
                parts.append(f"{label}:\n{value}")
        return "\n\n".join(parts)

    def _done(self, state: dict) -> bool:
        """Whether the ``done_keys`` guard is satisfied (no model needed).

        ``done_mode="all"`` requires every key non-empty, ``"any"`` requires
        at least one.
        """
        done_keys = set(self.config.get("done_keys") or ())
        if not done_keys:
            return False
        filled = [k for k in done_keys if state.get(k)]
        if self.config.get("done_mode", "all") == "any":
            return bool(filled)
        return len(filled) == len(done_keys)

    def _chain_route(self, state: dict) -> str:
        """The deterministic route a configured ``fill_order`` prescribes.

        Returns ``""`` when the model must still pick the entry agent.
        Once the entry slot is filled the pipeline runs the rest of the
        chain in order and finishes when every slot is full.  A mid-chain
        agent picked directly (a targeted question) runs once and finishes
        without dragging the whole pipeline in.
        """
        order = self.config.get("fill_order") or []
        if not order:
            return ""
        entry_slot = order[0][1]
        if state.get(entry_slot):
            for agent, slot in order:
                if not state.get(slot):
                    return agent
            return self._finish()
        for agent, slot in order[1:]:
            if state.get(slot):
                return self._finish()
        return ""

    def _needs_model(self, state: dict) -> bool:
        """Whether the model must be consulted this round.

        Default (chat routing): the model is needed only when there is a user
        message to route and no ``done_keys`` are already filled.  Set
        ``messages_key=""`` to always consult the model, or override this in
        a subclass whose :meth:`decide` resolves some states deterministically.
        With a ``fill_order`` the model is needed only for the entry decision;
        every mid-pipeline round is resolved deterministically.
        """
        cfg = self.config
        if self._done(state):
            return False
        if cfg.get("fill_order"):
            return not self._chain_route(state)
        messages_key = cfg.get("messages_key")
        if not messages_key:
            return True
        return bool(last_user_message(state.get(messages_key, [])))

    def decide(self, state: dict, proposal: str) -> str:
        """Resolve the route from the parsed *proposal* plus the guards.

        Default implements the chat guards on top of the model's single word:
        a filled ``done_keys`` set short-circuits to ``finish``, a premature
        ``finish`` falls back to *fallback_agent*, and a ``route_keys`` agent
        whose slot is already filled is not re-routed.  With a ``fill_order``
        the mid-pipeline route is deterministic (see :meth:`_chain_route`);
        only the entry decision comes from the model.  Subclasses override
        this for a deterministic policy; *proposal* is ``""`` when the model
        was not consulted.
        """
        proposal = proposal or self._finish()
        cfg = self.config
        finish = self._finish()
        if self._done(state):
            return finish
        fallback = cfg.get("fallback_agent") or ""
        if cfg.get("fill_order"):
            route = self._chain_route(state)
            if route:
                return route
            if proposal in self._agents() and proposal != finish:
                return proposal
            return fallback or cfg["fill_order"][0][0]
        if (
            proposal == finish
            and fallback
            and not any(state.get(k) for k in cfg.get("done_keys") or ())
        ):
            return fallback
        route_keys = cfg.get("route_keys") or {}
        if proposal in route_keys and state.get(route_keys[proposal]):
            return finish
        return proposal

    async def _ask_model(
        self, ctx, state: dict, *, rounds: int, max_rounds: int
    ) -> str:
        """Render the context, call the model, return the parsed proposal."""
        cfg = self.config
        harness = Harness.from_config(
            cfg,
            default_provider=getattr(ctx, "default_provider", None),
            default_model=getattr(ctx, "default_model", None),
            providers=getattr(ctx, "providers", None),
        )
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:

            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        payload_sink = getattr(ctx, "on_llm_payload", None)
        if payload_sink is not None:
            harness.on_llm_payload = payload_sink

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
            return {output_key: self._finish(), rounds_key: rounds}

        if not self._needs_model(state):
            return {output_key: self.decide(state, ""), rounds_key: rounds}

        proposal = await self._ask_model(
            ctx, state, rounds=rounds, max_rounds=max_rounds
        )
        return {output_key: self.decide(state, proposal), rounds_key: rounds}
