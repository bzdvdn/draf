"""Release coordinator — a supervisor graph with a human approval gate.

This example wires together almost every high-level Flow feature at once,
as a release-coordinator that routes three specialist agents, fans feature
checks out in parallel, and pauses for human approval before shipping:

* a **supervisor** decider that routes to one of three specialised agents
  (planner / estimator / tester) and, once all sections are ready, to a
  *human approval* step;
* a single ``Supervisor`` node — the core
  :class:`draf.node.Supervisor`, subclassed with the release policy: it
  renders the section progress into the model context, gets a *proposal*,
  then enforces the policy (``approved``, then fill order) and bounds the
  loop with a ``supervisor_rounds`` counter — so a stray free-form answer
  can neither end the graph silently nor hang it;
* three **agent_step** sub-flows, each given a stable ``id=`` so the
  compiled graph reads like the domain, not ``subflow_3``;
* a **map** that fans the feature list out to parallel checks;
* an **interrupt** that pauses for approval — and, crucially, lives
  *inside* the supervisor cycle, so rejecting the release routes control
  back through the supervisor to re-plan (``нет`` -> planner).

The approval gate is part of the ``route()`` loop, not a separate
``loop()``: when the supervisor is told to *approve*, it runs the fan-out
checks and then the ``Interrupt`` node.  After the operator answers, the
run resumes straight back at the supervisor, which reads ``approved``:
``да`` -> ``finish`` (final summary); ``нет`` -> ``planner`` (re-plan).

Runs against a local Ollama model (llama3.1:8b):

    ollama pull llama3.1:8b
    python examples/release_coordinator/main.py
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from draf.flow import Flow, agent_step
from draf.logging import configure_logging
from draf.node import LLM, Interrupt, Map
from draf.node import Supervisor as BaseSupervisor
from draf.node.interrupt import GraphInterrupt
from draf.provider import ProviderRegistry
from draf.trace import RunTracer

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

AGENT_SECTIONS = {
    "plan": "План",
    "estimate": "Оценка",
    "tests": "Тесты",
}

SUPERVISOR_SYSTEM = (
    "Ты супервайзор выпуска релиза. Смотри на состояние и отвечай ровно "
    "одним словом из: planner, estimator, tester, approve, finish.\n"
    "- если approved равно 'да' — ответь: finish\n"
    "- если approved равно 'нет' — ответь: planner (пересмотреть план)\n"
    "- если план пуст — ответь: planner\n"
    "- если оценка пуста — ответь: estimator\n"
    "- если тесты пусты — ответь: tester\n"
    "- иначе — ответь: approve"
)
PLANNER_SYSTEM = (
    "Ты планировщик релиза. По описанию фич составь краткий план работ"
    " на русском, 2-3 пункта."
)
ESTIMATOR_SYSTEM = "Ты оценщик. Оцени трудозатраты и бюджет на основе плана."
TESTER_SYSTEM = "Ты тестировщик. Перечисли результаты прогона тестов."


class Supervisor(BaseSupervisor):
    """Release-coordinator decider — the core ``Supervisor`` plus a policy.

    The core :class:`draf.node.Supervisor` renders the section progress into
    the model context, parses a single-word *proposal* and bounds the loop
    with a ``supervisor_rounds`` counter.  This subclass keeps that machinery
    and overrides the decision hooks with the release fill-order policy:

    * ``approved == 'да'`` -> ``finish``;
    * ``approved == 'нет'`` -> ``planner`` — and the rejection is consumed
      (``approved`` is cleared), so the revised release is re-submitted for
      approval instead of being re-planned forever;
    * missing ``plan`` / ``estimate`` / ``tests`` -> the next one in order;
    * everything filled -> the model is consulted: it may ``approve``, request
      more work on a section, or (never) ship — ``finish`` without approval
      is blocked.

    The states the policy fixes never reach the model: only
    everything-filled-but-unapproved consults :meth:`_ask_model`.
    """

    _AGENTS = frozenset({"planner", "estimator", "tester", "approve", "finish"})

    def __init__(
        self,
        config: dict | None = None,
        *,
        system: str = SUPERVISOR_SYSTEM,
        model: str = "",
        provider: str = "",
        sections: dict[str, str] | None = None,
        output_key: str = "next_agent",
        rounds_key: str = "supervisor_rounds",
        max_rounds: int = 30,
        **kwargs,
    ):
        super().__init__(
            config=config,
            system=system,
            model=model,
            provider=provider,
            messages_key="",
            sections=sections,
            output_key=output_key,
            rounds_key=rounds_key,
            max_rounds=max_rounds,
            agents=self._AGENTS,
            **kwargs,
        )

    def _route(self, state: dict) -> str:
        """Return the route the policy fixes deterministically (no model)."""
        approved = state.get("approved", "")
        if approved == "да":
            return "finish"
        if approved == "нет":
            return "planner"
        if not state.get("plan"):
            return "planner"
        if not state.get("estimate"):
            return "estimator"
        if not state.get("tests"):
            return "tester"
        return "approve"

    def _needs_model(self, state: dict) -> bool:
        """Consult the model only when everything is filled but unapproved."""
        return self._route(state) == "approve"

    def decide(self, state: dict, proposal: str) -> str:
        route = self._route(state)
        if route != "approve":
            return route
        # Everything is filled but not approved: the model may re-route to a
        # section for more work; "approve"/"finish"/garbage all mean approve.
        return proposal if proposal in ("planner", "estimator", "tester") else "approve"

    async def execute(self, ctx, state: dict) -> dict:
        out = await super().execute(ctx, state)
        # Consume the rejection: routing to "planner" from approved == "нет"
        # clears it, so the revised release is re-submitted for approval
        # instead of being re-planned forever.
        if (
            out.get(self.config["output_key"]) == "planner"
            and state.get("approved") == "нет"
        ):
            out["approved"] = ""
        return out


class ReleaseState(TypedDict):
    features: list[str]
    plan: str
    estimate: str
    tests: str
    checks: list[str]
    approved: str
    next_agent: str
    final: str
    supervisor_rounds: int


def build_flow(model: str, provider: str) -> Flow:
    """Assemble the coordinator: supervisor cycle with an approval gate."""

    def agent(system: str, output_key: str, name: str):
        return agent_step(
            system,
            output_key,
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
            id=name,
        )

    flow = Flow(
        "release-coordinator", providers=ProviderRegistry.from_presets(provider)
    )

    # 1. Supervisor loop: planner -> estimator -> tester -> approve -> finish.
    #    One node owns the routing (renders section progress, gets a model
    #    proposal, then enforces the policy deterministically).
    flow.supervisor(
        Supervisor(
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
            system=SUPERVISOR_SYSTEM,
        ),
        id="supervisor",
    )
    flow.route(
        "next_agent",
        finish=LLM(
            model=model,
            provider=provider,
            system="Ты выпускающий. Кратко подведи итог релиза.",
            prompt="План: {plan}\nОценка: {estimate}\nТесты: {tests}\nПроверки: {checks}",
            output_key="final",
            id="final",
        ),
        planner=agent(PLANNER_SYSTEM, "plan", "planner"),
        estimator=agent(ESTIMATOR_SYSTEM, "estimate", "estimator"),
        tester=agent(TESTER_SYSTEM, "tests", "tester"),
        # Approval gate: fan the feature list out to parallel checks, then
        # pause for a human.  After the answer the run resumes at the
        # supervisor, which reads ``approved`` and re-plans or ships.
        approve=[
            Map(
                LLM(
                    model=model,
                    provider=provider,
                    system="Ты инженер релиза. Верни 'ok'.",
                    prompt="Проверь фичу: {features}",
                    output_key="check",
                    id="checker",
                ),
                input_keys=["features"],
                output_key="checks",
                id="mapper",
            ),
            Interrupt(
                key="approved",
                prompt=(
                    "РЕЛИЗ ГОТОВ. Утвердить?\n"
                    "План: {plan}\nОценка: {estimate}\nТесты: {tests}\n"
                    "Проверки: {checks}\n\n(да/нет)"
                ),
                id="interrupt",
            ),
        ],
    )

    return flow


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def main() -> None:
    model = "llama3.1:8b"
    provider = "ollama"

    flow = build_flow(model, provider)
    graph = flow.compile()

    print("=== Compiled node ids (id= in action) ===")
    for nid, node in graph.nodes.items():
        tag = node.type
        if node.type == "subflow":
            inner = sorted(node._graph.nodes)  # type: ignore[attr-defined]
            tag = f"subflow -> {', '.join(inner)}"
        print(f"  {nid:16} {tag}")

    print("\n=== Run ===")
    tracer = RunTracer()
    from draf.checkpoint import JSONFileCheckpointer

    cp = JSONFileCheckpointer("checkpoints")
    await cp.delete("release-run")  # always start from a clean sheet

    state: ReleaseState = {
        "features": ["release API v2", "migrate storage", "add tests"],
        "plan": "",
        "estimate": "",
        "tests": "",
        "checks": [],
        "approved": "",
        "next_agent": "",
        "final": "",
        "supervisor_rounds": 0,
    }
    resume = None
    while True:
        try:
            result = await graph.run(
                state=dict(state),
                checkpointer=cp,
                checkpoint_id="release-run",
                max_iterations=30,
                resume=resume,
                tracer=tracer,
            )
            break
        except GraphInterrupt as interrupt:
            print(f"\n  [pause] {interrupt.prompt}")
            try:
                answer = input("  > (да/нет) ") or "нет"
            except EOFError:  # non-interactive run auto-approves
                answer = "да"
            resume = {"approved": answer}

    summary = tracer.summary()
    print(
        f"\nstatus: {summary.status}  llm_calls: {summary.llm_calls}  tokens: {summary.tokens}"
    )
    print("final:", result.get("final"))
    print("checks:", result.get("checks"))
    print("approved:", result.get("approved"))


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
