"""Supervisor graph builder — a staged, human-in-the-loop repair pipeline.

The graph runs a **one-shot** supervisor that routes the incoming request to
either a direct assistant reply or a repair-planning pipeline.  The pipeline
is staged so the human approves both the plan and the estimate:

    supervisor ─ next_agent=direct ─► direct agent (terminal)
    supervisor ─ next_agent=pipeline ─► planner
    planner ──► interrupt("plan_approved")  ──► asks the user
        plan_approved=да  ─► extractor (collect project info)
        plan_approved≠да  ─► planner ─────────────────────┘  (re-plan, ask again)
    extractor ─► estimator ─► materials ─► QA fix-loop
    ──► interrupt("estim_approved") ──► asks the user
        estim_approved=да  ─► assemble final План/Смета answer
        estim_approved≠да  ─► estimator ─► materials ─► QA fix-loop ─┘  (re-calc, re-ask)

Both interrupts are *top-level* nodes (not nested in a SubFlow), which is
required for the checkpoint/resume bookkeeping to work.  Each approval is a
:class:`~draf.node.ask.Ask` ``model`` strategy: the free-form answer is
normalized by a verdict ``LLM`` into ``{ok: bool}`` and decoded by a
:class:`~draf.node.ask.Validate` node into a ``flow.loop`` decider, so
"конечно", "ок", "хорошо" all count as approval.  The QA fix loop lives as
a self-contained SubFlow whose verdict (``LLM(json_schema=...)``) is
decoded by :class:`~draf.node.Gate` into a ``flow.loop`` decider; a ``fix``
verdict re-runs estimator/materials (the plan is already approved, so Planner
is *not* re-run) and the loop terminates once QA passes.  The deterministic
План/Смета answer is appended last by the estimate-approval ``done`` chain,
so ``last_reply`` returns it.
"""

from __future__ import annotations

from draf.flow import Case, Flow, SubFlow, agent_step
from draf.node import LLM, AppendAssistant, Ask, Gate, Transform
from draf.provider import ProviderRegistry
from src.core.deps import build_deps
from src.graphs.prompts import (
    DIRECT_PROMPT,
    ESTIMATE_APPROVAL_PROMPT,
    ESTIMATE_VERDICT_SYSTEM,
    ESTIMATE_VERDICT_USER,
    ESTIMATOR_PROMPT,
    MATERIALS_PROMPT,
    PLAN_APPROVAL_PROMPT,
    PLAN_VERDICT_SYSTEM,
    PLAN_VERDICT_USER,
    PLANNER_PROMPT,
    QA_VERDICT_SYSTEM,
    QA_VERDICT_USER,
    SUPERVISOR_PROMPT,
)
from src.graphs.schemas import (
    ESTIMATE_VERDICT_SCHEMA,
    PLAN_VERDICT_SCHEMA,
    QA_VERDICT_SCHEMA,
)
from src.nodes.extractor import Extractor
from src.tools import build_tools

MODEL_DEFAULT = "llama3.1:8b"

#: Shared state keys rendered into each agent's context.
AGENT_SECTIONS = {
    "project_info": "Проект",
    "plan": "План",
    "estimate": "Смета",
    "material_findings": "Материалы",
    "qa_feedback": "Замечания QA",
}

_ROOM_TOOLS = [
    "calculate_wall_area",
    "calculate_floor_area",
    "calculate_ceiling_area",
    "calculate_perimeter",
]
_MATERIAL_TOOLS = [
    "calculate_tiles",
    "calculate_paint",
    "calculate_laminate",
    "calculate_plaster",
    "calculate_putty",
]
_BUDGET_TOOLS = ["estimate_material_cost", "estimate_total"]
_RAG_TOOLS = ["search_materials", "find_similar_material"]

def build_flow(
    model: str = MODEL_DEFAULT,
    *,
    provider: str = "ollama",
    services=None,
    catalog=None,
):
    """Assemble the staged supervisor graph.

    Flow::

        supervisor ─ next_agent=direct ──► direct agent (terminal)
        supervisor ─ next_agent=pipeline ─► planner
        planner ─► interrupt("plan_approved") ─► asks the human
            plan_approved=да  ─► estimator ─► materials ─► qa-fix-loop
            plan_approved≠да  ─► planner ────────────────┘ (re-plan + re-ask)

    The ``qa`` stage is a self-contained fix loop: the structured QA verdict
    (``LLM(json_schema=QA_VERDICT_SCHEMA)``) is decoded by a
    :class:`~draf.node.Gate` into a ``flow.loop`` decider value. A ``faix``
    verdict re-runs estimator/materials/qa (the user already approved the
    plan, and the non-planning body. Once QA passes (or after ``max_rounds``)
    a deterministic ``Transform`` assembles the final План/Смета message,
    appended last so ``last_reply`` returns it.

    *provider* is threaded into every agent's harness config (per-node) so
    the graph never touches the framework's global defaults.
    """
    services = services or build_deps(provider=provider)[0]
    catalog = catalog if catalog is not None else build_deps(provider=provider)[1]
    tools = build_tools(services, catalog)

    planner = agent_step(
        system=PLANNER_PROMPT,
        output_key="plan",
        use_tools=_ROOM_TOOLS,
        model=model,
        provider=provider,
        sections=AGENT_SECTIONS,
        id="planner",
    )
    estimator = agent_step(
        system=ESTIMATOR_PROMPT,
        output_key="estimate",
        use_tools=[*_ROOM_TOOLS, *_MATERIAL_TOOLS, *_BUDGET_TOOLS],
        model=model,
        provider=provider,
        sections=AGENT_SECTIONS,
        id="estimator",
    )
    materials = agent_step(
        system=MATERIALS_PROMPT,
        output_key="material_findings",
        use_tools=_RAG_TOOLS,
        model=model,
        provider=provider,
        sections=AGENT_SECTIONS,
        id="materials",
    )

    qa_verdict = LLM(
        system=QA_VERDICT_SYSTEM,
        prompt=QA_VERDICT_USER,
        output_key="qa_verdict",
        json_schema=QA_VERDICT_SCHEMA,
        model=model,
        provider=provider,
    )
    qa_gate = Gate(
        input_key="qa_verdict",
        output_key="qa_ok",
        rounds_key="qa_rounds",
        message_key="qa_feedback",
        max_rounds=3,
    )

    # The QA fix loop is a self-contained SubFlow: verdict -> gate -> re-run
    # estimator/materials until it passes.  Its ``done`` just records the pass;
    # the final План/Смета answer is assembled by the estimate-approval chain.

    qa_loop = (
        Flow("qa-fix-loop")
        .llm(qa_verdict, id="qa-fix-loop-verdict")
        .step(qa_gate, id="qa-fix-loop-gate")
        .loop(
            key="qa_ok",
            until="yes",
            done=[
                Transform(
                    action="value",
                    value="passed",
                    output_key="qa_passed",
                    id="qa-fix-loop-done",
                )
            ],
            body=[estimator, materials, qa_verdict],
        )
    )

    flow = Flow(
        "production-repair-ai", providers=ProviderRegistry.from_presets(provider)
    )
    flow.supervisor(
        system=SUPERVISOR_PROMPT,
        model=model,
        provider=provider,
        sections=AGENT_SECTIONS,
        agents={"direct", "pipeline", "finish"},
        id="supervisor",
    )
    flow.branch(
        "next_agent",
        Case("direct").add(
            agent_step(
                system=DIRECT_PROMPT,
                output_key="direct_reply",
                model=model,
                provider=provider,
                sections=AGENT_SECTIONS,
                id="direct",
            ),
            id="direct",
        ),
        Case("pipeline").add(planner, id="pipeline-planner"),
    )
    # Top-level interrupt: pausing for the human to approve the plan.  Its
    # resume value lands in "plan_approved" by the caller passing
    # resume={"plan_approved": ...} — the Ask.model strategy then normalizes
    # the free-form answer (да/конечно/хорошо → pass, нет → fail) and the
    # Validate node decodes it into the loop decider.  Approving runs the
    # extractor to collect project info; rejecting re-plans.
    flow.interrupt_loop(
        key="plan_approved",
        prompt=PLAN_APPROVAL_PROMPT,
        accept=Ask.model(
            system=PLAN_VERDICT_SYSTEM,
            user=PLAN_VERDICT_USER,
            schema=PLAN_VERDICT_SCHEMA,
            model=model,
            provider=provider,
            verdict_key="plan_verdict",
            decision_key="plan_ok",
            rounds_key="plan_rounds",
        ),
        id="plan-approval",
        body=planner,
        done=[Extractor(model=model, provider=provider, id="extractor")],
    )
    # Once the plan is approved: estimate -> materials -> QA (fix loop).
    flow.step(estimator, id="estimator")
    flow.step(materials, id="materials")
    qa_subflow = SubFlow(qa_loop.compile(), id_prefix="qa")
    flow.step(qa_subflow, id="qa-subflow")
    # Second top-level interrupt: the human approves the checked estimate.
    # A "нет" re-runs estimator/materials and the QA fix loop before asking
    # again; a "да" assembles and appends the final План/Смета answer.
    flow.interrupt_loop(
        key="estim_approved",
        prompt=ESTIMATE_APPROVAL_PROMPT,
        accept=Ask.model(
            system=ESTIMATE_VERDICT_SYSTEM,
            user=ESTIMATE_VERDICT_USER,
            schema=ESTIMATE_VERDICT_SCHEMA,
            model=model,
            provider=provider,
            verdict_key="est_verdict",
            decision_key="est_ok",
            rounds_key="est_rounds",
        ),
        id="est-approval",
        body=[estimator, materials, qa_subflow],
        done=[
            Transform(
                action="render",
                template="План:\n{plan}\n\nСмета:\n{estimate}",
                output_key="final_answer",
                id="final-transform",
            ),
            AppendAssistant(output_key="final_answer", id="final-append-assistant"),
        ],
    )
    return flow, tools
