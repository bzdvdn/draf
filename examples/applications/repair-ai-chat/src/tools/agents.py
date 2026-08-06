"""Sub-agent tools: domain experts the coordinator drives like ordinary tools.

Each tool is an :class:`~draf.tool.agent.AgentTool` — a short ReAct loop
(:class:`~draf.harness.Harness`) against a slice of the domain tool set.
The coordinator sees them as plain tools; the runtime injects ``__state__``
/ ``__ctx__`` (see :func:`draf.harness.tools`), so a tool reads the shared
workflow state, runs its own LLM loop, and writes the result back — the
sub-agent is invisible in the graph topology.
"""

from __future__ import annotations

from draf.tool.agent import AgentTool

from src.domain.models import (
    PROJECT_INFO_LABELS,
    PROJECT_INFO_SCHEMA,
    merge_project_info,
)
from src.graphs.prompts import (
    ESTIMATOR_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    MATERIALS_PROMPT,
    PLANNER_PROMPT,
    QA_VERDICT_SYSTEM,
)
from src.graphs.schemas import QA_VERDICT_SCHEMA
from src.nodes.extractor import room_from_first_user
from src.tools.user_templates import (
    ESTIMATE_USER,
    EXTRACT_USER,
    FORMATTERS,
    PLAN_USER,
    QA_USER,
    SELECT_MATERIALS_USER,
)


class ExtractProjectInfo(AgentTool):
    """Extract the project facts (room, area, budget, style) from the chat."""

    name = "extract_project_info"
    description = (
        "Извлечь информацию о ремонте из сообщения пользователя: тип "
        "помещения, площадь, высота потолков, бюджет, стиль. Результат "
        "сохраняется в состояние проекта. Вызови первой."
    )
    system = EXTRACTOR_SYSTEM_PROMPT
    max_rounds = 3
    user_template = EXTRACT_USER
    formatters = FORMATTERS

    def handle_reply(self, state: dict, reply) -> str:
        extracted = self.json_reply(reply, PROJECT_INFO_SCHEMA) or {}
        merged = merge_project_info(state.get("project_info"), extracted)
        room = extracted.get("room_type")
        if not room:
            room = room_from_first_user(state)
            if room:
                merged["room_type"] = room
        state["project_info"] = merged
        if not merged:
            return "Не удалось извлечь информацию о проекте."
        return "Извлечено: " + "; ".join(
            f"{PROJECT_INFO_LABELS[k]}: {v}" for k, v in merged.items()
        )


class ProposePlan(AgentTool):
    """Compose the step-by-step renovation plan."""

    name = "propose_plan"
    description = (
        "Составить поэтапный план ремонта на основе информации о проекте. "
        "Результат сохраняется в состояние как план. Вызови после "
        "extract_project_info."
    )
    system = PLANNER_PROMPT
    max_rounds = 8
    writes = ("plan",)
    user_template = PLAN_USER
    formatters = FORMATTERS


class SelectMaterials(AgentTool):
    """Pick materials (tiles, paint, laminate, ...) with prices from the catalog."""

    name = "select_materials"
    description = (
        "Подобрать материалы (плитка, краска, ламинат, шпаклёвка и т.п.) "
        "с ценами из каталога по конкретному запросу пользователя. Вызови, "
        "когда пользователь спрашивает про материалы или называет "
        "конкретный материал — без полного процесса сметы."
    )
    system = MATERIALS_PROMPT
    max_rounds = 10
    writes = ("material_findings",)
    user_template = SELECT_MATERIALS_USER
    formatters = FORMATTERS


class PrepareEstimate(AgentTool):
    """Compute the estimate and pick materials in one sub-agent loop."""

    name = "prepare_estimate"
    description = (
        "Рассчитать смету (площади, объёмы, стоимость) и подобрать "
        "материалы с ценами на основе плана. Результаты сохраняются в "
        "состояние. Вызови после одобрения плана."
    )
    system = f"{ESTIMATOR_PROMPT}\n\n{MATERIALS_PROMPT}"
    max_rounds = 12
    writes = ("estimate", "material_findings")
    user_template = ESTIMATE_USER
    formatters = FORMATTERS


class RunQaCheck(AgentTool):
    """Check the plan / estimate / materials for contradictions."""

    name = "run_qa_check"
    description = (
        "Проверить план, смету и материалы на противоречия. Возвращает OK, "
        "если всё корректно, или список замечаний. Вызови после "
        "prepare_estimate; при замечаниях исправь смету и проверь снова."
    )
    system = QA_VERDICT_SYSTEM
    max_rounds = 3
    user_template = QA_USER
    formatters = FORMATTERS

    def handle_reply(self, state: dict, reply) -> str:
        verdict = self.json_reply(reply, QA_VERDICT_SCHEMA) or {}
        ok = bool(verdict.get("ok"))
        message = str(verdict.get("message") or "")
        state["qa_feedback"] = message
        if ok:
            return "OK: план, смета и материалы согласованы."
        return f"Замечания: {message}"


__all__ = [
    "ExtractProjectInfo",
    "ProposePlan",
    "SelectMaterials",
    "PrepareEstimate",
    "RunQaCheck",
]
