"""RAG tools over the service-desk knowledge base, one per specialist.

Each tool views the shared :class:`~service_desk.rag.knowledge.KnowledgeBase` through
a fixed ``domain`` filter, so a specialist can only retrieve its own
knowledge (the LLM cannot leak incidents into billing and vice versa).
"""

from __future__ import annotations

from teff.tool.tool import Tool


class _KnowledgeTool(Tool):
    """Base: ask the shared KB for a fixed domain."""

    domain = ""

    def __init__(self, knowledge):
        super().__init__()
        self.knowledge = knowledge

    async def arun(  # type: ignore[override]
        self, query: str = "", top_k: int = 3
    ) -> str:
        return await self.knowledge.search(query or "", domain=self.domain, top_k=top_k)


class SearchIncidentKnowledge(_KnowledgeTool):
    name = "search_incident_knowledge"
    domain = "incidents"
    description = (
        "Поиск по базе знаний об инцидентах (известные проблемы, симптомы, "
        "шаги устранения). Вызови, когда пользователь описывает сбой/ошибку, "
        "чтобы свериться с известными решениями. Аргумент query — симптом "
        "или описание проблемы."
    )


class SearchBillingKnowledge(_KnowledgeTool):
    name = "search_billing_knowledge"
    domain = "billing"
    description = (
        "Поиск по базе знаний о счетах и платежах (правила возвратов, "
        "сроки списаний, тарифы, документы). Вызови, когда нужна актуальная "
        "политика биллинга. Аргумент query — тема вопроса."
    )


class SearchDeployKnowledge(_KnowledgeTool):
    name = "search_deploy_knowledge"
    domain = "deploy"
    description = (
        "Поиск по базе знаний о выкатке (чек-листы релизов, откат, миграции "
        "БД, взаимные действия). Вызови, чтобы сверить план с регламентом. "
        "Аргумент query — сценарий выкатки."
    )


__all__ = [
    "SearchBillingKnowledge",
    "SearchDeployKnowledge",
    "SearchIncidentKnowledge",
]
