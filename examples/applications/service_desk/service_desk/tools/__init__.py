"""Tool registry for the service-desk specialists.

Each specialist is scoped to a single RAG tool over the shared knowledge
base; the supervisor and the coordinator graph never call them directly.
"""

from __future__ import annotations

from service_desk.tools.knowledge import (
    SearchBillingKnowledge,
    SearchDeployKnowledge,
    SearchIncidentKnowledge,
)

#: Tool name per specialist slot (used by ``agent_step(use_tools=...)``).
KNOWLEDGE_TOOLS = {
    "incident": "search_incident_knowledge",
    "billing": "search_billing_knowledge",
    "deploy": "search_deploy_knowledge",
}


def build_tools(knowledge) -> list:
    """Instantiate the RAG tools against the shared knowledge base."""
    return [
        SearchIncidentKnowledge(knowledge),
        SearchBillingKnowledge(knowledge),
        SearchDeployKnowledge(knowledge),
    ]


__all__ = ["KNOWLEDGE_TOOLS", "build_tools"]
