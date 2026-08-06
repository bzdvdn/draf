"""User-message templates and formatters for the sub-agent tools.

``AgentTool(user_template=..., formatters=...)`` resolves each ``{name}``
placeholder: a name that is a plain state key takes that key's value (empty
when absent), and derived values — like the schema-rendered ``{project_info}``
or the raw ``{last_user_message}`` — resolve through :data:`FORMATTERS`.
All the experts share the project's formatter, so the ``Проект:`` block reads
the same schema-backed facts everywhere.
"""

from __future__ import annotations

from collections.abc import Callable

from src.domain.models import project_info_text


def _last_user_message(state: dict) -> str:
    """The most recent user message (drives the extractor)."""
    for message in reversed(state.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


#: Resolve derived ``{name}`` placeholders that aren't plain state keys.
#: ``project_info`` overrides the same-named state key (the raw dict) with a
#: schema-rendered, LLM-readable projection.
FORMATTERS: dict[str, Callable[[dict], str]] = {
    "project_info": project_info_text,
    "last_user_message": _last_user_message,
}


EXTRACT_USER = """\
Сообщение пользователя:
{last_user_message}"""

PLAN_USER = """\
Проект:
{project_info}

Составь план ремонта."""

ESTIMATE_USER = """\
Проект:
{project_info}

План:
{plan}

Рассчитай смету и подбери материалы."""

SELECT_MATERIALS_USER = """\
Запрос пользователя:
{last_user_message}

Проект:
{project_info}

Подбери материалы и предложи 2-3 варианта с ценами."""

QA_USER = """\
Проект:
{project_info}

План:
{plan}

Смета:
{estimate}

Материалы:
{material_findings}

Проверь план, смету и материалы на противоречия и верни вердикт."""

__all__ = ["FORMATTERS", "EXTRACT_USER", "PLAN_USER", "ESTIMATE_USER", "QA_USER"]
