"""Extract the final assistant reply from a completed turn and shape the
channel response.

The graph's durable ``messages`` list is the canonical source, but a node
may store its answer under a dedicated output key (``answer``/``summary``)
without appending it to ``messages`` — e.g. a plain ``llm_chat`` step.
``reply_text`` walks both so every channel (HTTP, webhook, Telegram)
returns the same, best-effort text for any workflow shape.

Precedence (via :func:`reply_from_state`):
1. Well-known output keys on the finished state (``answer``, ``summary``,
   ``output``, ``reply``).
2. The last assistant message in ``state["messages"]`` when present.

``reply_text`` additionally prefers ``result.reply`` (the graph's own
extraction from ``messages``) when the run populated it.

All channels share one response shape — :func:`turn_response` — so a turn
is always ``{"session_id", "waiting", "message"}``::

    {"session_id": "...", "waiting": false, "message": "<the reply>"}
    {"session_id": "...", "waiting": true, "message": "<the prompt>"}

When ``waiting`` the client resumes simply by sending the operator's answer
as the next turn ``message``; ``key`` (the interrupt's state key) is
optional metadata only.
"""

from __future__ import annotations

from typing import Any

from teff.graph import TurnResult

_REPLY_KEYS = ("answer", "summary", "output", "reply")


def reply_from_state(state: dict | None) -> str:
    """Extract the best-effort assistant reply from a finished *state*."""
    if not state:
        return ""
    for key in _REPLY_KEYS:
        value = state.get(key)
        if isinstance(value, str) and value:
            return value
    messages = state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
    return ""


def reply_text(result: TurnResult) -> str:
    """Return the best-effort assistant reply for *result*.

    ``""`` when the turn is paused (``waiting``) or produced no text.
    """
    if result.waiting:
        return ""
    if result.reply:
        return result.reply
    return reply_from_state(result.state)


def turn_response(result: TurnResult, session_id: str) -> dict[str, Any]:
    """Shape one turn into the channel response format.

    A paused turn carries the interrupt prompt as ``message`` plus the
    optional ``key``; a completed turn carries the final reply.  Used by
    every channel so HTTP, webhook and Telegram answer identically.
    """
    if result.waiting:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "waiting": True,
            "message": result.prompt or "",
        }
        if result.key:
            payload["key"] = result.key
        return payload
    return {
        "session_id": session_id,
        "waiting": False,
        "message": reply_text(result),
    }
