"""Context management — token estimation and message trimming."""

from __future__ import annotations

import json
import typing

from draf.errors import WorkflowError


def _content_tokens(content: typing.Any) -> int:
    """Rough token estimate for a message payload (~4 chars per token)."""
    if isinstance(content, str):
        return max(1, len(content) // 4)
    try:
        return max(1, len(json.dumps(content)) // 4)
    except TypeError:
        return 1


def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += _content_tokens(content)
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                total += _content_tokens(block.get("text", ""))
    return total


def trim_messages(
    messages: list[dict],
    max_tokens: int | None = None,
    max_messages: int | None = None,
) -> list[dict]:
    """Trim *messages* down to fit context limits.

    The leading ``system`` message (if any) is always preserved; older
    messages are dropped from the front of the conversation until the
    estimated token count and message count fit the limits.

    Args:
        messages: The conversation history.
        max_tokens: Maximum estimated tokens to keep.
        max_messages: Maximum number of messages to keep.

    Returns:
        A new list of messages, trimmed from the front (system kept).
    """
    if not messages:
        return []
    if max_messages is not None and max_messages <= 0:
        return messages
    if max_tokens is not None and max_tokens <= 0:
        return messages

    system: list[dict] = []
    body: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system.append(msg)
        else:
            body.append(msg)

    if max_messages is not None and len(body) > max_messages:
        body = body[-max_messages:]
    if max_tokens is not None and _estimate_tokens(messages) > max_tokens:
        while body and _estimate_tokens(system + body) > max_tokens:
            body.pop(0)
    return system + body


class ContextLimitError(WorkflowError):
    """Raised when a conversation cannot fit the configured context limits."""
