"""Context injection helpers for long-term memory.

These turn recalled memories into a block of text that can be inserted
into an agent's system prompt (or a LLM call's messages) so a model sees
relevant cross-session facts without needing the ``memory`` tool.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from teff.memory.base import MemoryStore

DEFAULT_HEADER = "Relevant memories:"

#: ``${NAME}`` references in a ``namespace`` are resolved per run.
_ENV_VAR = re.compile(r"\$\{([^}]*)\}")

#: Dynamic run-scoped variables that a ``namespace`` may reference.
_RUN_VARS = ("owner", "session_id", "checkpoint_id")


@dataclass
class MemoryConfig:
    """Declarative memory injection for ``agent`` / ``llm`` nodes.

    Passed to :class:`~teff.node.agent.ReActAgent`,
    :class:`~teff.node.llm.LLM` (and the ``flow.react()`` /
    ``flow.harness()`` / ``flow.llm()`` helpers) via the ``memory``
    parameter.  A plain config dict is accepted too — that is what YAML
    workflows deserialize to.

    Attributes:
        store: A ready :class:`~teff.memory.base.MemoryStore`, or a store
            config dict (``{"type": "sqlite", ...}``) to build with the
            node's provider registry.
        namespace: Namespace subtree to recall from (a string becomes a
            single-segment namespace).  Each segment may reference
            ``${owner}``, ``${session_id}`` or ``${checkpoint_id}``, which
            are resolved from the enclosing run — so a shared graph serves
            per-user memory via ``["users", "${owner}"]``.
        k: Maximum number of memories recalled per turn.
        header: First line of the injected block.
    """

    store: MemoryStore | dict | None = None
    namespace: tuple[str, ...] | list[str] | str | None = None
    k: int = 5
    header: str = DEFAULT_HEADER

    def to_dict(self) -> dict:
        """Config-dict form (used internally; YAML round-trips as this)."""
        return {
            "store": self.store,
            "namespace": self.namespace,
            "k": self.k,
            "header": self.header,
        }


async def memory_context_from_config(cfg: dict, *, state: dict, ctx: Any) -> str:
    """Recall block for a node's ``memory`` config, or ``""`` when off.

    Shared by :class:`~teff.node.agent.ReActAgent` and
    :class:`~teff.node.llm.LLM`.  Reads the node's ``memory`` config
    (``{store, namespace, k, header}``), resolves the store — a
    :class:`~teff.memory.base.MemoryStore` instance, or a config dict
    built via ``memory_from_config`` using *ctx*'s provider registry —
    and recalls memories for the most recent user message.  Namespace
    segments may reference ``${owner}`` / ``${session_id}`` /
    ``${checkpoint_id}``, resolved from *ctx* — the building block for
    per-user memory behind a shared multi-tenant graph.

    The returned block is meant to be prepended to the LLM messages as a
    ``system`` message; it is empty when memory is unconfigured or
    nothing matched.
    """
    memory_cfg = cfg.get("memory")
    if not memory_cfg:
        return ""
    if isinstance(memory_cfg, MemoryConfig):
        memory_cfg = memory_cfg.to_dict()
    mem_store = memory_cfg.get("store")
    if isinstance(mem_store, dict):
        from teff.memory.tool import memory_from_config

        mem_store = memory_from_config(
            memory_cfg,
            providers=getattr(ctx, "providers", None),
            default_provider=getattr(ctx, "default_provider", None),
        )
    if mem_store is None:
        return ""
    ns_raw = memory_cfg.get("namespace")
    namespace = _resolve_namespace(ns_raw, ctx)
    messages = list(state.get(cfg.get("messages_key", "messages"), []) or [])
    fallback = str(state.get(cfg.get("input_key", "input"), ""))
    return await memory_context(
        mem_store,
        last_user_text(messages, fallback=fallback),
        namespace=namespace,
        k=int(memory_cfg.get("k", 5)),
        header=str(memory_cfg.get("header", DEFAULT_HEADER)),
    )


async def memory_context(
    store: Any,
    query: str,
    *,
    namespace: tuple[str, ...] = (),
    k: int = 5,
    header: str = DEFAULT_HEADER,
    bullet: str = "-",
) -> str:
    """Return a formatted block of recalled memories, or ``""`` if none.

    Args:
        store: A :class:`~teff.memory.base.MemoryStore`.
        query: Natural-language query used for the semantic recall.
        namespace: Namespace subtree to recall from.
        k: Maximum number of memories to include.
        header: First line of the block.
        bullet: Per-item bullet prefix.

    The returned string is meant to be appended to a system prompt; it is
    empty when nothing matched, so callers can skip it entirely.
    """
    if not query or not str(query).strip():
        return ""
    items = await store.search(namespace, query=str(query), k=k)
    lines = [
        f"{bullet} {item.value.get('text', '')}"
        for item in items
        if item.value.get("text")
    ]
    if not lines:
        return ""
    return f"{header}\n" + "\n".join(lines)


def last_user_text(messages: list[dict], fallback: str = "") -> str:
    """Return the most recent non-empty user message text."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content", "")).strip()
        if text:
            return text
    return fallback


def _resolve_namespace(
    ns_raw: tuple[str, ...] | list[str] | str | None, ctx: Any
) -> tuple[str, ...]:
    """Resolve per-run ``${owner}`` / ``${session_id}`` / ``${checkpoint_id}``
    references in a ``namespace`` from the node's run context, falling back
    to the process environment (and then leaving the reference untouched).
    """
    if isinstance(ns_raw, str):
        ns_raw = (ns_raw,)
    if not ns_raw:
        return ()
    return tuple(_interpolate_part(str(part), ctx) for part in ns_raw)


def _interpolate_part(part: str, ctx: Any) -> str:
    return _ENV_VAR.sub(lambda m: _replace_var(m, ctx), part)


def _replace_var(m: re.Match, ctx: Any) -> str:
    name = m.group(1)
    if name in _RUN_VARS and ctx is not None:
        value = getattr(ctx, name, None)
        if value:
            return str(value)
    return os.environ.get(name, m.group(0))
