"""Declarative hook wiring for the ``hooks:`` workflow block.

Hooks are Python callables, so a YAML ``hooks:`` block cannot define them
inline. Instead it *references* named hooks from a shared registry that
plugins and scripts populate with :func:`register` (or the :func:`hook`
decorator):

    from teff import hooks

    @hooks.hook("telemetry")
    def telemetry(node_id, node, state, **kw):
        metrics.counter("graph.node", node_id=node_id)

    hooks:
      on_node_start: telemetry
      on_node_error: [telemetry, on_error]

The resolved mapping is passed to ``graph.run(hooks=...)``. Hook callables
may be sync or async. When several hooks share one event, they run in
registration order.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from teff.errors import ConfigError

#: The three hook kinds understood by ``graph.run(hooks=...)``.
KINDS = ("on_node_start", "on_node_end", "on_node_error")

#: name -> callable
_HOOK_REGISTRY: dict[str, Callable] = {}


def register(name: str, fn: Callable) -> Callable:
    """Register *fn* under *name* (last registration wins)."""
    if not callable(fn):
        raise ConfigError(f"hook {name!r} must be callable")
    _HOOK_REGISTRY[name] = fn
    return fn


def hook(name: str) -> Callable:
    """Decorator form of :func:`register`."""

    def deco(fn: Callable) -> Callable:
        return register(name, fn)

    return deco


def _compose(names: list[str]) -> Callable:
    fns = []
    for nm in names:
        fn = _HOOK_REGISTRY.get(nm)
        if fn is None:
            raise ConfigError(
                f"hooks: unknown hook {nm!r} (register it with `@hooks.hook(name)` "
                "in a plugin)"
            )
        fns.append(fn)

    if len(fns) == 1:
        return fns[0]

    if any(inspect.iscoroutinefunction(f) for f in fns):

        async def adispatch(*args: Any, **kwargs: Any) -> None:
            for fn in fns:
                res = fn(*args, **kwargs)
                if inspect.isawaitable(res):
                    await res

        return adispatch

    def dispatch(*args: Any, **kwargs: Any) -> None:
        for fn in fns:
            fn(*args, **kwargs)

    return dispatch


def _names_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(
                    f"hooks entries must be hook-name strings, got {type(item).__name__}"
                )
            out.append(item)
        return out
    raise ConfigError(
        f"hooks value must be a hook name or a list of names, "
        f"got {type(value).__name__}"
    )


def resolve_hooks(block: dict | None) -> dict | None:
    """Resolve a ``hooks:`` YAML block into a ``graph.run(hooks=...)`` dict.

    Each event key (``on_node_start``/``on_node_end``/``on_node_error``)
    is a hook name or a list of names.  An unset or ``null`` kind is
    skipped.  Returns ``None`` for an empty block.
    """
    if not isinstance(block, dict) or not block:
        return None
    out: dict[str, Callable] = {}
    for kind in KINDS:
        value = block.get(kind)
        if value is None:
            continue
        out[kind] = _compose(_names_from(value))
    return out or None
