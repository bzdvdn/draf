"""Edge condition parsing and evaluation.

Conditions are lightweight string expressions attached to edges.  They
are evaluated against the current workflow state to decide which node
runs next.  Splitting them out from the :class:`~teff.graph.Graph` class
keeps the execution engine free of expression-language details.
"""

from __future__ import annotations

from typing import Callable

from teff.graph.edge import _ERROR_CONDITION, Edge


def _gte(a: float, b: float) -> bool:
    return a >= b


def _lte(a: float, b: float) -> bool:
    return a <= b


def _gt(a: float, b: float) -> bool:
    return a > b


def _lt(a: float, b: float) -> bool:
    return a < b


def _norm(value: str) -> str:
    """Normalise a condition value for comparison.

    Strips whitespace, lowercases, and removes trailing punctuation
    so LLM output like ``"Positive."`` matches ``positive``.
    """
    return value.strip().lower().rstrip(".,!?;:")


def _split_condition(condition: str) -> tuple[str, str, str] | None:
    """Split a condition into ``(op, key, raw)`` with explicit precedence.

    Multi-character operators (``>=``, ``<=``, ``!=``) are tried before
    the single characters they contain (``>``, ``<``, ``=``), so
    ``count>=3`` and ``status!=done`` parse on the compound operator
    instead of accidentally matching ``>`` / ``=``.  ``!=`` must be
    matched before ``=`` for the same reason.
    """
    for op in (">=", "<=", "!=", ">", "<", "="):
        idx = condition.find(op)
        if idx != -1:
            return op, condition[:idx].strip(), condition[idx + len(op) :].strip()
    return None


def evaluate(condition: str | Callable[[dict], bool], state: dict) -> bool:
    """Whether *condition* matches *state*.

    Supports equality/inequality on string keys, comma-separated
    disjunctions (``key=a,b``), numeric comparisons
    (``key>=N`` / ``key<=N`` / ``key>N`` / ``key<N``), and callable
    predicates ``fn(state) -> bool`` (evaluated verbatim).
    """
    if callable(condition):
        return bool(condition(state))
    parts = _split_condition(condition)
    if parts is None:
        return False
    op, key, raw = parts
    state_val = state.get(key)

    if op in (">=", "<=", ">", "<"):
        if state_val is None:
            return False
        try:
            left = float(state_val)
            right = float(raw)
        except (TypeError, ValueError):
            return False
        return {">": _gt, "<": _lt, ">=": _gte, "<=": _lte}[op](left, right)

    if op == "!=":
        if raw == "":
            return state_val is not None and state_val != ""
        if state_val is None:
            return True
        state_str = _norm(str(state_val))
        if "," in raw:
            values = [_norm(v) for v in raw.split(",")]
            return state_str not in values
        return state_str != _norm(raw)

    # Equality ("=").
    if raw == "":
        return state_val is None or state_val == ""
    if state_val is None:
        return False
    state_str = _norm(str(state_val))
    if "," in raw:
        values = [_norm(v) for v in raw.split(",")]
        return state_str in values
    return state_str == _norm(raw)


def find_error_edge(edges: list[Edge], node_id: str) -> Edge | None:
    """Return the ``__error__`` edge leaving *node_id*, if any."""
    for e in edges:
        if e.source_id == node_id and e.condition == _ERROR_CONDITION:
            return e
    return None


def resolve_edge(edges: list[Edge], state: dict) -> str | None:
    """Return the target of the first edge matching *state*, or ``None``."""
    for edge in edges:
        if edge.condition is None:
            return edge.target_id
        if evaluate(edge.condition, state):
            return edge.target_id
    return None


def matched_condition(
    edges: list[Edge], state: dict, target_id: str
) -> "str | Callable[[dict], bool] | None":
    """Return the condition of the first edge matching *state* and *target_id*."""
    for edge in edges:
        if edge.target_id != target_id:
            continue
        if edge.condition is None or evaluate(edge.condition, state):
            return edge.condition
    return None


__all__ = [
    "evaluate",
    "find_error_edge",
    "resolve_edge",
    "matched_condition",
]
