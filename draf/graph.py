"""Graph data structure for representing agent workflows."""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from draf.node.node import Node
from draf.node.registry import NodeRegistry, default_registry
from draf.node.context import ExecContext
from draf.tool.tool import Tool
from draf.state import Reducer, State, apply_reducers


_ERROR_CONDITION = "__error__"


@dataclass
class Edge:
    """A directed edge between two nodes with an optional condition.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        condition: Expression ``key=value``, ``key!=value``,
            or comma-separated disjunction ``key=a,b``.
            ``None`` means unconditional.
            ``"__error__"`` matches when the source node raises an exception.
    """

    source_id: str
    target_id: str
    condition: str | None = None


Hook = Callable[[str, Node, dict], Any]
"""Signature for observability hooks: ``(node_id, node, state)``."""


class Graph:
    """A directed graph of nodes connected by edges with conditions.

    The graph executes by walking from the *entry_point* node,
    following edges whose conditions match the current state,
    and shallow-merging each node's output back into the state.

    Error handling::

        Edge("parse", "fallback", "__error__")   # catch exceptions

    Observability hooks::

        await graph.run(state, hooks={
            "on_node_start": callback,
            "on_node_end": callback,
            "on_node_error": callback,
        })

    Hook callbacks receive ``(node_id, node, state)``.
    ``on_node_end`` additionally receives the result dict.
    ``on_node_error`` additionally receives the exception.
    """

    def __init__(self, nodes: dict[str, Node], edges: list[Edge], entry_point: str):
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point

    async def run(
        self,
        state: dict | State,
        tools: list[Tool] | None = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
    ) -> dict | State:
        """Execute the graph starting from the entry point.

        Args:
            state: Initial workflow state (plain ``dict`` or :class:`State`).
            tools: Optional list of Tool instances available to nodes.
            registry: Node registry (defaults to ``default_registry``).
            reducers: Per-key merge strategies
                (see :func:`draf.state.reducers_from_typeddict`).
                Ignored when *state* is a :class:`State` instance.
            hooks: Observability hooks (see class docstring).
            node_timeout: Max seconds per node.  ``asyncio.TimeoutError``
                triggers error edges (``__error__``) like any other exception.
            max_iterations: Max total node executions before raising
                ``RuntimeError``.  Guards against infinite loops in
                cyclic graphs (e.g. agentic loops).  ``None`` means unlimited.

        Raises:
            RuntimeError: If *max_iterations* is exceeded.

        Returns:
            Final state (same type as passed in).
        """
        registry = registry or default_registry
        tool_dict: dict[str, Tool] = {}
        if tools:
            for t in tools:
                tool_dict[t.name] = t

        hooks = hooks or {}

        current_id = self.entry_point
        iteration = 0
        while current_id:
            if max_iterations is not None and iteration >= max_iterations:
                raise RuntimeError(f"graph exceeded max_iterations={max_iterations}")
            iteration += 1

            node = self.nodes[current_id]
            ctx = ExecContext(state, tool_dict)

            _call_hook(hooks, "on_node_start", current_id, node, state)

            try:
                if node_timeout is not None:
                    result = await asyncio.wait_for(
                        node.execute(ctx, state), timeout=node_timeout
                    )
                else:
                    result = await node.execute(ctx, state)
            except Exception as exc:
                _call_hook(hooks, "on_node_error", current_id, node, state, exc)
                error_edge = self._find_error_edge(current_id)
                if error_edge is not None:
                    current_id = error_edge.target_id
                    continue
                raise

            _call_hook(hooks, "on_node_end", current_id, node, state, result)

            if result:
                if isinstance(state, State):
                    state.merge(result)
                else:
                    apply_reducers(state, result, reducers or {})

            outgoing = [
                e
                for e in self.edges
                if e.source_id == current_id and e.condition != _ERROR_CONDITION
            ]
            if not outgoing:
                break

            next_id = self._resolve_edge(outgoing, state)
            if next_id is None:
                break
            current_id = next_id

        return state

    def _find_error_edge(self, node_id: str) -> Edge | None:
        for e in self.edges:
            if e.source_id == node_id and e.condition == _ERROR_CONDITION:
                return e
        return None

    def _resolve_edge(self, edges: list[Edge], state: dict) -> str | None:
        for edge in edges:
            if edge.condition is None:
                return edge.target_id
            if self._evaluate(edge.condition, state):
                return edge.target_id
        return None

    @staticmethod
    def _norm(value: str) -> str:
        """Normalise a condition value for comparison.

        Strips whitespace, lowercases, and removes trailing punctuation
        so LLM output like ``"Positive."`` matches ``positive``.
        """
        return value.strip().lower().rstrip(".,!?;:")

    def _evaluate(self, condition: str, state: dict) -> bool:
        if "!=" in condition:
            key, value = condition.split("!=", 1)
            key = key.strip()
            raw = value.strip()
            state_val = state.get(key)
            if raw == "":
                return state_val is not None and state_val != ""
            if state_val is None:
                return True
            state_str = self._norm(str(state_val))
            if "," in raw:
                values = [self._norm(v) for v in raw.split(",")]
                return state_str not in values
            return state_str != self._norm(raw)
        if "=" in condition:
            parts = condition.split("=", 1)
            key = parts[0].strip()
            raw = parts[1].strip()
            state_val = state.get(key)
            if raw == "":
                return state_val is None or state_val == ""
            if state_val is None:
                return False
            state_str = self._norm(str(state_val))
            if "," in raw:
                values = [self._norm(v) for v in raw.split(",")]
                return state_str in values
            return state_str == self._norm(raw)
        return False

    def to_yaml(self) -> str:
        """Serialize this graph to a YAML string."""
        from draf.yaml import graph_to_yaml

        return graph_to_yaml(self)


def _call_hook(hooks: dict, name: str, *args: Any) -> None:
    fn = hooks.get(name)
    if fn is not None:
        fn(*args)
