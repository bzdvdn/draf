"""Execution context passed to every node."""

import typing
from typing import Any

from draf.tool.tool import Tool

if typing.TYPE_CHECKING:
    from draf.trace import RunTracer


class ExecContext:
    """Context available to nodes during graph execution.

    Provides access to registered tools and a placeholder for
    LLM calls (overridden by the built-in LLM node).

    Attributes:
        state: Current workflow state dict.
        tools: Dict of tool name to Tool instance.
        node_id: Graph node id of the running node.
        node_type: Node type string of the running node.
        tracer: Optional :class:`~draf.trace.RunTracer` collecting
            observability events for the current run.
        reducers: Per-key merge strategies for state updates.
    """

    def __init__(
        self,
        state: dict,
        tools: dict[str, Tool],
        *,
        node_id: str | None = None,
        node_type: str | None = None,
        tracer: "RunTracer | None" = None,
        reducers: dict[str, Any] | None = None,
    ):
        self.state = state
        self.tools = tools
        self.node_id = node_id
        self.node_type = node_type
        self.tracer = tracer
        self.reducers = reducers

    def tool(self, name: str) -> Tool:
        """Look up a tool by name.

        Args:
            name: Tool name registered in the tool registry.

        Returns:
            Tool instance.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self.tools:
            msg = f"unknown tool: {name}"
            raise KeyError(msg)
        return self.tools[name]

    async def llm(self, model: str, messages: list) -> str:
        """Placeholder for LLM calls (not used by built-in LLM node)."""
        raise NotImplementedError("LLM provider not configured")
