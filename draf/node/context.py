"""Execution context passed to every node."""

from draf.tool.tool import Tool


class ExecContext:
    """Context available to nodes during graph execution.

    Provides access to registered tools and a placeholder for
    LLM calls (overridden by the built-in LLM node).

    Attributes:
        state: Current workflow state dict.
        tools: Dict of tool name to Tool instance.
    """

    def __init__(self, state: dict, tools: dict[str, Tool]):
        self.state = state
        self.tools = tools

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
