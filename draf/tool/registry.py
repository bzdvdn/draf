"""Tool registry and decorator."""

import inspect

from draf.tool.tool import Tool


class ToolRegistry:
    """Registry mapping tool names to tool classes."""

    def __init__(self) -> None:
        self._tools: dict[str, type[Tool]] = {}

    def register(self, tool_cls: type[Tool]) -> None:
        """Register a tool class.

        Raises:
            ValueError: If the class has no non-empty *name* attribute.
        """
        if not hasattr(tool_cls, "name") or not tool_cls.name:
            raise ValueError(f"Tool class {tool_cls.__name__} must have a non-empty 'name' attribute")
        self._tools[tool_cls.name] = tool_cls

    def create(self, name: str) -> Tool:
        """Instantiate a tool by name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._tools:
            msg = f"unknown tool: {name}"
            raise KeyError(msg)
        return self._tools[name]()

    def list(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())


default_tool_registry = ToolRegistry()


def tool(tool_name: str, description: str | None = None):
    """Decorator that registers a function as a tool.

    The decorated function can be sync or async.  An async function
    gets both *run* (with ``asyncio.run``) and *arun*; a sync function
    gets only *run*.

    Args:
        tool_name: Unique tool name.
        description: Optional description (falls back to docstring).
    """

    def decorator(fn):
        tool_desc = description or (fn.__doc__ or "").strip() or ""
        is_async = inspect.iscoroutinefunction(fn)

        if is_async:

            class DecoratedTool(Tool):
                name = tool_name
                description = tool_desc

                async def arun(self, **kwargs):
                    return await fn(**kwargs)

                def run(self, **kwargs):
                    import asyncio
                    return asyncio.run(fn(**kwargs))

        else:

            class DecoratedTool(Tool):
                name = tool_name
                description = tool_desc

                def run(self, **kwargs):
                    return fn(**kwargs)

        DecoratedTool.__name__ = fn.__name__
        DecoratedTool.__qualname__ = fn.__qualname__
        default_tool_registry.register(DecoratedTool)
        return fn

    return decorator
