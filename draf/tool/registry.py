"""Tool registry and decorator."""

import inspect

from draf.tool.tool import Tool


def _accepts_config_dict(cls: type[Tool]) -> bool:
    """Return True if *cls*'s constructor takes a ``config`` dict positionally."""
    try:
        params = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return False
    if not params:
        return False
    first = params[0]
    return first.name == "config" and first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


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
            raise ValueError(
                f"Tool class {tool_cls.__name__} must have a non-empty 'name' attribute"
            )
        self._tools[tool_cls.name] = tool_cls

    def create(self, name: str, config: dict | None = None) -> Tool:
        """Instantiate a tool by name.

        If *config* is provided, it is passed to the tool's constructor as a
        dict when the constructor accepts one (a leading ``config``
        parameter); otherwise the config keys are passed as keyword
        arguments, and if the constructor rejects them the values are
        assigned as attributes on an argument-less instance.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._tools:
            msg = f"unknown tool: {name}"
            raise KeyError(msg)
        cls = self._tools[name]
        if config is None:
            return cls()
        if _accepts_config_dict(cls):
            return cls(config)  # type: ignore[call-arg]
        try:
            return cls(**config)
        except TypeError:
            tool = cls()
            for k, v in config.items():
                setattr(tool, k, v)
            return tool

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

            class AsyncTool(Tool):
                name = tool_name
                description = tool_desc

                async def arun(self, **kwargs):
                    return await fn(**kwargs)

                def run(self, **kwargs):
                    import asyncio

                    return asyncio.run(fn(**kwargs))

            tool_cls = AsyncTool
        else:

            class SyncTool(Tool):
                name = tool_name
                description = tool_desc

                def run(self, **kwargs):
                    return fn(**kwargs)

            tool_cls = SyncTool

        tool_cls.__name__ = fn.__name__
        tool_cls.__qualname__ = fn.__qualname__
        default_tool_registry.register(tool_cls)
        return fn

    return decorator
