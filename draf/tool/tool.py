"""Abstract base for all tools."""

import asyncio
import typing


def coerce_args(tool: "Tool", kwargs: dict) -> dict:
    """Coerce tool-call arguments to match the tool's type hints.

    LLMs often pass values as strings (e.g. ``k="1"`` instead of ``1``);
    coerce them so tools receive properly typed arguments.
    """
    method = tool.arun if type(tool).run is Tool.run else tool.run
    try:
        hints = typing.get_type_hints(method)
    except Exception:
        hints = {}
    for key, value in kwargs.items():
        tp = hints.get(key)
        if tp is int and not isinstance(value, int):
            kwargs[key] = int(value)
        elif tp is float and not isinstance(value, float):
            kwargs[key] = float(value)
        elif tp is bool and not isinstance(value, bool):
            kwargs[key] = str(value).lower() in ("true", "1")
        elif tp is str and not isinstance(value, str):
            kwargs[key] = str(value)
    return kwargs


class Tool:
    """Abstract base class for tools callable by nodes.

    Subclasses define *name* and *description* as class attributes,
    then implement *run* (sync) and/or *arun* (async).

    Attributes:
        name: Unique tool name (defaults to lowercase class name).
        description: Human-readable description for LLM tool selection.
    """

    name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name == "":
            cls.name = cls.__name__.lower()

    def __init__(self):
        pass

    def run(self, **kwargs):
        """Execute the tool synchronously.

        Args:
            **kwargs: Tool-specific keyword arguments.

        Returns:
            Tool-specific result (typically a string).
        """
        raise NotImplementedError

    async def arun(self, **kwargs):
        """Execute the tool asynchronously.

        Falls back to *run* via ``asyncio.to_thread`` if not overridden.

        Args:
            **kwargs: Tool-specific keyword arguments.

        Returns:
            Tool-specific result (typically a string).
        """
        return await asyncio.to_thread(self.run, **kwargs)
