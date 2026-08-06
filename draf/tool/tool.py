"""Abstract base for all tools."""

import asyncio
import typing
from types import UnionType


def _unwrap_optional(tp: typing.Any) -> typing.Any:
    """Unwrap ``Optional[T]`` / ``T | None`` to ``T`` (``None`` if all-None)."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
        if args:
            return typing.Union[tuple(args)]
    return tp


def coerce_args(tool: "Tool", kwargs: dict) -> dict:
    """Coerce tool-call arguments to match the tool's type hints.

    LLMs often pass values as strings (e.g. ``k="1"`` instead of ``1``);
    coerce them so tools receive properly typed arguments.  Optional hints
    (``float | None``, ``Optional[int]``) are unwrapped before matching.
    """
    method = tool.arun if type(tool).run is Tool.run else tool.run
    try:
        hints = typing.get_type_hints(method)
    except Exception:
        hints = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        tp = _unwrap_optional(hints.get(key))
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
        schema: Optional JSON Schema dict for the tool's arguments.  When
            set (e.g. by :class:`~draf.tool.mcp.McpTool`), it is used as-is
            instead of being inferred from the ``run``/``arun`` signature.
    """

    name: str = ""
    description: str = ""
    schema: dict | None = None

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
