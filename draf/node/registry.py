"""Node registry and decorator for registering node types."""

import inspect
from typing import Any, Callable

from draf.errors import ConfigError
from draf.node.command import Command
from draf.node.node import Node

NodeFactory = Callable[..., Node]


class NodeRegistry:
    """Registry mapping node type names to factory functions.

    Used by the YAML loader and pipeline compiler to instantiate
    nodes by their string type identifier.
    """

    def __init__(self) -> None:
        self._factories: dict[str, NodeFactory] = {}

    def register(self, name: str, factory: NodeFactory) -> None:
        """Register a node factory under a type name."""
        self._factories[name] = factory

    def create(self, name: str, config: dict | None = None, **kwargs: Any) -> Node:
        """Create a node instance by type name.

        Args:
            name: Registered node type name.
            config: Optional configuration dict (backward-compatible).
            **kwargs: Additional keyword arguments merged into config.

        Returns:
            A Node instance.

        Raises:
            ConfigError: If the type name is not registered
                (also a ``KeyError``).
        """
        if name not in self._factories:
            msg = f"unknown node type: {name}"
            raise ConfigError(msg)
        merged = {**(config or {}), **kwargs}
        return self._factories[name](merged)

    def list(self) -> list[str]:
        """Return all registered node type names."""
        return list(self._factories.keys())

    def copy(self) -> "NodeRegistry":
        """Return a shallow copy with the same factory registrations."""
        reg = NodeRegistry()
        reg._factories = dict(self._factories)
        return reg


default_registry = NodeRegistry()


def make_function_node(fn: Callable, type_name: str | None = None) -> Node:
    """Wrap an async (or sync) function into a :class:`Node` instance.

    The function is called as ``fn(ctx, state)`` and must return a dict of
    state updates or a :class:`~draf.node.command.Command`.  Sync functions
    are supported.  *type_name* sets the node's ``type`` (defaults to the
    function's ``__name__``).

    This is the building block behind ``Flow.step(fn)``; it does **not**
    register the node type in the registry.
    """
    if not callable(fn):
        raise TypeError("make_function_node requires a callable")
    name = type_name or getattr(fn, "__name__", "function")

    class _FunctionNode(Node):
        type = str(name)

        async def execute(self, ctx, state: dict) -> dict | Command:
            result = fn(ctx, state)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return {}
            if not isinstance(result, (dict, Command)):
                raise TypeError(
                    f"function node {name!r} must return a dict or Command, "
                    f"got {type(result).__name__}"
                )
            return result

    return _FunctionNode()


def node(node_name: str, config: type[Any] | None = None):
    """Decorator that registers an async function as a node type.

    The decorated function receives ``(ctx, state)`` or ``(ctx, config, state)``
    when a typed *config* dataclass is provided.

    Args:
        node_name: Type name to register under.
        config: Optional dataclass type for typed config parsing.
    """

    def decorator(fn: Callable) -> Callable:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("node function must be async")

        if config is not None:
            config_cls: type[Any] = config

            def factory(cfg: dict) -> Node:
                class DecoratedNode(Node):
                    type = node_name

                    async def execute(self, ctx, state: dict) -> dict:
                        parsed = config_cls(**cfg)
                        return await fn(ctx, parsed, state)

                return DecoratedNode(cfg)

        else:

            def factory(cfg: dict) -> Node:
                class DecoratedNode(Node):
                    type = node_name

                    async def execute(self, ctx, state: dict) -> dict:
                        return await fn(ctx, state)

                return DecoratedNode(cfg)

        factory.__name__ = fn.__name__
        factory.__qualname__ = fn.__qualname__
        default_registry.register(node_name, factory)
        return fn

    return decorator
