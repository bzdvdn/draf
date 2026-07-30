"""Node registry and decorator for registering node types."""

import inspect
from typing import Any, Callable

from draf.node.node import Node

NodeFactory = Callable[[dict], Node]


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

    def create(self, name: str, config: dict | None = None) -> Node:
        """Create a node instance by type name.

        Args:
            name: Registered node type name.
            config: Configuration dict passed to the node factory.

        Returns:
            A Node instance.

        Raises:
            KeyError: If the type name is not registered.
        """
        if name not in self._factories:
            msg = f"unknown node type: {name}"
            raise KeyError(msg)
        return self._factories[name](config or {})

    def list(self) -> list[str]:
        """Return all registered node type names."""
        return list(self._factories.keys())

    def copy(self) -> "NodeRegistry":
        """Return a shallow copy with the same factory registrations."""
        reg = NodeRegistry()
        reg._factories = dict(self._factories)
        return reg


default_registry = NodeRegistry()


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
