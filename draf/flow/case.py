"""Branch case for conditional routing."""

from draf.node.node import Node


class Case:
    """A single branch case used with ``Flow.branch()``."""

    def __init__(self, value: str):
        self.value = value
        self._nodes: list[Node] = []
        self._ids: list[str | None] = []

    def add(self, node: Node, id: str | None = None) -> "Case":
        """Add a node to this case branch.

        *id* optionally names the node in the compiled graph instead of
        the auto-generated ``{type}_{n}``.
        """
        self._nodes.append(node)
        self._ids.append(id)
        return self
