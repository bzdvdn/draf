"""Branch case for conditional routing."""

from draf.node.node import Node


class Case:
    """A single branch case used with ``Flow.branch()``."""

    def __init__(self, value: str):
        self.value = value
        self._nodes: list[Node] = []

    def add(self, node: Node) -> "Case":
        """Add a node to this case branch."""
        self._nodes.append(node)
        return self
