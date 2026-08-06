"""Abstract base for all graph nodes."""

import typing
from abc import ABC, abstractmethod

from draf.node.command import Command


class Node(ABC):
    """Abstract base class for all graph nodes.

    Subclasses must set *type* and implement *execute*.

    Attributes:
        type: Unique node type identifier used for registry lookups.
        config: Configuration dict (merged from constructor kwargs).
    """

    type: str = ""

    def __init__(self, config: dict | None = None, **kwargs):
        self.config = {**(config or {}), **kwargs}

    @abstractmethod
    async def execute(self, ctx: typing.Any, state: dict) -> "dict | Command":
        """Execute the node's logic.

        Args:
            ctx: Execution context providing tool/LLM access.
            state: Current workflow state dict (shallow-merge in/out).

        Returns:
            State updates to shallow-merge into the workflow state, or a
            :class:`~draf.node.Command` that additionally routes the graph
            to a specific next node.
        """
