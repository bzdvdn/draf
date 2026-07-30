"""Abstract base for all graph nodes."""

from abc import ABC, abstractmethod


class Node(ABC):
    """Abstract base class for all graph nodes.

    Subclasses must set *type* and implement *execute*.

    Attributes:
        type: Unique node type identifier used for registry lookups.
        config: Arbitrary configuration dict passed at construction.
    """

    type: str = ""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, ctx, state: dict) -> dict:
        """Execute the node's logic.

        Args:
            ctx: Execution context providing tool/LLM access.
            state: Current workflow state dict (shallow-merge in/out).

        Returns:
            State updates to shallow-merge into the workflow state.
        """
