"""Abstract base for all tools."""

import asyncio


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
