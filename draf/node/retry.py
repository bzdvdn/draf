"""Retry wrapper node with configurable attempts and delay."""

import asyncio

from draf.node.node import Node


class Retry(Node):
    """Wrap a node with retry logic.

    Retries the inner node up to *max_retries* times with an
    optional *delay* (in seconds) between attempts.
    """

    type = "retry"

    def __init__(self, node: Node, max_retries: int = 3, delay: float = 0.0):
        super().__init__({})
        self._node = node
        self._max_retries = max_retries
        self._delay = delay

    async def execute(self, ctx, state):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._node.execute(ctx, state)
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries - 1 and self._delay:
                    await asyncio.sleep(self._delay)
        raise last_exc  # type: ignore[misc]
