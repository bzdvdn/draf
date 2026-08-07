"""Small async helpers shared across the framework."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


async def gather_or_cancel(*coros: Coroutine[Any, Any, T]) -> list[T]:
    """Await *coros* concurrently, cancelling siblings when one fails.

    Unlike plain ``asyncio.gather``, a failing task cancels the remaining
    tasks and waits for them to settle, so no orphaned coroutine keeps
    running (and mutating state / burning provider quota) in the background
    after an error.  Raises the *first* exception — never an
    ``ExceptionGroup`` — so callers that match specific exception types
    (e.g. ``GraphInterrupt``) keep working.
    """
    if not coros:
        return []
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
