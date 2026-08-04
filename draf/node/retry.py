"""Retry wrapper node with configurable attempts, backoff, and timeout."""

import asyncio

from draf.node.context import ExecContext
from draf.node.interrupt import GraphInterrupt
from draf.node.node import Node


def _match_exception(exc: Exception, retry_on: list[str]) -> bool:
    """Whether *exc* matches one of the *retry_on* selectors.

    Selectors may be exception type names (``"TimeoutError"``,
    ``"httpx.HTTPStatusError"``) or, for exceptions that carry a status
    code (``httpx`` transport errors), a numeric HTTP status code.
    An empty selector list means *everything* is retried.
    """
    if not retry_on:
        return True
    for selector in retry_on:
        selector = str(selector)
        if selector.isdigit():
            code = getattr(exc, "response", None)
            if code is not None and getattr(code, "status_code", None) == int(selector):
                return True
            continue
        name = type(exc).__name__
        if selector == name:
            return True
        if "." in selector:
            mod = selector.rsplit(".", 1)[0]
            if type(exc).__module__ == mod and selector.rsplit(".", 1)[1] == name:
                return True
    return False


class Retry(Node):
    """Wrap a node with retry logic.

    Retries the inner node up to *max_retries* attempts total.  Between
    attempts it waits ``delay`` seconds (scaled by *backoff* per retry,
    e.g. ``backoff=2.0`` gives delay, 2×, 4×, …).  Each attempt is bounded
    by *timeout* seconds when set.  *retry_on* restricts which failures are
    retried (exception type names or HTTP status codes); by default any
    exception is retried.

    Config (all optional): ``max_retries`` (default 3), ``delay`` (default
    0.0), ``backoff`` (default 1.0), ``timeout`` (default None),
    ``retry_on`` (default [] = all).
    """

    type = "retry"

    def __init__(
        self,
        node: Node,
        max_retries: int = 3,
        delay: float = 0.0,
        backoff: float = 1.0,
        timeout: float | None = None,
        retry_on: list | None = None,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self._node = node
        self._max_retries = max(1, int(max_retries))
        self._delay = float(delay)
        self._backoff = float(backoff)
        self._timeout = float(timeout) if timeout else None
        self._retry_on = list(retry_on or [])

    async def execute(self, ctx: ExecContext, state: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                coro = self._node.execute(ctx, state)
                if self._timeout:
                    coro = asyncio.wait_for(coro, timeout=self._timeout)
                return await coro
            except GraphInterrupt:
                raise
            except Exception as e:
                last_exc = e
                if attempt >= self._max_retries - 1:
                    break
                if not _match_exception(e, self._retry_on):
                    break
                tracer = getattr(ctx, "tracer", None)
                if tracer is not None:
                    tracer.retry(ctx.node_id, ctx.node_type, attempt + 1, e)
                wait = self._delay * (self._backoff**attempt)
                if wait:
                    await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]


def wrap_with_retry(node: Node, config: dict | None) -> Node:
    """Wrap *node* in a :class:`Retry` when *config* enables it.

    Accepts the YAML ``retry:`` block (``max_retries``, ``delay``,
    ``backoff``, ``timeout``, ``retry_on``).  Returns the node unchanged
    when *config* is empty or the block is explicitly disabled.
    """
    if not config:
        return node
    cfg = dict(config)
    if cfg.pop("enabled", True) is False:
        return node
    return Retry(node, **cfg)
