"""WaitForTool — poll until a condition holds or a timeout elapses.

Daemons often need to wait: for a deployment URL to come up, for a CI
pipeline to report, for a key to appear in a cache.  This tool blocks
(synchronously; the async ``arun`` runs it in a thread) until the
condition is met or the timeout is reached.
"""

import time

from draf.tool.tool import Tool


class WaitForTool(Tool):
    """Poll until a condition holds or a timeout elapses.

    Conditions (``condition``):

    - ``url`` — poll ``target`` (a URL) with HTTP GET until it responds;
      ``status`` controls what counts as success (default ``success`` =
      2xx).
    - ``redis_key`` — poll ``target`` (a key) in a Redis-compatible
      store until it exists.

    Args:
        condition: ``url`` or ``redis_key``.
        target: URL or key to poll.
        timeout: Seconds before giving up (default from config, 120.0).
        poll_interval: Seconds between checks (default from config, 1.0).
        status: For ``url``: ``success`` (2xx), ``any``, or an exact
            HTTP status code.

    Args (config): ``poll_interval``, ``timeout``, and connection keys
        for the ``redis_key`` condition (same as the ``redis`` tool).
    """

    name = "wait_for"
    description = (
        "Poll until a condition holds (URL reachable, Redis key exists) "
        "or a timeout elapses"
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.poll_interval = float(cfg.get("poll_interval", 1.0))
        self.timeout = float(cfg.get("timeout", 120.0))
        self.url = cfg.get("url", "")
        self.host = cfg.get("host", "localhost")
        self.port = cfg.get("port", 6379)
        self.db = cfg.get("db", 0)
        self.password = cfg.get("password", "")
        self.username = cfg.get("username", "")

    def _redis(self):
        try:
            import redis
        except ImportError as e:
            msg = "wait_for redis_key requires the 'redis' package (pip install draf[tools])"
            raise ImportError(msg) from e
        if self.url:
            return redis.Redis.from_url(self.url, decode_responses=True)
        kwargs: dict = {}
        if self.username:
            kwargs["username"] = self.username
        if self.password:
            kwargs["password"] = self.password
        return redis.Redis(
            host=self.host,
            port=int(self.port),
            db=int(self.db),
            decode_responses=True,
            **kwargs,
        )

    def run(  # type: ignore[override]
        self,
        condition: str,
        target: str = "",
        timeout: float | None = None,
        poll_interval: float | None = None,
        status: str = "success",
    ) -> str:
        if not condition:
            raise ValueError("condition is required (url, redis_key)")
        if not target:
            raise ValueError("target is required")
        timeout = float(timeout if timeout is not None else self.timeout)
        interval = float(
            poll_interval if poll_interval is not None else self.poll_interval
        )
        start = time.monotonic()
        if condition == "url":
            self._poll_url(target, timeout, interval, status)
        elif condition == "redis_key":
            client = self._redis()
            try:
                self._poll(lambda: bool(client.exists(target)), timeout, interval)
            finally:
                client.close()
        else:
            raise ValueError(f"unknown condition: {condition}")
        return f"condition met after {time.monotonic() - start:.1f}s"

    def _poll(self, check, timeout: float, interval: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if check():
                    return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise ValueError(f"timed out after {timeout:.0f}s")
            time.sleep(interval)

    def _poll_url(
        self, url: str, timeout: float, interval: float, status: str
    ) -> None:
        if status not in ("success", "any") and not str(status).isdigit():
            raise ValueError(f"unknown status expectation: {status}")

        def check() -> bool:
            import httpx

            try:
                response = httpx.get(
                    url, timeout=interval + 2, follow_redirects=True
                )
            except Exception:
                return False
            code = response.status_code
            if status == "any":
                return True
            if status == "success":
                return 200 <= code < 300
            return code == int(status)

        self._poll(check, timeout, interval)


__all__ = ["WaitForTool"]
