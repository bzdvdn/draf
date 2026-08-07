"""Redis-like store tools — Redis, KeyDB, Valkey (any RESP server).

A single tool with an ``action`` selector covering the common operations
(get/set/delete/list/exists/ttl/expire/incr, plus lists, sets, hashes and
pub/sub) against any RESP-compatible server.  It uses the ``redis``
package, which speaks RESP to Redis, KeyDB and Valkey alike, so the same
config works for all of them.  The distributed ``lock`` tool reuses the
shared client setup in :class:`_RedisBase`.
"""

import fnmatch
from typing import Any

from teff.tool.tool import Tool


class _RedisBase(Tool):
    """Shared RESP client setup for Redis-like tools.

    Args (config):
        url: Optional ``redis://[[username]:[password]]@host:port/db``
            connection string (e.g. ``redis://localhost:6379/0``).
        host: Host (default ``localhost``) when ``url`` is not given.
        port: Port (default ``6379``).
        db: Database index (default ``0``).
        password: Optional password.
        username: Optional username (ACL setups).
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.url = cfg.get("url", "")
        self.host = cfg.get("host", "localhost")
        self.port = cfg.get("port", 6379)
        self.db = cfg.get("db", 0)
        self.password = cfg.get("password", "")
        self.username = cfg.get("username", "")

    def _client(self) -> Any:
        try:
            import redis as _redis
        except ImportError as e:
            msg = "redis tools require the 'redis' package (pip install teff[tools])"
            raise ImportError(msg) from e
        if self.url:
            return _redis.Redis.from_url(self.url, decode_responses=True)
        kwargs: dict = {}
        if self.username:
            kwargs["username"] = self.username
        if self.password:
            kwargs["password"] = self.password
        return _redis.Redis(
            host=self.host,
            port=int(self.port),
            db=int(self.db),
            decode_responses=True,
            **kwargs,
        )


class RedisTool(_RedisBase):
    """Read/write a Redis-compatible store (KeyDB/Valkey supported).

    Args:
        action: ``ping`` | ``get`` | ``set`` | ``delete`` | ``exists``
            | ``list`` | ``ttl`` | ``expire`` | ``incr`` | ``rpush``
            | ``lrange`` | ``sadd`` | ``smembers`` | ``hset`` | ``hget``
            | ``hgetall`` | ``publish``.
        key: Key to operate on (not needed for ``ping``/``list``).
        value: Value for ``set``/``rpush``/``hset``.
        ttl: Expiry in seconds for ``set`` (``ex``) or ``expire``.
        pattern: Glob pattern for ``list`` (default ``*``).
        field: Hash field for ``hset``/``hget``.
        member: Set member for ``sadd``.
        channel: Channel for ``publish``.
        message: Message for ``publish``.
        start, stop: Range for ``lrange`` (default all).
    """

    name = "redis"
    description = (
        "Read/write a Redis-compatible key-value store (get, set, delete, "
        "list, exists, ttl, expire, incr, lists, sets, hashes, publish)"
    )

    def run(  # type: ignore[override]
        self,
        action: str,
        key: str = "",
        value: str = "",
        ttl: int = -1,
        pattern: str = "*",
        field: str = "",
        member: str = "",
        channel: str = "",
        message: str = "",
        start: int = 0,
        stop: int = -1,
    ) -> str:
        if not action:
            raise ValueError("action is required (get, set, delete, list, ...)")
        client = self._client()
        try:
            a = action.lower()
            if a == "ping":
                return "PONG" if client.ping() else "no response"
            if a == "get":
                if not key:
                    raise ValueError("key is required")
                val = client.get(key)
                return "not found" if val is None else str(val)
            if a == "set":
                if not key:
                    raise ValueError("key is required")
                if int(ttl) > 0:
                    client.set(key, value, ex=int(ttl))
                else:
                    client.set(key, value)
                return f"set {key}"
            if a == "delete":
                if not key:
                    raise ValueError("key is required")
                return f"deleted {key}" if client.delete(key) else "not found"
            if a == "exists":
                if not key:
                    raise ValueError("key is required")
                return "yes" if client.exists(key) else "no"
            if a == "list":
                keys = [
                    k
                    for k in client.scan_iter(match=pattern, count=100)
                    if fnmatch.fnmatchcase(k, pattern)
                ]
                return "\n".join(sorted(keys)) if keys else "no keys"
            if a == "ttl":
                if not key:
                    raise ValueError("key is required")
                return str(client.ttl(key))
            if a == "expire":
                if not key:
                    raise ValueError("key is required")
                client.expire(key, int(ttl))
                return f"expire {key} {ttl}s"
            if a == "incr":
                if not key:
                    raise ValueError("key is required")
                return str(client.incr(key))
            if a == "rpush":
                if not key:
                    raise ValueError("key is required")
                return str(client.rpush(key, value))
            if a == "lrange":
                if not key:
                    raise ValueError("key is required")
                items = client.lrange(key, int(start), int(stop))
                return "\n".join(str(i) for i in items) if items else "empty"
            if a == "sadd":
                if not key:
                    raise ValueError("key is required")
                return str(client.sadd(key, member))
            if a == "smembers":
                if not key:
                    raise ValueError("key is required")
                members = client.smembers(key)
                return (
                    "\n".join(sorted(str(m) for m in members)) if members else "empty"
                )
            if a == "hset":
                if not key or not field:
                    raise ValueError("key and field are required")
                return str(client.hset(key, field, value))
            if a == "hget":
                if not key or not field:
                    raise ValueError("key and field are required")
                val = client.hget(key, field)
                return "not found" if val is None else str(val)
            if a == "hgetall":
                if not key:
                    raise ValueError("key is required")
                data = client.hgetall(key)
                return (
                    "\n".join(f"{k}={v}" for k, v in data.items()) if data else "empty"
                )
            if a == "publish":
                if not channel:
                    raise ValueError("channel is required")
                subs = client.publish(channel, message)
                return f"published to {channel} ({subs} subscriber(s))"
            raise ValueError(f"unknown action: {action}")
        finally:
            client.close()


__all__ = ["RedisTool", "_RedisBase"]
