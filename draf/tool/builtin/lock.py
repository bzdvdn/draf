"""Distributed lock over a Redis-compatible store (Redis/KeyDB/Valkey).

A single ``lock`` tool with an ``action`` selector: ``acquire``,
``release``, ``renew``, ``status``.  A unique token is generated per tool
instance, so only the instance that acquired a lock can release or renew
it — release/renew are compare-and-{del,expire} Lua scripts, atomic on
the server, so a stale instance can never clobber a lock it does not own.

Useful in daemon workflows where several processes (or several ticks)
must not review the same pull request at once.
"""

import uuid

from draf.tool.builtin.redis import _RedisBase

_DEL_IF_MATCH = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('del', KEYS[1]) else return 0 end"
)
_EXPIRE_IF_MATCH = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
)


class LockTool(_RedisBase):
    """Distributed lock over Redis (KeyDB/Valkey supported).

    Args:
        action: ``acquire`` | ``release`` | ``renew`` | ``status``.
        key: Lock name.
        ttl: Lease length in seconds for ``acquire`` (default 30) or the
            new lease for ``renew``.

    Args (config): same as the ``redis`` tool — ``url`` or
        ``host``/``port``/``db``/``password``/``username``.
    """

    name = "lock"
    description = (
        "Distributed lock over Redis-compatible stores (acquire, release, "
        "renew, status)"
    )

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.token = uuid.uuid4().hex

    def run(  # type: ignore[override]
        self, action: str, key: str = "", ttl: int = 30
    ) -> str:
        if not action:
            raise ValueError("action is required (acquire, release, renew, status)")
        if not key:
            raise ValueError("key is required")
        client = self._client()
        try:
            a = action.lower()
            if a == "acquire":
                if int(ttl) <= 0:
                    raise ValueError("ttl must be > 0")
                ok = client.set(key, self.token, nx=True, ex=int(ttl))
                return "acquired" if ok else "held by someone else"
            if a == "release":
                released = client.eval(_DEL_IF_MATCH, 1, key, self.token)
                return "released" if released else "not held (or owned by someone else)"
            if a == "renew":
                renewed = client.eval(_EXPIRE_IF_MATCH, 1, key, self.token, int(ttl))
                return (
                    f"renewed {key} for {ttl}s"
                    if renewed
                    else "not held (or owned by someone else)"
                )
            if a == "status":
                holder = client.get(key)
                if holder is None:
                    return f"{key} is free"
                remaining = client.ttl(key)
                who = (
                    "me"
                    if holder == self.token
                    else f"another holder ({holder[:8]}...)"
                )
                return f"{key} held by {who} ({remaining}s left)"
            raise ValueError(f"unknown action: {a}")
        finally:
            client.close()


__all__ = ["LockTool"]
