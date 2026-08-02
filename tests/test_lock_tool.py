"""Offline tests for the lock tool (Redis/KeyDB/Valkey).

A fake ``redis`` module (with Lua ``eval`` support) is installed into
``sys.modules``; ``data`` is shared across client instances like a real
server so acquire/release/status can be exercised across tool instances.
"""

import sys

import pytest


class FakeRedis:
    data: dict = {}
    instances: list = []

    def __init__(self, *a, **k):
        self.data = FakeRedis.data
        self.calls: list[tuple] = []
        FakeRedis.instances.append(self)

    @classmethod
    def from_url(cls, url, **kwargs):
        client = cls()
        client.url = url
        client.kwargs = kwargs
        return client

    def set(self, key, value, ex=None, nx=False):
        self.calls.append(("set", key, value, ex, nx))
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    def get(self, key):
        self.calls.append(("get", key))
        return self.data.get(key)

    def ttl(self, key):
        self.calls.append(("ttl", key))
        return 30

    def eval(self, script, numkeys, *args):
        self.calls.append(("eval", script, numkeys, args))
        key = args[0]
        if key not in self.data:
            return 0
        if "del" in script:
            if self.data[key] == args[1]:
                del self.data[key]
                return 1
            return 0
        if "expire" in script:
            if self.data[key] == args[1]:
                return 1
            return 0
        return 0

    def close(self):
        self.calls.append(("close",))


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    import types

    FakeRedis.data.clear()
    FakeRedis.instances.clear()
    module = types.ModuleType("redis")
    module.Redis = FakeRedis
    monkeypatch.setitem(sys.modules, "redis", module)


class TestLockTool:
    def _tool(self, **cfg):
        from draf.tool.builtin import LockTool

        return LockTool(cfg or {})

    def test_acquire(self):
        tool = self._tool()
        assert tool.run(action="acquire", key="job:1") == "acquired"
        assert FakeRedis.data["job:1"] == tool.token

    def test_acquire_conflict(self):
        a = self._tool()
        b = self._tool()
        assert a.run(action="acquire", key="job:1") == "acquired"
        assert b.run(action="acquire", key="job:1") == "held by someone else"

    def test_release_by_owner(self):
        a = self._tool()
        a.run(action="acquire", key="job:1")
        assert a.run(action="release", key="job:1") == "released"
        assert "job:1" not in FakeRedis.data

    def test_release_by_other(self):
        a = self._tool()
        b = self._tool()
        a.run(action="acquire", key="job:1")
        assert b.run(action="release", key="job:1") == (
            "not held (or owned by someone else)"
        )
        assert "job:1" in FakeRedis.data

    def test_renew_by_owner(self):
        a = self._tool()
        a.run(action="acquire", key="job:1")
        assert a.run(action="renew", key="job:1", ttl="60") == "renewed job:1 for 60s"

    def test_renew_by_other(self):
        a = self._tool()
        b = self._tool()
        a.run(action="acquire", key="job:1")
        assert "not held" in b.run(action="renew", key="job:1", ttl="60")

    def test_status_free(self):
        assert "free" in self._tool().run(action="status", key="job:1")

    def test_status_held_by_me(self):
        a = self._tool()
        a.run(action="acquire", key="job:1")
        assert "held by me" in a.run(action="status", key="job:1")

    def test_status_held_by_other(self):
        a = self._tool()
        b = self._tool()
        a.run(action="acquire", key="job:1")
        assert "another holder" in b.run(action="status", key="job:1")

    def test_acquire_zero_ttl_rejected(self):
        with pytest.raises(ValueError, match="ttl must be > 0"):
            self._tool().run(action="acquire", key="job:1", ttl="0")

    def test_action_required(self):
        with pytest.raises(ValueError, match="action is required"):
            self._tool().run(action="")

    def test_key_required(self):
        with pytest.raises(ValueError, match="key is required"):
            self._tool().run(action="acquire")

    def test_unknown_action(self):
        with pytest.raises(ValueError, match="unknown action: unlock"):
            self._tool().run(action="unlock", key="job:1")

    def test_schema_action_required(self):
        from draf.harness import tool_to_schema
        from draf.tool.builtin import LockTool

        schema = tool_to_schema(LockTool({}))["function"]["parameters"]
        assert "action" in schema["required"]

    def test_registered(self):
        from draf.tool.registry import default_tool_registry

        assert "lock" in default_tool_registry.list()
