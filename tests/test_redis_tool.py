"""Offline tests for the redis tool (Redis/KeyDB/Valkey, RESP).

``redis`` is not installed in the test environment; a fake module is
installed into ``sys.modules["redis"]`` to exercise the tool's own
argument handling, config parsing and error paths without a server.

The fake keeps shared ``data`` across client instances (like a real
server) and records every created client in ``instances`` so tests can
inspect the connection config and the exact calls issued per ``run``.
"""

import sys

import pytest


class FakeRedis:
    """Minimal stand-in for ``redis.Redis`` (decode_responses=True)."""

    data: dict = {}
    instances: list = []

    def __init__(self, *a, **k):
        self.kwargs = k
        self.url = ""
        self.calls: list[tuple] = []
        self.data = FakeRedis.data
        FakeRedis.instances.append(self)

    @classmethod
    def from_url(cls, url, **kwargs):
        client = cls()
        client.url = url
        client.kwargs = kwargs
        return client

    def _record(self, name, *a):
        self.calls.append((name, a))

    def get(self, key):
        self._record("get", key)
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self._record("set", key, value, ex)
        self.data[key] = value
        return True

    def delete(self, key):
        self._record("delete", key)
        removed = self.data.pop(key, None)
        return 1 if removed is not None else 0

    def exists(self, key):
        self._record("exists", key)
        return 1 if key in self.data else 0

    def scan_iter(self, match="*", count=100):
        self._record("scan_iter", match, count)
        return list(self.data.keys())

    def ttl(self, key):
        self._record("ttl", key)
        return 60

    def expire(self, key, ttl):
        self._record("expire", key, ttl)
        return True

    def incr(self, key):
        self._record("incr", key)
        val = int(self.data.get(key, 0))
        val += 1
        self.data[key] = str(val)
        return val

    def rpush(self, key, value):
        self._record("rpush", key, value)
        items = self.data.setdefault(key, [])
        if not isinstance(items, list):
            items = []
            self.data[key] = items
        items.append(value)
        return len(items)

    def lrange(self, key, start, stop):
        self._record("lrange", key, start, stop)
        items = self.data.get(key, [])
        if not isinstance(items, list):
            return []
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    def sadd(self, key, member):
        self._record("sadd", key, member)
        members = self.data.setdefault(key, set())
        if not isinstance(members, set):
            members = set()
            self.data[key] = members
        members.add(member)
        return len(members)

    def smembers(self, key):
        self._record("smembers", key)
        members = self.data.get(key, set())
        return members if isinstance(members, set) else set()

    def hset(self, key, field, value):
        self._record("hset", key, field, value)
        mapping = self.data.setdefault(key, {})
        if not isinstance(mapping, dict):
            mapping = {}
            self.data[key] = mapping
        mapping[field] = value
        return 1

    def hget(self, key, field):
        self._record("hget", key, field)
        mapping = self.data.get(key, {})
        if not isinstance(mapping, dict):
            return None
        return mapping.get(field)

    def hgetall(self, key):
        self._record("hgetall", key)
        mapping = self.data.get(key, {})
        return mapping if isinstance(mapping, dict) else {}

    def publish(self, channel, message):
        self._record("publish", channel, message)
        return 2

    def ping(self):
        self._record("ping")
        return True

    def close(self):
        self._record("close")


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    import types

    FakeRedis.data.clear()
    FakeRedis.instances.clear()
    module = types.ModuleType("redis")
    module.Redis = FakeRedis
    monkeypatch.setitem(sys.modules, "redis", module)


class TestRedisTool:
    def _tool(self, **cfg):
        from teff.tool.builtin import RedisTool

        return RedisTool(cfg or {"host": "localhost", "port": 6379})

    def _last(self) -> FakeRedis:
        return FakeRedis.instances[-1]

    def test_ping(self):
        assert self._tool().run(action="ping") == "PONG"
        assert any(c[0] == "ping" for c in self._last().calls)

    def test_get_and_set(self):
        tool = self._tool(url="redis://localhost:6379/0")
        assert tool.run(action="set", key="a", value="1") == "set a"
        assert tool.run(action="get", key="a") == "1"
        client = self._last()
        assert client.url == "redis://localhost:6379/0"
        assert client.kwargs.get("decode_responses") is True

    def test_get_not_found(self):
        assert self._tool().run(action="get", key="nope") == "not found"

    def test_set_with_ttl(self):
        tool = self._tool()
        assert tool.run(action="set", key="a", value="1", ttl="30") == "set a"
        assert any(c[0] == "set" and c[1][2] == 30 for c in self._last().calls)

    def test_delete(self):
        tool = self._tool()
        assert tool.run(action="set", key="a", value="1") == "set a"
        assert tool.run(action="delete", key="a") == "deleted a"
        assert tool.run(action="delete", key="a") == "not found"

    def test_exists(self):
        tool = self._tool()
        assert tool.run(action="set", key="a", value="1") == "set a"
        assert tool.run(action="exists", key="a") == "yes"
        assert tool.run(action="exists", key="b") == "no"

    def test_list_keys_with_pattern(self):
        tool = self._tool()
        tool.run(action="set", key="reviewed:x", value="1")
        tool.run(action="set", key="reviewed:y", value="1")
        tool.run(action="set", key="other", value="1")
        result = tool.run(action="list", pattern="reviewed:*")
        assert "reviewed:x" in result and "reviewed:y" in result
        assert "other" not in result

    def test_list_empty(self):
        assert self._tool().run(action="list") == "no keys"

    def test_ttl_and_expire(self):
        tool = self._tool()
        assert tool.run(action="ttl", key="a") == "60"
        assert tool.run(action="expire", key="a", ttl="120") == "expire a 120s"

    def test_incr(self):
        tool = self._tool()
        assert tool.run(action="incr", key="n") == "1"
        assert tool.run(action="incr", key="n") == "2"

    def test_lists(self):
        tool = self._tool()
        tool.run(action="rpush", key="queue", value="a")
        tool.run(action="rpush", key="queue", value="b")
        assert tool.run(action="lrange", key="queue") == "a\nb"
        assert tool.run(action="lrange", key="queue", start="0", stop="0") == "a"

    def test_sets(self):
        tool = self._tool()
        tool.run(action="sadd", key="s", member="a")
        tool.run(action="sadd", key="s", member="b")
        assert tool.run(action="smembers", key="s") == "a\nb"
        assert tool.run(action="smembers", key="missing") == "empty"

    def test_hashes(self):
        tool = self._tool()
        assert tool.run(action="hset", key="h", field="f1", value="v1") == "1"
        assert tool.run(action="hget", key="h", field="f1") == "v1"
        assert tool.run(action="hget", key="h", field="nope") == "not found"
        assert tool.run(action="hgetall", key="h") == "f1=v1"

    def test_publish(self):
        tool = self._tool()
        assert tool.run(action="publish", channel="alerts", message="hi") == (
            "published to alerts (2 subscriber(s))"
        )

    def test_unknown_action(self):
        with pytest.raises(ValueError, match="unknown action: frobnicate"):
            self._tool().run(action="frobnicate")

    def test_action_required(self):
        with pytest.raises(ValueError, match="action is required"):
            self._tool().run(action="")

    def test_key_required(self):
        tool = self._tool()
        with pytest.raises(ValueError, match="key is required"):
            tool.run(action="get")
        with pytest.raises(ValueError, match="key and field are required"):
            tool.run(action="hget", field="f")

    def test_missing_package_message(self, monkeypatch):
        # Pinning redis to None makes the lazy import fail with ImportError
        # whether or not the 'redis' package is installed in the test env.
        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(ImportError, match="teff\\[tools\\]"):
            self._tool().run(action="ping")

    def test_import_error_when_module_is_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(ImportError):
            self._tool().run(action="ping")

    def test_schema_has_required_action(self):
        from teff.harness import tool_to_schema
        from teff.tool.builtin import RedisTool

        schema = tool_to_schema(RedisTool({}))["function"]["parameters"]
        assert "action" in schema["required"]
        assert "key" not in schema["required"]

    def test_registered(self):
        from teff.tool.registry import default_tool_registry

        assert "redis" in default_tool_registry.list()
