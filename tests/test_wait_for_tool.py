"""Offline tests for the wait_for tool.

httpx (for the ``url`` condition) and the ``redis`` module (for the
``redis_key`` condition) are faked; poll intervals are kept tiny so the
timeout paths stay fast.
"""

import sys

import pytest


class TestWaitForTool:
    def _tool(self, **cfg):
        from draf.tool.builtin import WaitForTool

        return WaitForTool(cfg or {})

    def _fake_url(self, monkeypatch, statuses):
        import httpx

        state = {"i": 0}

        def fake_get(*a, **k):
            code = statuses[min(state["i"], len(statuses) - 1)]
            state["i"] += 1

            class FakeResponse:
                status_code = code

            return FakeResponse()

        monkeypatch.setattr(httpx, "get", fake_get)

    def test_url_immediate_success(self, monkeypatch):
        self._fake_url(monkeypatch, [200])
        result = self._tool(poll_interval=0.01).run(
            condition="url", target="https://example.com/ready", timeout=5
        )
        assert "condition met" in result

    def test_url_eventually_up(self, monkeypatch):
        self._fake_url(monkeypatch, [404, 404, 200])
        result = self._tool(poll_interval=0.01).run(
            condition="url", target="https://example.com", timeout=5
        )
        assert "condition met" in result

    def test_url_timeout(self, monkeypatch):
        self._fake_url(monkeypatch, [500])
        with pytest.raises(ValueError, match="timed out"):
            self._tool(poll_interval=0.01).run(
                condition="url", target="https://example.com", timeout=0.1
            )

    def test_url_exact_status(self, monkeypatch):
        self._fake_url(monkeypatch, [503])
        result = self._tool(poll_interval=0.01).run(
            condition="url",
            target="https://example.com",
            status="503",
            timeout=5,
        )
        assert "condition met" in result

    def test_redis_key(self, monkeypatch):
        import types

        class FakeRedis:
            def __init__(self, *a, **k):
                self.count = 0

            def exists(self, key):
                self.count += 1
                return self.count >= 2

            def close(self):
                pass

        module = types.ModuleType("redis")
        module.Redis = FakeRedis
        monkeypatch.setitem(sys.modules, "redis", module)

        result = self._tool(poll_interval=0.01).run(
            condition="redis_key", target="task:done", timeout=5
        )
        assert "condition met" in result

    def test_redis_key_timeout(self, monkeypatch):
        import types

        class FakeRedis:
            def __init__(self, *a, **k):
                pass

            def exists(self, key):
                return 0

            def close(self):
                pass

        module = types.ModuleType("redis")
        module.Redis = FakeRedis
        monkeypatch.setitem(sys.modules, "redis", module)

        with pytest.raises(ValueError, match="timed out"):
            self._tool(poll_interval=0.01).run(
                condition="redis_key", target="task:done", timeout=0.1
            )

    def test_condition_required(self):
        with pytest.raises(ValueError, match="condition is required"):
            self._tool().run(condition="")

    def test_target_required(self):
        with pytest.raises(ValueError, match="target is required"):
            self._tool().run(condition="url")

    @pytest.mark.asyncio
    async def test_arun_condition_required(self):
        with pytest.raises(ValueError, match="condition is required"):
            await self._tool().arun(condition="")

    @pytest.mark.asyncio
    async def test_arun_target_required(self):
        with pytest.raises(ValueError, match="target is required"):
            await self._tool().arun(condition="url")

    @pytest.mark.asyncio
    async def test_arun_redis_key(self, monkeypatch):
        import types

        class FakeRedis:
            def __init__(self, *a, **k):
                self.count = 0

            def exists(self, key):
                self.count += 1
                return self.count >= 2

            def close(self):
                pass

        module = types.ModuleType("redis")
        module.Redis = FakeRedis
        monkeypatch.setitem(sys.modules, "redis", module)

        result = await self._tool(poll_interval=0.01).arun(
            condition="redis_key", target="task:done", timeout=5
        )
        assert "condition met" in result

    @pytest.mark.asyncio
    async def test_arun_redis_key_timeout(self, monkeypatch):
        import types

        class FakeRedis:
            def __init__(self, *a, **k):
                pass

            def exists(self, key):
                return 0

            def close(self):
                pass

        module = types.ModuleType("redis")
        module.Redis = FakeRedis
        monkeypatch.setitem(sys.modules, "redis", module)

        with pytest.raises(ValueError, match="timed out"):
            await self._tool(poll_interval=0.01).arun(
                condition="redis_key", target="task:done", timeout=0.1
            )

    @pytest.mark.asyncio
    async def test_arun_unknown_condition(self):
        with pytest.raises(ValueError, match="unknown condition: kafka"):
            await self._tool().arun(condition="kafka", target="x")

    @pytest.mark.asyncio
    async def test_arun_unknown_status(self):
        with pytest.raises(ValueError, match="unknown status expectation"):
            await self._tool().arun(
                condition="url", target="http://x", status="sometimes"
            )

    def test_unknown_condition(self):
        with pytest.raises(ValueError, match="unknown condition: kafka"):
            self._tool().run(condition="kafka", target="x")

    def test_unknown_status(self):
        with pytest.raises(ValueError, match="unknown status expectation"):
            self._tool().run(condition="url", target="http://x", status="sometimes")

    def test_schema_condition_required(self):
        from draf.harness import tool_to_schema
        from draf.tool.builtin import WaitForTool

        schema = tool_to_schema(WaitForTool({}))["function"]["parameters"]
        assert "condition" in schema["required"]

    def test_registered(self):
        from draf.tool.registry import default_tool_registry

        assert "wait_for" in default_tool_registry.list()
