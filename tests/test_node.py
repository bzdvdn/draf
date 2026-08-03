import pytest


class TestNodeABC:
    def test_abstract_base_cannot_be_instantiated(self):
        from draf.node import Node

        with pytest.raises(TypeError):
            Node()

    def test_subclass_with_execute_works(self):
        import asyncio

        from draf.node import Node

        class MyNode(Node):
            type = "my"

            async def execute(self, ctx, state):
                state["x"] = 1
                return state

        n = MyNode({"a": 1})
        assert n.type == "my"
        assert n.config == {"a": 1}
        r = asyncio.run(n.execute(None, {}))
        assert r == {"x": 1}


class TestNodeDecorator:
    def test_registers_in_default_registry(self):
        import asyncio

        from draf.node import default_registry, node

        @node("test_simple")
        async def fn(ctx, state):
            state["done"] = True
            return state

        assert "test_simple" in default_registry.list()
        created = default_registry.create("test_simple", {})
        r = asyncio.run(created.execute(None, {}))
        assert r == {"done": True}

    def test_with_typed_config(self):
        import asyncio

        from draf.node import default_registry, node

        @node("test_config", config=dict)
        async def fn(ctx, cfg, state):
            state["val"] = cfg["x"]
            return state

        created = default_registry.create("test_config", {"x": 42})
        r = asyncio.run(created.execute(None, {}))
        assert r == {"val": 42}

    def test_non_async_function_raises(self):
        from draf.node import node

        with pytest.raises(TypeError, match="must be async"):

            @node("bad")
            def sync_fn(ctx, state):
                return state


class _DummyNode:
    type = "dummy"

    def __init__(self, config=None):
        self.config = config or {}

    async def execute(self, ctx, state):
        return state


class TestNodeRegistry:
    def test_isolated_from_default(self):
        from draf.node import NodeRegistry

        reg = NodeRegistry()
        assert reg.list() == []

    def test_register_and_create(self):
        from draf.node import NodeRegistry

        reg = NodeRegistry()
        reg.register("custom", lambda cfg: _DummyNode(cfg))
        assert "custom" in reg.list()
        n = reg.create("custom", {})
        assert isinstance(n, _DummyNode)

    def test_unknown_type_raises(self):
        from draf.node import NodeRegistry

        reg = NodeRegistry()
        with pytest.raises(KeyError):
            reg.create("nonexistent", {})


class TestExecContext:
    def test_tool_lookup(self):
        from draf.node import ExecContext
        from draf.tool import Tool

        class FT(Tool):
            name = "ft"
            description = "ft"

            def run(self):
                return "ok"

        ctx = ExecContext(state={}, tools={"ft": FT()})
        assert ctx.tool("ft").run() == "ok"

    def test_missing_tool_raises_keyerror(self):
        from draf.node import ExecContext

        ctx = ExecContext(state={}, tools={})
        with pytest.raises(KeyError):
            ctx.tool("nope")


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        from draf.node import ExecContext, Node, Retry

        attempt = 0

        class Flaky(Node):
            type = "flaky"

            async def execute(self, ctx, state):
                nonlocal attempt
                attempt += 1
                if attempt < 2:
                    raise ValueError("nope")
                return {"ok": True}

        retry = Retry(Flaky({}), max_retries=3)
        ctx = ExecContext(state={}, tools={})
        result = await retry.execute(ctx, {})
        assert result == {"ok": True}
        assert attempt == 2

    @pytest.mark.asyncio
    async def test_retry_exhausts_and_raises(self):
        from draf.node import ExecContext, Node, Retry

        class AlwaysFail(Node):
            type = "af"

            async def execute(self, ctx, state):
                raise RuntimeError("boom")

        retry = Retry(AlwaysFail({}), max_retries=2, delay=0.01)
        ctx = ExecContext(state={}, tools={})
        with pytest.raises(RuntimeError, match="boom"):
            await retry.execute(ctx, {})

    @pytest.mark.asyncio
    async def test_retry_passthrough_on_success(self):
        from draf.node import ExecContext, Node, Retry

        class Fast(Node):
            type = "fast"

            async def execute(self, ctx, state):
                return {"done": True}

        retry = Retry(Fast({}), max_retries=3)
        ctx = ExecContext(state={}, tools={})
        result = await retry.execute(ctx, {})
        assert result == {"done": True}

    @pytest.mark.asyncio
    async def test_retry_backoff_scales_delay(self):
        from draf.node import ExecContext, Node, Retry

        sleeps = []
        import asyncio

        original_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            await original_sleep(0)

        asyncio.sleep = fake_sleep
        try:
            attempt = 0

            class Flaky(Node):
                type = "flaky"

                async def execute(self, ctx, state):
                    nonlocal attempt
                    attempt += 1
                    raise ValueError("nope")

            retry = Retry(Flaky({}), max_retries=4, delay=1.0, backoff=2.0)
            ctx = ExecContext(state={}, tools={})
            with pytest.raises(ValueError, match="nope"):
                await retry.execute(ctx, {})
        finally:
            asyncio.sleep = original_sleep
        assert sleeps == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_retry_timeout_bounds_each_attempt(self):
        import asyncio

        from draf.node import ExecContext, Node, Retry

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                await asyncio.sleep(10)
                return {}

        retry = Retry(Slow({}), max_retries=2, timeout=0.01)
        ctx = ExecContext(state={}, tools={})
        with pytest.raises(asyncio.TimeoutError):
            await retry.execute(ctx, {})

    @pytest.mark.asyncio
    async def test_retry_on_matching_exception(self):
        from draf.node import ExecContext, Node, Retry

        attempt = 0

        class Flaky(Node):
            type = "flaky"

            async def execute(self, ctx, state):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise ValueError("nope")
                return {"ok": True}

        retry = Retry(Flaky({}), max_retries=3, retry_on=["ValueError"])
        ctx = ExecContext(state={}, tools={})
        result = await retry.execute(ctx, {})
        assert result == {"ok": True}
        assert attempt == 2

    @pytest.mark.asyncio
    async def test_retry_on_non_matching_raises_immediately(self):
        from draf.node import ExecContext, Node, Retry

        attempt = 0

        class Flaky(Node):
            type = "flaky"

            async def execute(self, ctx, state):
                nonlocal attempt
                attempt += 1
                raise ValueError("nope")

        retry = Retry(Flaky({}), max_retries=3, retry_on=["TimeoutError"])
        ctx = ExecContext(state={}, tools={})
        with pytest.raises(ValueError, match="nope"):
            await retry.execute(ctx, {})
        assert attempt == 1

    @pytest.mark.asyncio
    async def test_wrap_with_retry_noop_without_config(self):
        from draf.node import Node
        from draf.node.retry import Retry, wrap_with_retry

        class Plain(Node):
            type = "plain"

            async def execute(self, ctx, state):
                return {}

        node = Plain({})
        assert wrap_with_retry(node, None) is node
        assert wrap_with_retry(node, {"enabled": False}) is node
        wrapped = wrap_with_retry(node, {"max_retries": 2, "delay": 0.01})
        assert isinstance(wrapped, Retry)
