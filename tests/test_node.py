import pytest


class TestNodeABC:
    def test_abstract_base_cannot_be_instantiated(self):
        from teff.node import Node

        with pytest.raises(TypeError):
            Node()

    def test_subclass_with_execute_works(self):
        import asyncio

        from teff.node import Node

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

        from teff.node import default_registry, node

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

        from teff.node import default_registry, node

        @node("test_config", config=dict)
        async def fn(ctx, cfg, state):
            state["val"] = cfg["x"]
            return state

        created = default_registry.create("test_config", {"x": 42})
        r = asyncio.run(created.execute(None, {}))
        assert r == {"val": 42}

    def test_non_async_function_raises(self):
        from teff.node import node

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
        from teff.node import NodeRegistry

        reg = NodeRegistry()
        assert reg.list() == []

    def test_register_and_create(self):
        from teff.node import NodeRegistry

        reg = NodeRegistry()
        reg.register("custom", lambda cfg: _DummyNode(cfg))
        assert "custom" in reg.list()
        n = reg.create("custom", {})
        assert isinstance(n, _DummyNode)

    def test_unknown_type_raises(self):
        from teff.node import NodeRegistry

        reg = NodeRegistry()
        with pytest.raises(KeyError):
            reg.create("nonexistent", {})


class TestExecContext:
    def test_tool_lookup(self):
        from teff.node import ExecContext
        from teff.tool import Tool

        class FT(Tool):
            name = "ft"
            description = "ft"

            def run(self):
                return "ok"

        ctx = ExecContext(state={}, tools={"ft": FT()})
        assert ctx.tool("ft").run() == "ok"

    def test_missing_tool_raises_keyerror(self):
        from teff.node import ExecContext

        ctx = ExecContext(state={}, tools={})
        with pytest.raises(KeyError):
            ctx.tool("nope")


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        from teff.node import ExecContext, Node, Retry

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
        from teff.node import ExecContext, Node, Retry

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
        from teff.node import ExecContext, Node, Retry

        class Fast(Node):
            type = "fast"

            async def execute(self, ctx, state):
                return {"done": True}

        retry = Retry(Fast({}), max_retries=3)
        ctx = ExecContext(state={}, tools={})
        result = await retry.execute(ctx, {})
        assert result == {"done": True}

    @pytest.mark.asyncio
    async def test_retry_does_not_retry_graph_interrupt(self):
        from teff.node import ExecContext, Node, Retry
        from teff.node.interrupt import GraphInterrupt

        attempt = 0

        class Interrupting(Node):
            type = "interrupting"

            async def execute(self, ctx, state):
                nonlocal attempt
                attempt += 1
                raise GraphInterrupt(key="pause", prompt="wait")

        retry = Retry(Interrupting({}), max_retries=3)
        ctx = ExecContext(state={}, tools={})
        with pytest.raises(GraphInterrupt):
            await retry.execute(ctx, {})
        assert attempt == 1

    @pytest.mark.asyncio
    async def test_retry_backoff_scales_delay(self):
        from teff.node import ExecContext, Node, Retry

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

        from teff.node import ExecContext, Node, Retry

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
        from teff.node import ExecContext, Node, Retry

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
        from teff.node import ExecContext, Node, Retry

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
        from teff.node import Node
        from teff.node.retry import Retry, wrap_with_retry

        class Plain(Node):
            type = "plain"

            async def execute(self, ctx, state):
                return {}

        node = Plain({})
        assert wrap_with_retry(node, None) is node
        assert wrap_with_retry(node, {"enabled": False}) is node
        wrapped = wrap_with_retry(node, {"max_retries": 2, "delay": 0.01})
        assert isinstance(wrapped, Retry)


class TestGate:
    async def _run(self, gate, state):
        from teff.node import ExecContext

        ctx = ExecContext(state=state, tools={})
        return await gate.execute(ctx, state)

    @pytest.mark.asyncio
    async def test_passing_verdict_writes_pass_value_and_increments_rounds(self):
        from teff.node import Gate

        state = {"verdict": {"ok": True}}
        out = await self._run(Gate(), state)
        assert out["decision"] == "yes"
        assert out["rounds"] == 1

    @pytest.mark.asyncio
    async def test_failing_verdict_writes_fail_value(self):
        from teff.node import Gate

        state = {"verdict": {"ok": False}}
        out = await self._run(Gate(), state)
        assert out["decision"] == "fix"
        assert out["rounds"] == 1

    @pytest.mark.asyncio
    async def test_message_copied_on_fail_and_cleared_on_pass(self):
        from teff.node import Gate

        gate = Gate(message_key="feedback")
        out = await self._run(gate, {"verdict": {"ok": False, "message": "  x  "}})
        assert out["feedback"] == "x"
        out = await self._run(gate, {"verdict": {"ok": True, "message": "x"}})
        assert out["feedback"] == ""

    @pytest.mark.asyncio
    async def test_forces_pass_after_max_rounds(self):
        from teff.node import Gate

        gate = Gate(max_rounds=3)
        state = {"verdict": {"ok": False}, "rounds": 2}
        out = await self._run(gate, state)
        assert out["decision"] == "yes"
        assert out["rounds"] == 3

    @pytest.mark.asyncio
    async def test_missing_verdict_is_treated_as_pass(self):
        from teff.node import Gate

        out = await self._run(Gate(), {"verdict": None})
        assert out["decision"] == "yes"

    @pytest.mark.asyncio
    async def test_configurable_key_names_and_values(self):
        from teff.node import Gate

        gate = Gate(
            input_key="qa",
            output_key="go",
            rounds_key="n",
            pass_value="done",
            fail_value="redo",
            max_rounds=2,
        )
        out = await self._run(gate, {"qa": {"ok": False}})
        assert out["go"] == "redo"
        assert out["n"] == 1
        out = await self._run(gate, {"qa": {"ok": False}, "n": 1})
        assert out["go"] == "done"
        assert out["n"] == 2


class TestValidate:
    async def _run(self, validate, state):
        from teff.node import ExecContext

        ctx = ExecContext(state=state, tools={})
        return await validate.execute(ctx, state)

    @pytest.mark.asyncio
    async def test_equals_match_writes_pass_and_extracts_value(self):
        from teff.node import Validate

        v = Validate(
            input_key="answer",
            strategy="equals",
            equals="да",
            output_key="go",
            value_key="discount",
        )
        out = await self._run(v, {"answer": "Да"})
        assert out["go"] == "да"
        assert out["discount"] == "Да"
        assert out["rounds"] == 1

    @pytest.mark.asyncio
    async def test_equals_miss_writes_fail_and_clears_value(self):
        from teff.node import Validate

        v = Validate(
            input_key="answer",
            strategy="equals",
            equals="да",
            output_key="go",
            value_key="discount",
        )
        out = await self._run(v, {"answer": "нет"})
        assert out["go"] == "нет"
        assert out["discount"] == ""

    @pytest.mark.asyncio
    async def test_any_of_matches_normalized_value(self):
        from teff.node import Validate

        v = Validate(
            input_key="answer", strategy="any_of", any_of=["да", "ок", "конечно"]
        )
        out = await self._run(v, {"answer": "  ОК "})
        assert out["decision"] == "да"

    @pytest.mark.asyncio
    async def test_regex_extracts_capture_group(self):
        from teff.node import Validate

        v = Validate(
            input_key="answer",
            strategy="regex",
            regex=r"(\d{3,6})",
            output_key="go",
            value_key="code",
        )
        out = await self._run(v, {"answer": "мой код 12345"})
        assert out["go"] == "да"
        assert out["code"] == "12345"

    @pytest.mark.asyncio
    async def test_check_callable_tuple_result(self):
        from teff.node import Validate

        def check(value):
            return value == "12345", "captured"

        v = Validate(
            input_key="answer",
            strategy="check",
            check=check,
            output_key="go",
            value_key="code",
        )
        out = await self._run(v, {"answer": "12345"})
        assert out["go"] == "да"
        assert out["code"] == "captured"
        out = await self._run(v, {"answer": "wrong"})
        assert out["go"] == "нет"
        assert out["code"] == ""

    @pytest.mark.asyncio
    async def test_forces_pass_after_max_rounds(self):
        from teff.node import Validate

        v = Validate(
            input_key="answer",
            strategy="equals",
            equals="да",
            max_rounds=3,
        )
        out = await self._run(v, {"answer": "нет", "rounds": 2})
        assert out["decision"] == "да"
        assert out["rounds"] == 3


class TestAsk:
    def test_detects_strategy_from_kwargs(self):
        from teff.node import Ask

        assert Ask(equals="да").strategy == "equals"
        assert Ask(any_of=["да", "ок"]).strategy == "any_of"
        assert Ask(regex=r"\d+").strategy == "regex"
        assert Ask(system="s", schema={}).strategy == "model"

    def test_classmethod_constructors(self):
        from teff.node import Ask

        assert Ask.equals("да")._expected == "да"
        assert Ask.any_of("да", "ок")._allowed == ["да", "ок"]
        assert Ask.regex(r"\d+")._pattern == r"\d+"
        assert Ask.check(lambda v: True)._predicate is not None
        assert (
            Ask.model(
                system="s", user="u", schema={"x": 1}, model="m", provider="p"
            ).strategy
            == "model"
        )

    def test_model_ask_needs_classifier(self):
        from teff.node import Ask

        assert Ask.model(
            system="s", user="u", schema={}, model="m", provider="p"
        ).needs_classifier()
        assert not Ask.equals("да").needs_classifier()

    def test_validate_node_wiring(self):
        from teff.node import Ask

        v = Ask.equals("да", decision_key="go", value_key="code").validate_node(
            "answer"
        )
        assert v.config["input_key"] == "answer"
        assert v.config["output_key"] == "go"
        assert v.config["value_key"] == "code"
        assert v.config["equals"] == "да"
        assert v.config["strategy"] == "equals"

    def test_validate_registerable_and_type(self):
        from teff.node import default_registry

        v = default_registry.create(
            "validate",
            {"input_key": "answer", "strategy": "equals", "equals": "да"},
        )
        assert v.type == "validate"
