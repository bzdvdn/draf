"""Tests for the ``Command`` node return value (update + goto routing)."""

import pytest

from draf.node import Command, Transform
from draf.node.node import Node


class _Set(Node):
    """A node that writes *value* under *key* and optionally routes."""

    type = "set"

    def __init__(self, key, value, goto=None, *, update_only=False):
        super().__init__()
        self._key = key
        self._value = value
        self._goto = goto
        self._update_only = update_only

    async def execute(self, ctx, state):
        if self._goto is not None:
            return Command(update={self._key: self._value}, goto=self._goto)
        if self._update_only:
            return Command(update={self._key: self._value})
        return {self._key: self._value}


async def _run(graph, state=None, **kwargs):
    return await graph.run(state or {}, max_iterations=20, **kwargs)


class TestCommandRouting:
    @pytest.mark.asyncio
    async def test_goto_routes_directly(self):
        from draf.flow import Flow

        flow = Flow("cmd")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto="admin"), id="decider")
        flow.step(
            Transform(action="value", value="ADMIN_OK", output_key="who"), id="admin"
        )
        r = await _run(flow.compile())
        assert r["who"] == "ADMIN_OK"

    @pytest.mark.asyncio
    async def test_goto_skips_intermediate_edge(self):
        """goto jumps straight to the target, bypassing condition edges."""
        from draf.flow import Flow

        flow = Flow("cmd")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto="admin"), id="decider")
        # an unconditional sibling the Command must skip
        flow.step(
            Transform(action="value", value="WRONG", output_key="who"), id="other"
        )
        flow.step(
            Transform(action="value", value="ADMIN_OK", output_key="who"), id="admin"
        )
        r = await _run(flow.compile())
        assert r["who"] == "ADMIN_OK"

    @pytest.mark.asyncio
    async def test_command_update_only_keeps_normal_routing(self):
        """Command(update=...) without goto routes along the normal edges."""
        from draf.flow import Flow

        flow = Flow("cmd")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto=None, update_only=True), id="decider")
        flow.step(Transform(action="value", value="OK", output_key="who"), id="next")
        r = await _run(flow.compile())
        assert r["who"] == "OK"

    @pytest.mark.asyncio
    async def test_goto_targets_must_be_reachable(self):
        from draf.errors import WorkflowError
        from draf.flow import Flow

        flow = Flow("bad")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto="missing"), id="decider")
        flow.step(Transform(action="value", value="OK", output_key="who"), id="next")
        with pytest.raises(WorkflowError, match="unknown node"):
            await _run(flow.compile())

    @pytest.mark.asyncio
    async def test_stop_terminates_run(self):
        from draf.flow import Flow

        flow = Flow("stop")
        flow.step(_Set("who", "FIRST"), id="start")
        flow.step(_Set("who", "STOPPED", goto=Command.STOP), id="decider")
        flow.step(
            Transform(action="value", value="NEVER", output_key="who"), id="never"
        )
        r = await _run(flow.compile())
        assert r["who"] == "STOPPED"

    @pytest.mark.asyncio
    async def test_command_merges_via_reducers(self):
        from draf.flow import Flow

        def append(old, new):
            return (old or []) + new

        flow = Flow("reducers")
        flow.step(_Set("items", ["a"], goto=None, update_only=True), id="s1")
        flow.step(_Set("items", ["b"], goto=None, update_only=True), id="s2")
        r = await flow.compile().run({}, max_iterations=20, reducers={"items": append})
        assert r["items"] == ["a", "b"]


class TestCommandWithPlainFunctions:
    @pytest.mark.asyncio
    async def test_step_accepts_async_function(self):
        from draf.flow import Flow

        async def double(ctx, state):
            return {"doubled": int(state.get("n", 0)) * 2}

        flow = Flow("fn")
        flow.step(Transform(action="value", value="4", output_key="n"))
        flow.step(double)
        r = await _run(flow.compile())
        assert r["doubled"] == 8

    @pytest.mark.asyncio
    async def test_step_accepts_sync_function(self):
        from draf.flow import Flow

        def square(ctx, state):
            return {"squared": int(state.get("n", 0)) ** 2}

        flow = Flow("fn")
        flow.step(Transform(action="value", value="4", output_key="n"))
        flow.step(square)
        r = await _run(flow.compile())
        assert r["squared"] == 16

    @pytest.mark.asyncio
    async def test_step_function_may_return_command(self):
        from draf.flow import Flow

        def gate(ctx, state):
            return Command(update={"seen": True}, goto=Command.STOP)

        flow = Flow("fn")
        flow.step(gate, id="gate")
        flow.step(Transform(action="value", value="NEVER", output_key="x"), id="never")
        r = await _run(flow.compile())
        assert r["seen"] is True
        assert "x" not in r

    @pytest.mark.asyncio
    async def test_step_function_bad_return_type(self):
        from draf.flow import Flow

        def bad(ctx, state):
            return "nope"

        flow = Flow("fn")
        flow.step(bad)
        with pytest.raises(TypeError, match="must return a dict or Command"):
            await _run(flow.compile())

    def test_function_node_type_is_function_name(self):
        from draf.flow import Flow

        async def my_worker(ctx, state):
            return {}

        g = Flow("fn").step(my_worker).compile()
        types = {n.type for nid, n in g.nodes.items()}
        assert "my_worker" in types

    def test_step_rejects_non_callable(self):
        from draf.flow import Flow

        with pytest.raises(TypeError, match="must be a Node or function"):
            Flow("fn").step({"action": "uppercase"})  # type: ignore[arg-type]
