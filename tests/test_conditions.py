"""Tests for callable edge conditions and ``Flow.step(when=...)`` guards."""


import pytest

from draf.flow import Flow
from draf.graph.conditions import evaluate
from draf.node import Transform


class TestEvaluateCallable:
    def test_callable_predicate(self):
        assert evaluate(lambda s: s.get("n", 0) > 5, {"n": 10}) is True
        assert evaluate(lambda s: s.get("n", 0) > 5, {"n": 1}) is False

    def test_list_membership_predicate(self):
        assert evaluate(lambda s: "tiles" in s.get("tags", []), {"tags": ["tiles"]})
        assert not evaluate(lambda s: "paint" in s.get("tags", []), {"tags": ["tiles"]})

    def test_string_conditions_unchanged(self):
        assert evaluate("status=ok", {"status": "ok"}) is True
        assert evaluate("count>=3", {"count": "5"}) is True


class TestGraphCallableConditions:
    @pytest.mark.asyncio
    async def test_callable_condition_routes_in_graph(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Write(Node):
            type = "write"

            def __init__(self, key, value):
                super().__init__()
                self._key, self._value = key, value

            async def execute(self, ctx, state):
                return {self._key: self._value}

        g = Graph(
            nodes={"s": Write("seen", "y"), "a": Write("who", "YES"), "b": Write("who", "NO")},
            edges=[
                Edge("s", "a", lambda s: s.get("n", 0) > 5),
                Edge("s", "b"),
            ],
            entry_point="s",
        )
        assert (await g.run({"n": 10}))["who"] == "YES"
        assert (await g.run({"n": 1}))["who"] == "NO"

    @pytest.mark.asyncio
    async def test_error_edges_still_work_alongside_callables(self):
        from draf.errors import NodeError
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Boom(Node):
            type = "boom"

            async def execute(self, ctx, state):
                raise NodeError("boom")

        class Rescue(Node):
            type = "rescue"

            async def execute(self, ctx, state):
                return {"rescued": True}

        g = Graph(
            nodes={"s": Boom(), "r": Rescue()},
            edges=[Edge("s", "r", "__error__")],
            entry_point="s",
        )
        r = await g.run({})
        assert r["rescued"] is True


class TestFlowGuardedStep:
    def _flow(self):
        flow = Flow("g")
        flow.step(Transform(action="value", value="keep", output_key="seen"), id="decider")
        flow.step(
            Transform(action="value", value="YES", output_key="who"),
            id="yes",
            when=lambda s: s.get("v") == "x",
        )
        flow.default(
            Transform(action="value", value="NO", output_key="who"), id="no"
        )
        return flow

    def test_wiring(self):
        g = self._flow().compile()
        edges = {(e.source_id, e.target_id) for e in g.edges}
        assert ("decider", "yes") in edges
        assert ("decider", "no") in edges
        by_target = {e.target_id: e.condition for e in g.edges}
        assert callable(by_target["yes"])
        assert by_target["no"] is None

    @pytest.mark.asyncio
    async def test_guard_match_takes_branch(self):
        r = await self._flow().compile().run({"v": "x"}, max_iterations=10)
        assert r["who"] == "YES"

    @pytest.mark.asyncio
    async def test_guard_miss_takes_default(self):
        r = await self._flow().compile().run({"v": "yy"}, max_iterations=10)
        assert r["who"] == "NO"

    @pytest.mark.asyncio
    async def test_string_when_condition(self):
        flow = Flow("str")
        flow.step(Transform(action="value", value="ok", output_key="s"), id="decider")
        flow.step(
            Transform(action="value", value="MATCH", output_key="out"),
            id="m",
            when="s=ok",
        )
        flow.default(Transform(action="value", value="OTHER", output_key="out"), id="d")
        r = await flow.compile().run({}, max_iterations=10)
        assert r["out"] == "MATCH"

    def test_when_requires_preceding_node(self):
        with pytest.raises(ValueError, match="requires a preceding node"):
            Flow("empty").step(
                Transform(action="value", value="x", output_key="o"),
                id="a",
                when=lambda s: True,
            )

    def test_mermaid_renders_callable(self):
        diagram = self._flow().compile().to_mermaid()
        assert "when:" in diagram

    def test_yaml_rejects_callable_condition(self):
        with pytest.raises(ValueError, match="callable edge condition"):
            self._flow().compile().to_yaml()


class TestCommandRoutingWithConditions:
    @pytest.mark.asyncio
    async def test_command_goto_from_guarded_branch(self):
        """A node may return Command(goto=...) to jump anywhere in the graph."""
        from draf.graph import Edge, Graph
        from draf.node import Command, Node

        class Route(Node):
            type = "route"

            async def execute(self, ctx, state):
                target = "admin" if state.get("role") == "admin" else "guest"
                return Command(update={"seen": True}, goto=target)

        class Write(Node):
            type = "write"

            def __init__(self, value):
                super().__init__()
                self._value = value

            async def execute(self, ctx, state):
                return {"who": self._value}

        g = Graph(
            nodes={"decider": Route(), "admin": Write("ADMIN_OK"), "guest": Write("GUEST_OK")},
            edges=[Edge("decider", "admin"), Edge("decider", "guest")],
            entry_point="decider",
        )
        assert (await g.run({"role": "admin"}, max_iterations=10))["who"] == "ADMIN_OK"
        assert (await g.run({"role": "user"}, max_iterations=10))["who"] == "GUEST_OK"
