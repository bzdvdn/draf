"""Tests for the declarative ``loop`` node (body/until/max_rounds)."""

import pytest

from teff.node import Transform
from teff.node.node import Node


class _Counter(Node):
    """Increment *key* by 1 each execution."""

    type = "counter"

    def __init__(self, key="n"):
        super().__init__()
        self._key = key

    async def execute(self, ctx, state):
        current = int(state.get(self._key, 0))
        return {self._key: str(current + 1)}


class TestLoopNode:
    @pytest.mark.asyncio
    async def test_loops_until_condition(self):
        from teff.node import Loop

        node = Loop(_Counter("n"), key="n", until="2", max_rounds=10)
        state = {}
        out = await node.execute(None, state)
        assert out == {}
        assert state["n"] == "2"

    @pytest.mark.asyncio
    async def test_bounded_by_max_rounds(self):
        from teff.node import Loop

        node = Loop(_Counter("n"), key="n", until="999", max_rounds=3)
        state = {}
        await node.execute(None, state)
        assert state["n"] == "3"

    @pytest.mark.asyncio
    async def test_accepts_declarative_body(self):
        from teff.node import Loop

        node = Loop(
            {
                "type": "transform",
                "action": "value",
                "value": "x",
                "output_key": "out",
            },
            key="out",
            until="x",
            max_rounds=2,
        )
        state = {}
        await node.execute(None, state)
        assert state["out"] == "x"

    @pytest.mark.asyncio
    async def test_body_list_runs_in_order(self):
        from teff.node import Loop

        seen = []

        class _Record(Node):
            type = "record"

            async def execute(self, ctx, state):
                seen.append(state.get("out"))
                return {"out": "z"}

        node = Loop(
            [
                Transform(action="value", value="a", output_key="out"),
                _Record(),
            ],
            key="out",
            until="z",
            max_rounds=2,
        )
        state = {}
        await node.execute(None, state)
        assert seen == ["a"]
        assert state["out"] == "z"

    @pytest.mark.asyncio
    async def test_missing_key_raises(self):
        from teff.node import Loop

        node = Loop(_Counter("n"), until="2")
        with pytest.raises(ValueError, match="config.key"):
            await node.execute(None, {})

    def test_registered_in_registry(self):
        from teff.node import default_registry

        assert "loop" in default_registry.list()
        node = default_registry.create(
            "loop",
            {
                "body": {
                    "type": "transform",
                    "action": "value",
                    "value": "x",
                    "output_key": "o",
                },
                "key": "o",
                "until": "x",
            },
        )
        assert node.type == "loop"

    def test_missing_body_raises(self):
        from teff.errors import ConfigError
        from teff.node import default_registry

        with pytest.raises(ConfigError, match="config.body"):
            default_registry.create("loop", {"key": "o", "until": "x"})


class TestLoopViaYaml:
    @pytest.mark.asyncio
    async def test_yaml_loop_counts_up(self, tmp_path):
        from teff.node.registry import default_registry
        from teff.yaml import load_workflow

        counter_node = _Counter("n")
        default_registry.register("counter", lambda cfg: counter_node)
        try:
            path = tmp_path / "wf.yaml"
            path.write_text(
                """\
name: loop
steps:
  - id: seed
    type: transform
    config: {action: value, value: "0", output_key: n}
  - id: repeat
    type: loop
    config:
      key: n
      until: "2"
      max_rounds: 10
      body:
        - type: counter
edges:
  - {from: seed, to: repeat}
"""
            )
            graph, _, _, _ = load_workflow(str(path))
            r = await graph.run({}, max_iterations=50)
            assert r["n"] == "2"
        finally:
            default_registry._factories.pop("counter")
