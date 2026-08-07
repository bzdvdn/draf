import pytest


class TestTypedState:
    def test_reducers_from_typeddict_extracts_annotated(self):
        from typing import Annotated, TypedDict

        from teff.state import reducers_from_typeddict

        def dummy_reducer(old, new):
            return new

        class MyState(TypedDict):
            items: Annotated[list, "append"]
            name: str
            score: Annotated[int, dummy_reducer]

        reducers = reducers_from_typeddict(MyState)
        assert reducers["items"] == "append"
        assert reducers["score"] is dummy_reducer
        assert "name" not in reducers

    def test_reducers_from_typeddict_no_annotations_returns_empty(self):
        from typing import TypedDict

        from teff.state import reducers_from_typeddict

        class Plain(TypedDict):
            x: str

        assert reducers_from_typeddict(Plain) == {}

    def test_apply_override(self):
        from teff.state import apply_reducers

        state = {"a": 1}
        apply_reducers(state, {"a": 2}, {"a": "override"})
        assert state["a"] == 2

    def test_apply_append(self):
        from teff.state import apply_reducers

        state = {"msgs": ["hello"]}
        apply_reducers(state, {"msgs": ["world"]}, {"msgs": "append"})
        assert state["msgs"] == ["hello", "world"]

    def test_apply_append_new_key(self):
        from teff.state import apply_reducers

        state = {}
        apply_reducers(state, {"msgs": ["hello"]}, {"msgs": "append"})
        assert state["msgs"] == ["hello"]

    def test_apply_append_non_list_value(self):
        from teff.state import apply_reducers

        state = {"msgs": ["hello"]}
        apply_reducers(
            state, {"msgs": {"role": "assistant", "content": "hi"}}, {"msgs": "append"}
        )
        assert state["msgs"] == ["hello", {"role": "assistant", "content": "hi"}]

    def test_reducer_appends_classification(self):
        from teff.state import reducer_appends

        def callable_reducer(old, new):
            return old + new

        assert reducer_appends("append") is True
        assert reducer_appends(callable_reducer) is True
        assert reducer_appends(None) is False
        assert reducer_appends("override") is False
        assert reducer_appends("keep") is False

    def test_apply_keep_existing(self):
        from teff.state import apply_reducers

        state = {"x": "first"}
        apply_reducers(state, {"x": "second"}, {"x": "keep"})
        assert state["x"] == "first"

    def test_apply_keep_new(self):
        from teff.state import apply_reducers

        state = {}
        apply_reducers(state, {"x": "first"}, {"x": "keep"})
        assert state["x"] == "first"

    def test_apply_callable_reducer(self):
        from teff.state import apply_reducers

        def sum_counter(old, new):
            return (old or 0) + new

        state = {"count": 5}
        apply_reducers(state, {"count": 3}, {"count": sum_counter})
        assert state["count"] == 8

    def test_apply_callable_first_time(self):
        from teff.state import apply_reducers

        def sum_counter(old, new):
            return (old or 0) + new

        state = {}
        apply_reducers(state, {"count": 3}, {"count": sum_counter})
        assert state["count"] == 3

    def test_reducers_from_yaml_schema_with_dict_values(self):
        from teff.state import reducers_from_yaml_schema

        schema = {
            "messages": {"reducer": "append", "type": "list"},
            "status": {"reducer": "keep"},
            "count": {"reducer": "override"},
        }
        reducers = reducers_from_yaml_schema(schema)
        assert reducers["messages"] == "append"
        assert reducers["status"] == "keep"
        assert reducers["count"] == "override"

    def test_reducers_from_yaml_schema_defaults_to_override(self):
        from teff.state import reducers_from_yaml_schema

        schema = {"x": {}, "y": {"reducer": "append"}}
        reducers = reducers_from_yaml_schema(schema)
        assert reducers["x"] == "override"
        assert reducers["y"] == "append"

    def test_reducers_from_yaml_schema_ignores_invalid_reducer(self):
        from teff.state import reducers_from_yaml_schema

        schema = {"x": {"reducer": "invalid"}, "y": {"reducer": "append"}}
        reducers = reducers_from_yaml_schema(schema)
        assert "x" not in reducers
        assert reducers["y"] == "append"

    def test_reducers_from_yaml_schema_empty(self):
        from teff.state import reducers_from_yaml_schema

        assert reducers_from_yaml_schema({}) == {}

    def test_reducers_from_yaml_schema_with_non_dict(self):
        from teff.state import reducers_from_yaml_schema

        schema = {"a": 42}
        assert reducers_from_yaml_schema(schema) == {}

    def test_no_reducer_falls_back_to_override(self):
        from teff.state import apply_reducers

        state = {"x": 1}
        apply_reducers(state, {"x": 2}, {})
        assert state["x"] == 2

    def test_unrelated_keys_untouched(self):
        from teff.state import apply_reducers

        state = {"keep": "me"}
        apply_reducers(state, {"x": 1}, {"x": "override"})
        assert state["keep"] == "me"

    @pytest.mark.asyncio
    async def test_graph_run_with_reducers(self):
        from teff.graph import Edge, Graph
        from teff.node import Node

        class AppendOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                return {"msgs": ["one"]}

        class AppendTwo(Node):
            type = "at"

            async def execute(self, ctx, state):
                return {"msgs": ["two"]}

        g = Graph(
            nodes={"a": AppendOne({}), "b": AppendTwo({})},
            edges=[Edge("a", "b")],
            entry_point="a",
        )
        r = await g.run(state={}, reducers={"msgs": "append"})
        assert r["msgs"] == ["one", "two"]

    @pytest.mark.asyncio
    async def test_graph_run_without_reducers_backward_compat(self):
        from teff.graph import Edge, Graph
        from teff.node import Node

        class WriteX(Node):
            type = "wx"

            async def execute(self, ctx, state):
                return {"x": 1}

        class WriteXAgain(Node):
            type = "wx2"

            async def execute(self, ctx, state):
                return {"x": 2}

        g = Graph(
            nodes={"a": WriteX({}), "b": WriteXAgain({})},
            edges=[Edge("a", "b")],
            entry_point="a",
        )
        r = await g.run(state={})
        assert r["x"] == 2  # last write wins

    @pytest.mark.asyncio
    async def test_reducers_via_typeddict(self):
        from typing import Annotated, TypedDict

        from teff.graph import Edge, Graph
        from teff.node import Node
        from teff.state import reducers_from_typeddict

        class ChatState(TypedDict):
            msgs: Annotated[list, "append"]

        class AddMsg(Node):
            type = "am"

            async def execute(self, ctx, state):
                return {"msgs": ["hello"]}

        class AddMsg2(Node):
            type = "am2"

            async def execute(self, ctx, state):
                return {"msgs": ["world"]}

        g = Graph(
            nodes={"a": AddMsg({}), "b": AddMsg2({})},
            edges=[Edge("a", "b")],
            entry_point="a",
        )
        reducers = reducers_from_typeddict(ChatState)
        r = await g.run(state={}, reducers=reducers)
        assert r["msgs"] == ["hello", "world"]


class TestStateClass:
    def test_construct_and_access(self):
        from typing import TypedDict

        from teff.state import State

        class S(TypedDict):
            name: str
            score: int

        state = State(S, {"name": "alice", "score": 10})
        assert state["name"] == "alice"
        assert state["score"] == 10
        assert len(state) == 2
        assert "name" in state

    def test_merge_with_reducers(self):
        from typing import Annotated, TypedDict

        from teff.state import State

        class S(TypedDict):
            msgs: Annotated[list, "append"]
            x: str

        state = State(S, {"msgs": ["a"], "x": "old"})
        state.merge({"msgs": ["b", "c"], "x": "new"})
        assert state["msgs"] == ["a", "b", "c"]
        assert state["x"] == "new"

    def test_merge_keep(self):
        from typing import Annotated, TypedDict

        from teff.state import State

        class S(TypedDict):
            first: Annotated[str, "keep"]

        state = State(S, {"first": "original"})
        state.merge({"first": "override"})
        assert state["first"] == "original"

    def test_merge_callable_reducer(self):
        from typing import Annotated, TypedDict

        from teff.state import State

        def add(old, new):
            return (old or 0) + new

        class S(TypedDict):
            total: Annotated[int, add]

        state = State(S, {"total": 5})
        state.merge({"total": 3})
        assert state["total"] == 8

    def test_setitem_direct(self):
        from typing import Annotated, TypedDict

        from teff.state import State

        class S(TypedDict):
            x: Annotated[str, "keep"]

        state = State(S, {"x": "original"})
        state["x"] = "direct"
        assert state["x"] == "direct"

    def test_dict_methods(self):
        from typing import TypedDict

        from teff.state import State

        class S(TypedDict):
            a: str
            b: int

        state = State(S, {"a": "x", "b": 1})
        assert set(state.keys()) == {"a", "b"}
        assert set(state.values()) == {"x", 1}
        assert set(state.items()) == {("a", "x"), ("b", 1)}

    def test_get_with_default(self):
        from typing import TypedDict

        from teff.state import State

        class S(TypedDict):
            a: str

        state = State(S, {"a": "x"})
        assert state.get("a") == "x"
        assert state.get("missing", "default") == "default"

    def test_copy_returns_dict(self):
        from typing import TypedDict

        from teff.state import State

        class S(TypedDict):
            a: str

        state = State(S, {"a": "x"})
        d = state.copy()
        assert d == {"a": "x"}
        assert isinstance(d, dict)

    def test_repr(self):
        from typing import TypedDict

        from teff.state import State

        class S(TypedDict):
            x: str

        state = State(S, {"x": "hello"})
        assert repr(state) == "{'x': 'hello'}"

    def test_load_workflow_parses_state_schema_and_initial(self):
        import os
        import tempfile

        yaml_content = """
name: test
state:
  schema:
    messages:
      reducer: append
    status:
      reducer: keep
  initial:
    status: active
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
edges: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            from teff.yaml import load_workflow

            graph, tools, initial, reducers = load_workflow(path)
            assert initial == {"status": "active"}
            assert reducers == {"messages": "append", "status": "keep"}
        finally:
            os.unlink(path)

    def test_load_workflow_no_state_returns_empty(self):
        import os
        import tempfile

        yaml_content = """
name: test
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
edges: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            from teff.yaml import load_workflow

            graph, tools, initial, reducers = load_workflow(path)
            assert initial == {}
            assert reducers == {}
        finally:
            os.unlink(path)

    def test_roundtrip_via_graph(self):
        from typing import Annotated, TypedDict

        from teff.graph import Edge, Graph
        from teff.node import Node
        from teff.state import State

        class S(TypedDict):
            msgs: Annotated[list, "append"]

        class AppendOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                return {"msgs": ["one"]}

        class AppendTwo(Node):
            type = "at"

            async def execute(self, ctx, state):
                return {"msgs": ["two"]}

        g = Graph(
            nodes={"a": AppendOne({}), "b": AppendTwo({})},
            edges=[Edge("a", "b")],
            entry_point="a",
        )
        import asyncio

        state = State(S)
        result = asyncio.run(g.run(state))
        assert result is state
        assert result["msgs"] == ["one", "two"]


class TestStateSchema:
    def test_schema_to_jsonschema_string(self):
        from teff.state import state_schema_to_jsonschema

        schema = state_schema_to_jsonschema({"status": "string", "count": "integer"})
        assert schema["type"] == "object"
        assert schema["properties"]["status"] == {"type": "string"}
        assert schema["properties"]["count"] == {"type": "integer"}

    def test_schema_to_jsonschema_dict_spec(self):
        from teff.state import state_schema_to_jsonschema

        schema = state_schema_to_jsonschema(
            {"count": {"type": "integer", "minimum": 0}}
        )
        assert schema["properties"]["count"] == {
            "type": "integer",
            "minimum": 0,
        }

    def test_schema_to_jsonschema_list(self):
        from teff.state import state_schema_to_jsonschema

        schema = state_schema_to_jsonschema({"tags": "list"})
        assert schema["properties"]["tags"] == {"type": "array"}

    def test_schema_to_jsonschema_required(self):
        from teff.state import state_schema_to_jsonschema

        schema = state_schema_to_jsonschema(
            {"status": {"type": "string", "required": True}}
        )
        assert schema["required"] == ["status"]

    def test_schema_to_jsonschema_unknown_type_unconstrained(self):
        from teff.state import state_schema_to_jsonschema

        schema = state_schema_to_jsonschema({"x": "custom_thing"})
        assert schema["properties"]["x"] == {}

    def test_validate_state_passes(self):
        from teff.state import validate_state

        errors = validate_state(
            {"status": "active", "count": 3},
            {"status": "string", "count": "integer"},
        )
        assert errors == []

    def test_validate_state_reports_type_mismatch(self):
        from teff.state import validate_state

        errors = validate_state(
            {"count": "not-an-int"},
            {"count": "integer"},
        )
        assert len(errors) == 1
        assert "count" in errors[0]
        assert "integer" in errors[0]

    def test_validate_state_reports_missing_required(self):
        from teff.state import validate_state

        errors = validate_state(
            {"status": "active"},
            {
                "status": {"type": "string", "required": True},
                "count": {"type": "integer", "required": True},
            },
        )
        assert len(errors) == 1
        assert "count" in errors[0]

    def test_validate_state_ignores_reducer_keys(self):
        from teff.state import validate_state

        errors = validate_state(
            {"messages": ["a"]},
            {"messages": {"type": "list", "reducer": "append"}},
        )
        assert errors == []

    @pytest.mark.asyncio
    async def test_graph_run_state_schema_passes(self):
        from teff.graph import Graph
        from teff.node import Node

        class NoOp(Node):
            type = "noop"

            async def execute(self, ctx, state):
                return {}

        g = Graph(nodes={"a": NoOp({})}, edges=[], entry_point="a")
        result = await g.run(state={"status": "ok"}, state_schema={"status": "string"})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_graph_run_state_schema_rejects(self):
        from teff.errors import ConfigError
        from teff.graph import Graph
        from teff.node import Node

        class NoOp(Node):
            type = "noop"

            async def execute(self, ctx, state):
                return {}

        g = Graph(nodes={"a": NoOp({})}, edges=[], entry_point="a")
        with pytest.raises(ConfigError) as excinfo:
            await g.run(state={"status": 42}, state_schema={"status": "string"})
        assert "status" in str(excinfo.value)

    def test_load_workflow_validates_initial_against_schema(self):
        import os
        import tempfile

        from teff.errors import ConfigError

        yaml_content = """
name: test
state:
  schema:
    count: integer
  initial:
    count: not-a-number
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
edges: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            from teff.yaml import load_workflow

            with pytest.raises(ConfigError) as excinfo:
                load_workflow(path)
            assert "state.initial" in str(excinfo.value)
        finally:
            os.unlink(path)
