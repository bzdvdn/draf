"""Tests for structured output: JSON Schema validation + LLM retry loop."""

import pytest


class TestJsonSchemaFromType:
    def test_dict_field_map(self):
        from draf.schema import json_schema_from_type

        schema = json_schema_from_type({"name": str, "age": int})
        assert schema["type"] == "object"
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["age"] == {"type": "integer"}
        assert schema["required"] == ["name", "age"]

    def test_raw_schema_passthrough(self):
        from draf.schema import json_schema_from_type

        raw = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert json_schema_from_type(raw) is raw

    def test_typeddict(self):
        from typing import TypedDict

        from draf.schema import json_schema_from_type

        class Person(TypedDict):
            name: str
            age: int

        schema = json_schema_from_type(Person)
        assert schema["type"] == "object"
        assert schema["required"] == ["name", "age"]

    def test_list_and_literal(self):
        from typing import Literal

        from draf.schema import json_schema_from_type

        schema = json_schema_from_type({"tags": list[str], "kind": Literal["a", "b"]})
        assert schema["properties"]["tags"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert schema["properties"]["kind"] == {"enum": ["a", "b"]}


class TestValidateJson:
    def test_valid_object(self):
        from draf.schema import validate_json

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        assert validate_json({"name": "Иван", "age": 30}, schema) == []

    def test_missing_required(self):
        from draf.schema import validate_json

        schema = {"type": "object", "required": ["name"]}
        errors = validate_json({"age": 1}, schema)
        assert any("name" in e and "missing" in e for e in errors)

    def test_wrong_type(self):
        from draf.schema import validate_json

        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        errors = validate_json({"age": "x"}, schema)
        assert any("age" in e for e in errors)

    def test_unexpected_property_rejected(self):
        from draf.schema import validate_json

        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        errors = validate_json({"a": "x", "b": 1}, schema)
        assert any("b" in e for e in errors)

    def test_enum(self):
        from draf.schema import validate_json

        schema = {"type": "string", "enum": ["да", "нет"]}
        assert validate_json("да", schema) == []
        assert validate_json("maybe", schema) != []

    def test_oneof_nullable(self):
        from draf.schema import validate_json

        schema = {"oneOf": [{"type": "null"}, {"type": "string"}]}
        assert validate_json(None, schema) == []
        assert validate_json("ok", schema) == []
        assert validate_json(42, schema) != []

    def test_array_items_and_min_items(self):
        from draf.schema import validate_json

        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 1}
        assert validate_json([1, 2], schema) == []
        assert validate_json([], schema) != []
        assert validate_json([1, "x"], schema) != []

    def test_string_limits_and_pattern(self):
        from draf.schema import validate_json

        assert validate_json("ab", {"type": "string", "minLength": 3}) != []
        assert validate_json("abc123", {"type": "string", "pattern": r"^\d+$"}) != []

    def test_boolean_integer_strict(self):
        from draf.schema import validate_json

        assert validate_json(True, {"type": "integer"}) != []
        assert validate_json(1, {"type": "integer"}) == []


class TestParseJsonObject:
    def test_direct_json(self):
        from draf.schema import parse_json_object

        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_json_embedded_in_prose(self):
        from draf.schema import parse_json_object

        text = 'Sure! Here is the JSON:\n```json\n{"name": "x"}\n```'
        assert parse_json_object(text) == {"name": "x"}

    def test_invalid_raises(self):
        from draf.schema import parse_json_object

        with pytest.raises(ValueError):
            parse_json_object("not json at all")


class _MockResponses:
    """Queue of httpx-style responses; each call pops the next one.

    A string entry is wrapped in an OpenAI-style response; a dict entry
    is returned as-is (e.g. an Ollama-style ``{"message": {...}}``).
    """

    def __init__(self, contents):
        self.contents = list(contents)
        self.bodies = []

    async def post(self, *a, **kw):
        self.bodies.append(kw.get("json"))
        item = self.contents.pop(0)
        if isinstance(item, dict):
            payload = item
        else:
            payload = {"choices": [{"message": {"content": item}}]}

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        return MockResponse()


@pytest.fixture
def mock_llm(monkeypatch):
    import httpx

    mocks = []

    def install(contents):
        mock = _MockResponses(contents)
        mocks.append(mock)
        monkeypatch.setattr(httpx.AsyncClient, "post", mock.post)
        return mock

    yield install


class TestLLMStructuredOutput:
    @pytest.mark.asyncio
    async def test_valid_schema_returns_dict(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock = mock_llm(['{"name": "Иван", "age": 30}'])
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "person",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
                "provider": "openai",
            }
        )
        result = await node.execute(ExecContext(state={}, tools={}), {})
        assert result["person"] == {"name": "Иван", "age": 30}
        assert mock.bodies[0]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_retries_on_invalid_then_succeeds(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock = mock_llm(
            [
                '{"name": 123}',  # invalid: name must be string
                '{"name": "Иван", "age": 30}',  # valid
            ]
        )
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "person",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
                "max_retries": 2,
                "provider": "openai",
            }
        )
        result = await node.execute(ExecContext(state={}, tools={}), {})
        assert result["person"]["name"] == "Иван"
        assert len(mock.bodies) == 2
        feedback = mock.bodies[1]["messages"][-1]["content"]
        assert "failed JSON schema validation" in feedback

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext, StructuredOutputError

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock = mock_llm(['{"name": 123}'] * 3)
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "person",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "max_retries": 2,
                "provider": "openai",
            }
        )
        with pytest.raises(StructuredOutputError) as exc_info:
            await node.execute(ExecContext(state={}, tools={}), {})
        assert exc_info.value.attempts == 3
        assert exc_info.value.errors
        assert len(mock.bodies) == 3

    @pytest.mark.asyncio
    async def test_output_type_typeddict(self, mock_llm, monkeypatch):
        from typing import TypedDict

        from draf.node import LLM, ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock = mock_llm(['{"city": "Москва", "temp": 22.5}'])

        class Weather(TypedDict):
            city: str
            temp: float

        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "weather",
                "output_type": Weather,
                "provider": "openai",
            }
        )
        result = await node.execute(ExecContext(state={}, tools={}), {})
        assert result["weather"] == {"city": "Москва", "temp": 22.5}
        assert mock.bodies[0]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_ollama_uses_format_json(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext

        monkeypatch.setenv("OLLAMA_API_KEY", "")
        mock = mock_llm([{"message": {"content": '{"ok": true}'}}])
        node = LLM(
            {
                "model": "llama3.1:8b",
                "provider": "ollama",
                "output_key": "out",
                "json_schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                },
            }
        )
        await node.execute(ExecContext(state={}, tools={}), {})
        assert mock.bodies[0]["format"] == "json"
        assert "response_format" not in mock.bodies[0]

    @pytest.mark.asyncio
    async def test_parse_without_schema(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_llm(['here is the json: {"a": [1, 2]}'])
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "parsed",
                "parse": True,
                "provider": "openai",
            }
        )
        result = await node.execute(ExecContext(state={}, tools={}), {})
        assert result["parsed"] == {"a": [1, 2]}

    @pytest.mark.asyncio
    async def test_parse_failure_raises(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext, StructuredOutputError

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_llm(["not json"])
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "parsed",
                "parse": True,
                "provider": "openai",
            }
        )
        with pytest.raises(StructuredOutputError):
            await node.execute(ExecContext(state={}, tools={}), {})

    @pytest.mark.asyncio
    async def test_tracer_records_structured_events(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext
        from draf.trace import RunTracer

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_llm(
            [
                '{"name": 123}',
                '{"name": "ok"}',
            ]
        )
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "person",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "max_retries": 2,
                "provider": "openai",
            }
        )
        tracer = RunTracer()
        ctx = ExecContext(state={}, tools={}, tracer=tracer)
        await node.execute(ctx, {})
        structured = [ev for ev in tracer.events if ev.kind == "structured"]
        assert len(structured) == 1
        assert structured[0].data["attempt"] == 1
        assert "expected string" in structured[0].data["errors"]

    @pytest.mark.asyncio
    async def test_stream_emits_structured_event(self, mock_llm, monkeypatch):
        from draf.node import LLM, ExecContext
        from draf.stream import StreamEvent

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_llm(['{"name": 123}', '{"name": "ok"}'])
        node = LLM(
            {
                "model": "gpt-4",
                "output_key": "person",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "max_retries": 2,
                "provider": "openai",
            }
        )
        events = []

        async def emit(event: StreamEvent) -> None:
            events.append(event)

        ctx = ExecContext(state={}, tools={}, emit=emit)
        await node.execute(ctx, {})
        structured = [ev for ev in events if ev.type == "structured"]
        assert len(structured) == 1
        assert structured[0].data["attempt"] == 1
