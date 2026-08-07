"""Tests for the channel adapters: HTTP/SSE, generic webhooks, Telegram.

All transports bind the same :class:`~teff.assistant.Assistant` built from a
``workflow.yaml`` and run against a mocked LLM transport (monkeypatched
``httpx.AsyncClient.post``) — no network, no API keys.  Covers:

* ``build_assistant``: a durable, interrupt-aware service from YAML.
* ``WebhookChannel``: schema validation, session derivation, one-turn run.
* ``HTTPChannel``: ``/api/chat`` single-shot + durable runs + interrupts.
* ``TelegramChannel``: update handling, session mapping, interrupt resume.
"""

import json
import textwrap
from pathlib import Path

import httpx
import pytest

from teff.channels import (
    HTTPChannel,
    TelegramChannel,
    WebhookChannel,
    build_assistant,
    reply_text,
    turn_response,
)


def _mock_llm(monkeypatch, reply_by_text=None):
    """Mock ``httpx.AsyncClient.post`` for the Ollama provider.

    The harness calls ``post`` with headers both positionally and as a
    keyword, so the stub must accept ``*args``/``**kwargs``.  Ollama's
    payload carries ``chat_path``/``model`` and the messages; we dispatch on
    the joined system-prompt text and return ``{"message": {...}}``.
    """
    reply_by_text = reply_by_text or {}
    calls: list[str] = []

    async def mock_post(self, *args, **kwargs):
        payload = kwargs.get("json") or {}
        calls.append(payload)
        system = ""
        messages = payload.get("messages") or []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
        text = None
        for needle, answer in reply_by_text.items():
            if needle in system:
                text = answer
                break
        if text is None:
            text = "default reply"
        return _resp({"message": {"role": "assistant", "content": text}, "done": True})

    def _stream_resp(text: str):
        class StreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield json.dumps(
                    {"message": {"role": "assistant", "content": text}, "done": True}
                )

        class StreamContext:
            async def __aenter__(self):
                return StreamResponse()

            async def __aexit__(self, *exc):
                return False

        return StreamContext()

    def mock_stream(self, *args, **kwargs):
        payload = kwargs.get("json") or {}
        system = ""
        for m in payload.get("messages") or []:
            if m.get("role") == "system":
                system = m["content"]
        text = None
        for needle, answer in reply_by_text.items():
            if needle in system:
                text = answer
                break
        return _stream_resp(text or "default reply")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
    return calls


def _resp(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    p = tmp_path / "chat.yaml"
    p.write_text(
        textwrap.dedent(
            f"""
            name: chat_demo
            state:
              schema:
                messages: {{reducer: append, type: list}}
              initial:
                messages: []
            providers:
              - name: ollama
                type: ollama
                base_url: http://localhost:11434
                chat_path: /api/chat
            checkpoint:
              type: file
              path: {tmp_path / "cp"}
            steps:
              - id: reply
                type: llm_chat
                config:
                  system: "Reply with one word."
                  model: llama3.1:8b
                  provider: ollama
                  output_key: answer
                  messages_key: messages
            edges: []
            """
        ),
        encoding="utf-8",
    )
    return p


class TestBuildAssistant:
    def test_compiles_workflow_with_checkpoint(self, workflow):
        assistant = build_assistant(str(workflow))
        assert assistant.graph is not None
        assert assistant.checkpointer is not None

    async def test_run_durable_turn(self, workflow, monkeypatch):
        _mock_llm(monkeypatch, {"Reply with one word.": "hello"})
        assistant = build_assistant(str(workflow))
        result = await assistant.run("s1", "hi there")
        assert reply_text(result) == "hello"
        assert not result.waiting
        # the session is durable: a second turn continues the same history
        result2 = await assistant.run("s1", "again")
        assert reply_text(result2) == "hello"

    async def test_turn_response_shape(self, workflow, monkeypatch):
        _mock_llm(monkeypatch, {"Reply with one word.": "hi"})
        assistant = build_assistant(str(workflow))
        result = await assistant.run("s1", "hello")
        payload = turn_response(result, "s1")
        assert payload == {"session_id": "s1", "waiting": False, "message": "hi"}


class TestWebhookChannel:
    def _hook(self, assistant):
        return WebhookChannel(
            assistant,
            {
                "path": "/hooks/x",
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "input": {"message": "summarize: {text}"},
                "session_key": "text",
            },
        )

    async def test_handle_runs_turn(self, workflow, monkeypatch):
        _mock_llm(monkeypatch, {"Reply with one word.": "summary"})
        hook = self._hook(build_assistant(str(workflow)))
        out = await hook.handle({"text": "the quick fox"})
        assert out["ok"] is True
        assert out["message"] == "summary"
        assert not out["waiting"]
        assert out["session_id"] == "the quick fox"

    async def test_handle_validation_errors(self, workflow, monkeypatch):
        _mock_llm(monkeypatch)
        hook = self._hook(build_assistant(str(workflow)))
        out = await hook.handle({"other": 1})
        assert out["ok"] is False
        assert out["errors"]

    async def test_handle_uses_header_owner(self, workflow, monkeypatch):
        _mock_llm(monkeypatch, {"Reply with one word.": "owned"})
        hook = WebhookChannel(
            build_assistant(str(workflow)),
            {
                "input": {"message": "{text}"},
                "owner": "header.X-User-Id",
            },
        )
        out = await hook.handle({"text": "x"}, headers={"X-User-Id": "alice"})
        assert out["ok"] is True
        assert out["message"] == "owned"
        # same payload from a different owner resolves to the same session id,
        # but checkpoints are isolated per owner by the Assistant
        assert out["session_id"] == hook.session_id_for({"text": "x"})

    async def test_session_id_fallback_content_hash(self, workflow, monkeypatch):
        _mock_llm(monkeypatch)
        hook = WebhookChannel(
            build_assistant(str(workflow)),
            {"input": {"message": "{text}"}},
        )
        sid = hook.session_id_for({"text": "same"})
        assert sid == hook.session_id_for({"text": "same"})
        assert sid != hook.session_id_for({"text": "other"})
        assert sid.startswith("wh-")

    async def test_owner_resolution_specs(self, workflow, monkeypatch):
        _mock_llm(monkeypatch)

        def hook(owner_spec):
            return WebhookChannel(
                build_assistant(str(workflow)),
                {"input": {"message": "{text}"}, "owner": owner_spec},
            )

        assert (
            hook("payload.customer").owner_for({"customer": "alice"}) == "alice"
        )
        assert (
            hook("header.X-User-Id").owner_for(
                {"x": 1}, {"X-User-Id": "bob"}
            )
            == "bob"
        )
        # header lookup is case-insensitive
        assert (
            hook("header.X-User-Id").owner_for({}, {"x-user-id": "carol"})
            == "carol"
        )
        assert hook("fixed:ops").owner_for({}) == "ops"
        assert hook("default").owner_for({}) == "default"
        # missing payload/header fields fall back to "default"
        assert hook("payload.customer").owner_for({}) == "default"
        assert hook("header.X-User-Id").owner_for({}, {}) == "default"


class TestHTTPChannel:
    def test_create_app_and_chat(self, workflow, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        _mock_llm(monkeypatch, {"Reply with one word.": "ok"})
        channel = HTTPChannel(build_assistant(str(workflow)))
        client = TestClient(channel.app)

        r = client.post("/api/chat", json={"message": "hello"})
        assert r.status_code == 200
        body = r.json()
        assert body["waiting"] is False
        assert body["message"] == "ok"
        assert body["session_id"]

    def test_stream_ends_with_message(self, workflow, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        _mock_llm(monkeypatch, {"Reply with one word.": "streamed"})
        channel = HTTPChannel(build_assistant(str(workflow)))
        client = TestClient(channel.app)

        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as resp:
            events = []
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
        assert events
        assert events[-1]["message"] == "streamed"

    async def test_interrupt_surfaces_waiting(self, workflow, monkeypatch, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        # a workflow with an interrupt node
        p = tmp_path / "gate.yaml"
        p.write_text(
            textwrap.dedent(
                f"""
                name: gate_demo
                state:
                  schema:
                    messages: {{reducer: append, type: list}}
                  initial:
                    messages: []
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                checkpoint:
                  type: file
                  path: {tmp_path / "cpg"}
                steps:
                  - id: ask
                    type: interrupt
                    config:
                      prompt: "Approve?"
                      key: approved
                edges: []
                """
            ),
            encoding="utf-8",
        )
        _mock_llm(monkeypatch, {"Reply with one word.": "unused"})
        channel = HTTPChannel(build_assistant(str(p)))
        client = TestClient(channel.app)

        r = client.post("/api/chat", json={"message": "do it"})
        assert r.status_code == 200
        body = r.json()
        assert body["waiting"] is True
        assert body["key"] == "approved"
        assert body["message"] == "Approve?"
        # resume with the operator's answer
        r2 = client.post("/api/chat", json={"message": "yes", "session_id": body["session_id"]})
        assert r2.status_code == 200


class TestTelegramChannel:
    def _bot(self, assistant):
        return TelegramChannel(assistant, token="test:token")

    async def test_handle_update_runs_turn(self, workflow, monkeypatch):
        sent = []
        _mock_llm(monkeypatch, {"Reply with one word.": "hi!"})

        async def fake_api(self, method, **params):
            sent.append((method, params))
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update(
            {"update_id": 1, "message": {"chat": {"id": 42}, "text": "hello"}}
        )
        assert sent, "expected a sendMessage call"
        assert sent[0][0] == "sendMessage"
        assert sent[0][1]["chat_id"] == 42
        assert sent[0][1]["text"] == "hi!"

    async def test_handle_update_session_mapping(self, workflow, monkeypatch):
        sent = []
        _mock_llm(monkeypatch, {"Reply with one word.": "ok"})

        async def fake_api(self, method, **params):
            sent.append((method, params))
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update(
            {"update_id": 1, "message": {"chat": {"id": 99}, "text": "a"}}
        )
        assert bot.session_id_for(99) == "tg-99"
        assert sent[0][1]["text"] == "ok"

    async def test_handle_update_owner_is_user_id(self, workflow, monkeypatch):
        from teff.assistant import Assistant

        async def fake_api_ok(self, method, **params):
            return {}

        seen = {}

        async def fake_run(self, session_id, text, **kwargs):
            seen["owner"] = kwargs.get("owner")
            from teff.graph import TurnResult

            return TurnResult(session_id=session_id, reply="hi")

        _mock_llm(monkeypatch, {"Reply with one word.": "hi"})
        monkeypatch.setattr(Assistant, "run", fake_run)
        monkeypatch.setattr(TelegramChannel, "_api", fake_api_ok)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 7},
                    "from": {"id": 12345},
                    "text": "hello",
                },
            }
        )
        assert seen["owner"] == "12345"

    async def test_ignores_non_text_updates(self, workflow, monkeypatch):
        _mock_llm(monkeypatch)
        sent = []

        async def fake_api(self, method, **params):
            sent.append((method, params))
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update({"update_id": 1, "message": {"chat": {"id": 1}}})
        assert sent == []
