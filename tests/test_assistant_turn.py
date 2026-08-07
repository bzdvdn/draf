"""Tests for the abstract ``Assistant.run`` / ``pending`` / ``stream`` API.

Builds the ``repair-ai-chat`` graph (a two-approval human-in-the-loop flow)
and drives it through the unified ``Assistant.run`` loop, proving that
``GraphInterrupt`` pauses are folded into a ``TurnResult`` (``waiting``)
instead of leaking to the caller, and that ``Assistant`` resumes the run
transparently from the durable checkpoint.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "applications" / "repair-ai-chat"
)
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from src.graphs.build import build_flow  # noqa: E402
from src.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402
from src.storage import TRANSIENT_KEYS, build_checkpointer  # noqa: E402

from teff import Assistant, TurnResult  # noqa: E402


def _reply(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "message": {"role": "assistant", "content": content},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _stream_lines(content: str) -> list[str]:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]})
        for chunk in (content[i : i + 4] for i in range(0, len(content), 4))
    ]
    lines.append("data: [DONE]")
    return lines


def _tool_call(name: str, call_id: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ],
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


_PLAN = "1. Демонтаж.\n2. Стены.\n3. Пол.\n4. Отделка."
_ESTIMATE = "Смета: стены 12000, пол 20000, отделка 30000."
_FINAL = f"План: {_PLAN}\n\nСмета: стены 12000, пол 20000, отделка 30000."
_MATERIALS = "Плитка Керама-Белый 780 ₽/м², ламинат Дуб-Прованс 890 ₽/м²."
_HAPPY_PATH = ["extract", "plan", "ask_plan", "estimate", "qa", "ask_estimate"]


class _MockTransport:
    """Serves a canned, system-prompt-aware reply per LLM call.

    Installed on both ``httpx.AsyncClient.post`` and ``.stream``.  Dispatches
    on the system-prompt text: the coordinator gets a scripted sequence of
    tool calls (``coordinator_steps``), while each sub-agent runs its canned
    content.
    """

    def __init__(self, *, coordinator_steps: list[str] | None = None):
        self.calls: list[str] = []
        self.coordinator_calls = 0
        self.coordinator_steps = list(
            coordinator_steps if coordinator_steps is not None else _HAPPY_PATH
        )
        self._call_seq = 0
        self.last_tool: str | None = None

    def _with_tool(self, name: str, call_id: str, arguments: dict) -> dict:
        self.last_tool = name
        return _tool_call(name, call_id, arguments)

    def _coordinator_reply(self, messages: list[dict]) -> dict:
        self.coordinator_calls += 1
        self._call_seq += 1
        call_id = f"coord{self._call_seq}"
        # A resume returns the coordinator to a pending ask_human: the last
        # message is an assistant tool-call with no matching tool reply, so
        # re-emit the same ask for the executor to answer.
        last = messages[-1] if messages else None
        if last and last.get("role") == "assistant" and last.get("tool_calls"):
            fn = last["tool_calls"][0]["function"]
            return self._with_tool(
                fn["name"],
                call_id,
                json.loads(fn["arguments"]),
            )
        if not self.coordinator_steps:
            ran_tools = any(
                m.get("role") == "tool"
                or (m.get("role") == "assistant" and m.get("tool_calls"))
                for m in messages
            )
            if ran_tools:
                if self.last_tool == "select_materials":
                    return _reply(_MATERIALS)
                return _reply(_FINAL)
            return _reply("Здравствуйте! Помогу спланировать ремонт.")
        token = self.coordinator_steps.pop(0)
        if token == "extract":
            return self._with_tool("extract_project_info", call_id, {})
        if token == "plan":
            return self._with_tool("propose_plan", call_id, {})
        if token == "ask_plan":
            return self._with_tool(
                "ask_human",
                call_id,
                {"question": "План готов. Одобряешь план? Ответь: да или нет."},
            )
        if token == "materials":
            return self._with_tool("select_materials", call_id, {})
        if token == "estimate":
            return self._with_tool("prepare_estimate", call_id, {})
        if token == "qa":
            return self._with_tool("run_qa_check", call_id, {})
        if token == "ask_estimate":
            return self._with_tool(
                "ask_human",
                call_id,
                {"question": "Смета готова. Одобряешь смету? Ответь: да или нет."},
            )
        return _reply(_FINAL)

    def _content_for(self, body: dict) -> dict:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        self.calls.append(system[:40])
        if "Координатор" in system:
            return self._coordinator_reply(body.get("messages", []) or [])
        if "извлекаешь" in system:
            return _reply(json.dumps({"room_type": "bathroom", "area": 5.0}))
        if "Planner" in system:
            return _reply(_PLAN)
        if "Estimator" in system:
            return _reply(_ESTIMATE)
        if "Materials Agent" in system:
            return _reply("Плитка Керама-Белый 780 ₽/м², ламинат Дуб-Прованс 890 ₽/м².")
        if "QA Agent" in system:
            return _reply(json.dumps({"ok": True, "message": ""}))
        return _reply("Здравствуйте! Помогу спланировать ремонт.")

    def __call__(self, *args, **kwargs):
        data = self._content_for(kwargs.get("json") or {})

        if args and args[0] == "POST":  # streaming path

            class _StreamResp:
                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    content = (data.get("message") or {}).get("content", "")
                    for line in _stream_lines(content):
                        yield line

            class _StreamCM:
                async def __aenter__(self):
                    return _StreamResp()

                async def __aexit__(self, *exc):
                    return False

            return _StreamCM()

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return data

        async def _post():
            return _Resp()

        return _post()


@pytest.fixture
def transport(monkeypatch):
    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)
    return mock


def _build_assistant(checkpoint_dir: str) -> Assistant:
    flow, tools = build_flow()
    return Assistant(
        flow.compile(),
        tools,
        build_checkpointer(checkpoint_dir),
        reducers=STATE_REDUCERS,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )


@pytest.mark.asyncio
async def test_turn_loop_handles_interrupts(transport, tmp_path):
    """turn() folds each pause into waiting=True and resumes transparently."""
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-turn"

    first = await assistant.run(sid, "Спланируй ремонт ванной 5 м².")
    assert isinstance(first, TurnResult)
    assert first.waiting is True
    assert first.prompt  # the plan-approval question
    assert first.session_id == sid
    assert await assistant.pending(sid) is not None  # durable pause is visible

    # "да" approves the plan; the run continues to the estimate-approval pause
    second = await assistant.run(sid, "да")
    assert second.waiting is True
    assert "Смета" in (second.prompt or "")

    # "да" approves the estimate; the pipeline completes
    done = await assistant.run(sid, "да")
    assert done.waiting is False
    assert "Смета:" in done.reply
    assert done.key is None
    # after completion there is no longer a pending interrupt
    assert await assistant.pending(sid) is None


@pytest.mark.asyncio
async def test_turn_fresh_session_and_durable_history(transport, tmp_path):
    """A fresh session (no pending pause) starts from the entry point."""
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-fresh"
    assert await assistant.pending(sid) is None
    result = await assistant.run(sid, "Привет")
    # no interrupt raised at the entry router, so the turn runs to the end
    assert isinstance(result, TurnResult)


@pytest.mark.asyncio
async def test_stream_yields_interrupt_and_ends(transport, tmp_path):
    """stream() surfaces a terminal interrupt event instead of raising."""
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-stream"

    types = []
    async for ev in assistant.stream(sid, "Спланируй ремонт ванной 5 м²."):
        types.append(ev.type)
        if ev.type == "interrupt":
            break

    assert "run_start" in types
    assert types[-1] == "interrupt"
    # the durable summary counts the plan-approval pause already present
    assert await assistant.pending(sid) is not None


@pytest.mark.asyncio
async def test_materials_question_routes_to_select_materials(transport, tmp_path):
    """A pure materials question hits select_materials, not the full flow."""
    transport.coordinator_steps = ["materials"]
    assistant = _build_assistant(str(tmp_path))
    sid = "sess-materials"
    result = await assistant.run(sid, "Какую плитку лучше взять и какая есть сейчас?")
    assert result.waiting is False
    assert transport.last_tool == "select_materials"
    assert "Плитка Керама-Белый" in result.reply
    assert not any("Planner" in c for c in transport.calls)
    assert not any("Estimator" in c for c in transport.calls)


def _build_graph(checkpoint_dir: str):
    """Compile the graph, discarding the Assistant wrapper."""
    flow, tools = build_flow()
    return flow.compile(), tools, build_checkpointer(checkpoint_dir)


@pytest.mark.asyncio
async def test_graph_run_message_is_interrupt_aware(transport, tmp_path):
    """graph.run(message=...) folds pauses and resumes from the checkpoint."""
    graph, tools, checkpointer = _build_graph(str(tmp_path))
    sid = "sess-graph-run"

    first = await graph.run(
        {},
        message="Спланируй ремонт ванной 5 м².",
        tools=tools,
        reducers=STATE_REDUCERS,
        checkpointer=checkpointer,
        checkpoint_id=sid,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )
    assert isinstance(first, TurnResult)
    assert first.waiting is True
    assert first.prompt

    second = await graph.run(
        {},
        message="да",
        tools=tools,
        reducers=STATE_REDUCERS,
        checkpointer=checkpointer,
        checkpoint_id=sid,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )
    assert second.waiting is True
    assert "Смета" in (second.prompt or "")

    done = await graph.run(
        {},
        message="да",
        tools=tools,
        reducers=STATE_REDUCERS,
        checkpointer=checkpointer,
        checkpoint_id=sid,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )
    assert done.waiting is False
    assert "Смета:" in done.reply


@pytest.mark.asyncio
async def test_graph_run_message_requires_checkpointer(transport):
    """run(message=...) without a checkpointer is a clear error."""
    graph, _, _ = _build_graph(str(""))
    with pytest.raises(ValueError, match="requires checkpointer and checkpoint_id"):
        await graph.run({}, message="Привет")


@pytest.mark.asyncio
async def test_graph_stream_message_auto_resumes(transport, tmp_path):
    """stream(message=...) resumes a paused session from durable state."""
    graph, tools, checkpointer = _build_graph(str(tmp_path))
    sid = "sess-graph-stream"

    async def _turn(message: str) -> list[str]:
        types = []
        async for ev in graph.stream(
            {},
            message=message,
            tools=tools,
            reducers=STATE_REDUCERS,
            checkpointer=checkpointer,
            checkpoint_id=sid,
            initial_state=initial_state,
            transient_keys=TRANSIENT_KEYS,
        ):
            types.append(ev.type)
            if ev.type == "interrupt":
                break
        return types

    first = await _turn("Спланируй ремонт ванной 5 м².")
    assert first[-1] == "interrupt"
    assert await graph.pending(sid, checkpointer=checkpointer) is not None

    resumed = await _turn("да")
    assert resumed[-1] == "interrupt"  # now paused on the estimate approval
    assert "Смета" in str(
        (await graph.pending(sid, checkpointer=checkpointer) or {}).get("prompt")
    )
