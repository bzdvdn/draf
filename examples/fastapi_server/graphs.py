"""Code-first graphs exposed by the FastAPI server.

Every graph is defined in Python with the Flow builder (no workflow.yaml),
then registered by name.  The server picks a graph by name per request and
executes it with a shared checkpointer scoped to the caller's user id.

Requires Ollama running locally with llama3.1:8b:

    ollama pull llama3.1:8b
"""

from __future__ import annotations

from draf.flow import Flow
from draf.graph import Graph
from draf.provider import ProviderRegistry
from draf.tool import Tool

DEFAULT_MODEL = "llama3.1:8b"

CHAT_SYSTEM = (
    "You are the DRAFTFLOW demo assistant. Answer concisely, in the user's language."
)

CALCULATOR_SYSTEM = (
    "You are a helpful assistant with calculator access. "
    "Use the calculator tool whenever you need to compute something."
)

SUMMARIZE_SYSTEM = (
    "You are a precise summarizer. "
    "Return only the summary, 2-3 sentences, in the input's language."
)


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate a mathematical expression and return the numeric result."

    def run(self, expression: str = "") -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as exc:  # noqa: BLE001 - surface any eval error to the model
            return f"Error: {exc}"


def build_chat() -> Graph:
    """ReAct chat agent with message history.

    No tools, so the final text streams token-by-token (SSE ``token``
    events).  Conversation accumulates in the ``messages`` state key,
    which is durable via the checkpointer.
    """
    flow = Flow(
        "chat",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.react(
        model=DEFAULT_MODEL,
        system=CHAT_SYSTEM,
        input_key="query",
        output_key="answer",
        messages_key="messages",
        stream=True,
    )
    return flow.compile()


def build_calculator() -> Graph:
    """ReAct agent with a calculator tool.

    Tool calls and results surface as ``node_start``/``node_end`` and
    ``llm`` stream events as the ``agent -> tool_exec`` cycle runs.
    """
    flow = Flow(
        "calculator",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.react(
        model=DEFAULT_MODEL,
        system=CALCULATOR_SYSTEM,
        input_key="query",
        output_key="answer",
        messages_key="messages",
        stream=True,
    )
    return flow.compile()


def build_summarize() -> Graph:
    """Single-shot LLM transform: ``{text}`` -> ``{summary}``."""
    flow = Flow(
        "summarize",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.llm(
        model=DEFAULT_MODEL,
        system=SUMMARIZE_SYSTEM,
        prompt="Summarize the following text:\n\n{text}",
        output_key="summary",
    )
    return flow.compile()


#: Registry consumed by app.py: ``name -> compiled Graph``.
GRAPHS: dict[str, Graph] = {
    "chat": build_chat(),
    "calculator": build_calculator(),
    "summarize": build_summarize(),
}

#: Human-readable metadata surfaced by ``GET /api/graphs``.
GRAPH_META = {
    "chat": {
        "description": (
            "ReAct chat agent (llama3.1:8b), multi-turn via state key "
            "'messages', streams tokens over SSE."
        ),
        "messages_key": "messages",
        "tools": [],
    },
    "calculator": {
        "description": (
            "ReAct agent with a calculator tool. Same multi-turn state "
            "as 'chat'; tool rounds visible in the event stream."
        ),
        "messages_key": "messages",
        "tools": ["calculator"],
    },
    "summarize": {
        "description": "Summarize a text. Input key 'text', output key 'summary'.",
        "messages_key": None,
        "tools": [],
    },
}

#: Tools available to graphs that declare them in ``GRAPH_META``.
TOOLS: dict[str, Tool] = {t.name: t for t in [Calculator()]}

DEFAULT_GRAPH = "chat"
