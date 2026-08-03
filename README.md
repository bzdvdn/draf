# Draf

**Workflow as data. Agents as graphs.**

Draf is a Python framework for building durable AI agents and workflows —
an embeddable async library. Inspired by LangGraph and LangChain, it brings
graph-based, stateful agents to Python with minimal dependencies and zero
runtime magic.

- **Async by default** — nodes, tools, and LLM calls are all `async`.
- **Workflow as data** — the canonical graph is YAML/JSON; code is optional.
- **Durable** — checkpoint/resume across file, SQLite, and PostgreSQL backends.
- **Embeddable** — a library, not a platform. You import us; we never import you.

Full documentation (guide, node/tool/provider references, and an auto-generated
API reference) lives in [`docs/`](docs/). Build it locally with
`uv run pip install -e ".[docs]"` then `uv run mkdocs serve`.

## Install

```bash
pip install draf
# extras: draf[embedding] for RAG stores, draf[pg-checkpoint] for PostgreSQL
# checkpoints, draf[mcp] for MCP server tools, draf[tools] for built-in tools
# (web fetch, PDF, S3, Slack, SQL, email, Telegram, …), draf[all] for everything
```

Python >= 3.11. Core runtime depends only on `httpx`, `pyyaml`, and `typer`.

The `draf` CLI ships with the package. Prefer uv? The same command works — uv
installs the package **and** the CLI in one go:

```bash
uv tool install draf         # global `draf` CLI
uvx draf -f workflow.yaml    # run on the fly without installing anything
```

## Quick start

### YAML workflow

```yaml
name: text-pipeline
state:
  initial:
    title: "  hello world  "

steps:
  - id: trim
    type: transform
    config: { action: trim, input_key: title, output_key: trimmed }
  - id: uppercase
    type: transform
    config: { action: uppercase, input_key: trimmed, output_key: loud }
  - id: count
    type: transform
    config: { action: count_lines, input_key: loud, output_key: line_count }

edges:
  - from: trim
    to: uppercase
  - from: uppercase
    to: count
```

```python
import asyncio
from draf.yaml import load_workflow


async def main():
    graph, tools, state, reducers = load_workflow("workflow.yaml")
    result = await graph.run(state, tools=tools, reducers=reducers)
    print(result)


asyncio.run(main())
```

### Flow API (Python)

```python
import asyncio
from draf import set_defaults
from draf.flow import Flow, Case
from draf.node import LLM, Transform

set_defaults(provider="ollama")


async def main():
    flow = Flow("sentiment-router")
    flow.step(
        LLM(
            model="llama3.1:8b",
            system='Classify the sentiment. Reply "positive" or "negative".',
            input_key="text",
            output_key="sentiment",
        )
    )
    flow.branch(
        "sentiment",
        Case("positive").add(
            Transform(action="value", value="Glad you liked it!", output_key="reply")
        ),
        Case("negative").add(
            Transform(action="value", value="Sorry to hear that.", output_key="reply")
        ),
    ).converge(Transform(action="uppercase", input_key="reply", output_key="result"))

    result = await flow.compile().run(state={"text": "I love this product!"})
    print(result)


asyncio.run(main())
```

## Core concepts

- **State** — a flat, JSON-serializable dict. Nodes transform state, nothing else.
- **Nodes** — pure `async def fn(ctx, state) -> dict` functions (or built-ins:
  `LLM`, `Transform`, `ReActAgent`, `ToolExec`).
- **Graph** — nodes + edges, including conditional edges, branches, and
  `__error__` fallbacks. The graph owns routing and resilience.
- **Tools** — implement `Tool` or use the `@tool` decorator; shareable across nodes.
  A set of built-in tools ships in `draf[tools]` (see [Built-in tools](#built-in-tools)).
- **RAG** — `RAGTool` over pluggable vector stores (`InMemoryVectorStore`,
  `SQLite`, `Chroma`, `Qdrant`, `PGVector`).

## Durable execution (checkpoints)

`Graph.run()` accepts a `checkpointer` and a `checkpoint_id`. A checkpoint is
written **before** every node, so a crash or error resumes from the last safe
point instead of starting over.

```python
from draf import Graph
from draf.checkpoint import SQLiteCheckpointer
from draf.node import Transform

nodes = {"shout": Transform(action="uppercase", input_key="text", output_key="loud")}
graph = Graph(nodes, edges=[], entry_point="shout")
cp = SQLiteCheckpointer("checkpoints.db")

# first run crashes at some node
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")

# same id resumes from the saved checkpoint and completes
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")
```

Backends: `JSONFileCheckpointer` (core), `SQLiteCheckpointer` (core),
`PGCheckpointer` (`draf[pg-checkpoint]`, needs PostgreSQL). On resume the saved
state wins over the passed-in state; a `State` instance keeps its schema and
reducers.

### Multi-tenant checkpoints (owner scoping)

Pass `owner=` to scope checkpoints to a user/session/tenant. The same
`checkpoint_id` under different owners never collides, so one store can serve
many users without composite key conventions:

```python
await graph.run(state, checkpointer=cp, checkpoint_id="chat-1", owner="alice")
await graph.run(state, checkpointer=cp, checkpoint_id="chat-1", owner="bob")  # separate

chats = await cp.list("alice")  # ["chat-1", ...] — enumerate a user's runs
```

**Use `owner` for anything that should be isolated per end-user** — a user id,
session id, or tenant. Every owner gets its own namespace: file checkpoints
land in an `owner/` subdirectory, SQLite/PG store a composite
`(owner, checkpoint_id)` primary key (existing single-owner databases migrate
automatically).

When `owner` is omitted, runs fall under the default owner `"default"`
(`draf.checkpoint.DEFAULT_OWNER`) — so single-tenant callers work unchanged,
and you can still list them with `cp.list()`. The CLI exposes the same knob:
`--checkpoint-owner` on `draf run` and `draf inspect` (defaults to `default`).

## Parallel branches

Run independent branch chains concurrently and merge their results with
`Flow.parallel()` — each branch gets an isolated copy of the state, and
per-key reducers merge updates back so `append` branches accumulate instead
of overwriting each other.

```python
from draf.flow import Flow
from draf.node import Transform

flow = (
    Flow("p")
    .parallel(
        [Transform(action="uppercase", input_key="title", output_key="title")],
        [Transform(action="uppercase", input_key="body", output_key="body")],
    )
    .converge(Transform(action="value", value="done", output_key="status"))
)

result = await flow.compile().run(state={"title": "hi", "body": "world"})
# -> title/body uppercased in parallel, then status="done"
```

Branches can be single nodes, lists of nodes (run sequentially inside the
branch), or embedded `Flow` subgraphs. The parallel node also works directly:
`Parallel([[node1], [node2]])`.

For a full end-to-end demo with LLM calls, see
`examples/parallel/rag_report.py` — two RAG searches run in parallel branches,
an LLM merges the summaries into a report file, and a final LLM reviews it
(`VERDICT: pass/fail`). Requires local Ollama.

## Prompt templates

LLM nodes read _multiple_ state keys into one prompt with `{key}` templates
(also supported in `system`)::

```python
node = LLM(
    model="llama3.1:8b",
    system="Ты инженер по ремонту.",
    prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
    output_key="plan",
)
# state {"type": "кухни", "summ": 150000} -> user message:
# "Составь план для ремонта кухни на сумму 150000 рублей."
```

Values are stringified; a placeholder referencing a missing state key raises
`KeyError`. The underlying helper is `draf.prompt.render_template`.

## Dynamic fan-out (Map)

`Flow.map()` fans a state _list_ out into parallel branches at runtime —
branch count is derived from the data, not declared up front. The processor
reads the same keys the Map fans out, so no glue node is needed::

```python
flow = Flow("repair-plans").map(
    LLM(
        model="llama3.1:8b",
        prompt="Составь план для ремонта {type} на сумму {summ} рублей.",
        output_key="plan",
    ),
    input_keys=["type", "summ"],  # lists zipped per index
    output_key="plans",  # list of per-item results
    max_concurrency=2,
)
result = await flow.compile().run(
    state={
        "type": ["кухни", "санузел"],
        "summ": [150000, 80000],
    }
)
# -> {"plans": ["план для кухни...", "план для санузла..."]}
```

`chunk_size` batches items per branch, `max_concurrency` caps simultaneous
branches, and `result_key` overrides which per-item key to collect. Full demo:
`examples/map_repair_plans/` (Python with typed `State`, plus the same
workflow as YAML for `draf -f workflow.yaml`).

## Human-in-the-loop (interrupts)

Pause a workflow for operator input with an `Interrupt` node. When
execution reaches it, `graph.run()` raises `GraphInterrupt`; resume with
the same `checkpoint_id` plus a `resume` dict::

```python
from draf.checkpoint import JSONFileCheckpointer
from draf.node.interrupt import GraphInterrupt
from draf.flow import Flow

flow = Flow("approval")
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.step(
    LLM(model="llama3.1:8b", prompt="{draft}\nВердикт: {approved}", output_key="final")
)

graph = flow.compile()
cp = JSONFileCheckpointer("checkpoints")

try:
    await graph.run(state=state, checkpointer=cp, checkpoint_id="run-1")
except GraphInterrupt as interrupt:
    print(interrupt.prompt)  # "Одобрить? (да / правки)"
    answer = input("> ")
    result = await graph.run(
        state=state,
        checkpointer=cp,
        checkpoint_id="run-1",
        resume={"approved": answer},
    )
```

The answer lands in `state["approved"]` and execution continues past the
interrupt. Interrupts require a `checkpointer`; the resume value for an
already-paused run is requested again if `resume` is missing.

The same pause/resume mechanics gate risky tool calls on an agent: a
`tool_approval` callable that returns `"pause"` raises a `GraphInterrupt`
mid-round, and resuming with `resume={"tool_approval": "approve"}` lets the
approved call through (see [agent_approval](examples/agent_approval/)).

### Revision loop

To re-ask on rejection, wire a cycle with `Flow.loop()` — a conditional
edge on the answer that sends execution back to the `Interrupt` node::

```python
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.loop(
    key="approved",
    until="да",
    done=LLM(model="llama3.1:8b", prompt="{draft}", output_key="final"),
    body=LLM(
        model="llama3.1:8b",
        prompt="Переработай {draft} с учётом: {approved}",
        output_key="draft",
    ),
)
```

`loop()` wires `decider --key=until--> done` (stop) and
`decider --key!=until--> body -> decider` (repeat), so the graph returns
to the same `Interrupt` after each edit. `max_iterations` caps the
rounds. The decider can be any node that writes `key`, not just an
`Interrupt` — `loop()` also works for pure LLM self-check loops.

The same loop is described declaratively in YAML — conditional edges
already express the cycle, and `interrupt` is a registered node type.
See `examples/human_in_loop/workflow.yaml` for the full workflow;
running it still needs a `checkpointer` and a `resume` loop in Python.

## Streaming execution

`graph.stream()` runs the same execution core as `graph.run()` but yields
a `StreamEvent` for every observable step, so callers can render tokens and
progress before the run finishes. Build the graph with the `Flow` API (or
directly with `Graph`) and stream it:

```python
from draf import Flow, LLM

flow = Flow("chat")
flow.step(LLM(model="llama3.1:8b", prompt="Скажи привет", output_key="answer"))
graph = flow.compile()

async for event in graph.stream(state):
    if event.type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.type == "run_end":
        print("\nstatus:", event.data["status"])
```

Event types: `run_start`, `node_start`, `node_end`, `node_error`, `edge`,
`token`, `llm`, `structured`, `interrupt`, `interrupt_resume`,
`checkpoint`, `run_end`.
LLM tokens are emitted as they arrive (any node without tool calls streams
automatically in this mode); routing decisions, checkpoints, and interrupt
pauses are streamed the same way. `stream()` accepts the same parameters as
`run()` — tools, checkpointer, resume, tracer, `max_iterations`.

See [streaming](examples/streaming/) — `run.py` (Flow) and `graph.py`
(low-level `Graph`) — for a full console demo.

## Structured output

Guarantee the LLM returns a schema-conforming JSON object instead of free
text. Pass a JSON Schema (`json_schema`) or a Python type spec
(`output_type` — `TypedDict`, dataclass, or `dict[str, type]`):

```python
from typing import TypedDict
from draf import Flow, LLM


class Weather(TypedDict):
    city: str
    temp: float


flow = Flow("weather")
flow.step(LLM(model="llama3.1:8b", output_key="weather", output_type=Weather))
graph = flow.compile()

result = await graph.run({"city": "Москва"})
result["weather"]  # {"city": "...", "temp": 12.5} — a parsed dict, validated
```

The response is parsed as JSON, validated against the schema, and re-asked
with the validation error fed back (up to `max_retries`, default 2). If
all attempts fail, a `StructuredOutputError` is raised — route it with an
`__error__` edge. Schema errors are recorded as `structured` events in the
tracer and the stream. Without a schema, `parse=True` still parses the
response into a dict (no validation).

The same field map works in YAML:

```yaml
steps:
  - id: weather
    type: llm_chat
    config:
      model: llama3.1:8b
      output_key: weather
      json_schema:
        {
          type: object,
          properties: { city: { type: string }, temp: { type: number } },
          required: [city, temp],
        }
```

## Agents (ReAct loop)

Build a tool-calling agent loop with `flow.react()` — the LLM and tool
executor stay visible as graph topology, so the loop is inspectable and can
be followed by more nodes:

```python
from draf import Flow
from draf.node import Transform
from draf.tool import Tool


class Search(Tool):
    name = "search"
    description = "Search a local index"

    def run(self, query: str = "") -> str:  # type: ignore[override]
        return f"results for {query}"


flow = Flow("agent")
flow.react(
    model="gpt-4", system="Answer using tools.", input_key="query", output_key="answer"
)
flow.step(Transform(action="uppercase", input_key="answer", output_key="result"))

graph = flow.compile()
result = await graph.run({"query": "draf"}, tools=[Search()], max_iterations=10)
```

The agent calls the model; if it requests tools, the executor runs them
**all in parallel** in one round (`asyncio.gather`) and loops back. When
the model answers without a tool call, the response lands at `output_key`
and execution continues with whatever is chained after. `max_iterations`
on `graph.run()` caps the loop.

`flow.harness()` is the full API (an alias, `flow.react()` is kept for
backwards compatibility). It accepts extra knobs:

- `max_tool_rounds` — max model calls per graph visit (default 10).
- `tool_error_mode` — `"message"` (default: a failed tool becomes a `tool`
  message the model can react to) or `"raise"` (the failure propagates, so
  you can route it with a low-level `Graph` `__error__` edge to a fallback).
- `parse_text_tool_calls` — decode tool calls embedded in plain text, for
  local models that skip the structured `tool_calls` field (default True).
- `temperature` / `max_tokens` / `response_format` — sampling knobs.
- `tool_timeout` / `tool_retries` — bound each tool call with a timeout and
  re-attempt it on failure.
- `tool_approval` — gate risky tools. A callable (or `"interactive"`) can
  `approve` / `deny` a call; returning `"pause"` raises a `GraphInterrupt`
  so a human can sign off and resume with `resume={"tool_approval": value}`
  (see [agent_approval](examples/agent_approval/)).
- `max_retries` / `fallbacks` / `retry_on` — retry failed model calls with
  backoff and fail over to backup models (see
  [agent_resilience](examples/agent_resilience/)).
- `max_total_tokens` — stop the agent once the token budget is spent.
- `max_context_tokens` / `trim_messages` — trim the conversation before each
  model call to stay under a context limit.
- `stream` / `on_token` — stream LLM tokens (nodes without tools stream
  automatically in `graph.stream()` mode).

```python
flow.harness(
    model="llama3.1:8b",
    input_key="query",
    output_key="answer",
    max_tool_rounds=5,
    tool_error_mode="raise",
    temperature=0.2,
    tool_timeout=30,
    tool_approval="interactive",
    max_retries=3,
    fallbacks=["llama3.1:8b"],
)
```

## Skills

Bundle instructions **and** a tool scope into a reusable folder following
the open _Agent Skills_ layout — `skills/<name>/SKILL.md` with YAML
frontmatter plus markdown instructions:

```markdown
---
name: city-guide
description: Answer questions about cities
allowed-tools: [city_weather, city_population]
disallowed-tools: [secret_tool]
---

You are a city guide. When asked to compare cities, call BOTH
`city_weather` and `city_population` in the SAME turn.
```

Mount it on any LLM-capable call — the `LLM` node or `react()`/`harness()`:

```python
flow.harness(
    model="llama3.1:8b",
    input_key="query",
    output_key="answer",
    skills=["city-guide"],
    skill_dir="skills",
)

# same for a plain LLM node
flow.step(LLM(model="llama3.1:8b", skills=["city-guide"], use_tools=True))
```

A mounted skill:

- merges its instructions into the system prompt;
- narrows the visible tools: `allowed-tools` intersects with the node's set,
  `disallowed-tools` removes tools outright — so `secret_tool` above stays
  out of the model's reach even though it is registered for the run.

Bare names resolve against `skill_dir`; you can also pass skill paths or
already-loaded `Skill` objects. `use_tools` gives the same per-node scope
without skills: `True` (all), `False` (none), or a list of names.

## Built-in tools

A library of ready-made `Tool` subclasses registers itself when `draf.tool.builtin`
is imported (the `load_workflow` YAML helper and the examples do this for you).
Most are dependency-free; the marked ones need `pip install draf[tools]`.

| Tool              | Name              | Dependencies        | What it does                                          |
| ----------------- | ----------------- | ------------------- | ----------------------------------------------------- |
| CalculatorTool    | `calculator`      | —                   | AST-based safe math evaluation                        |
| ShellTool         | `shell`           | —                   | Run shell commands with a blocklist/whitelist sandbox |
| ReadFileTool      | `read_file`       | —                   | Read a file's contents                                |
| WriteFileTool     | `write_file`      | —                   | Write content to a file                               |
| EditFileTool      | `edit_file`       | —                   | Replace text in a file                                |
| WebSearchTool     | `web_search`      | —                   | DuckDuckGo search, no API key                         |
| WebFetchTool      | `fetch_url`       | `beautifulsoup4`    | Fetch a URL and extract its text                      |
| PDFReadTool       | `read_pdf`        | `pypdf`             | Extract text from a PDF, page by page                 |
| S3Tool            | `s3_list`         | `boto3`             | List objects in an S3 bucket                          |
| S3GetTool         | `s3_get`          | `boto3`             | Download an object from S3                            |
| S3PutTool         | `s3_put`          | `boto3`             | Upload content to S3                                  |
| SlackSendTool     | `slack_send`      | `slack-sdk`         | Send a message to a Slack channel                     |
| SQLQueryTool      | `sql_query`       | sqlite3 / `psycopg` | Read-only SELECT against SQLite or PostgreSQL         |
| SQLListTablesTool | `sql_list_tables` | sqlite3 / `psycopg` | List tables in a database                             |
| SQLDescribeTool   | `sql_describe`    | sqlite3 / `psycopg` | Describe a table's columns and types                  |
| ListDirTool       | `list_dir`        | —                   | List files and directories (optionally recursive)     |
| GlobTool          | `glob`            | —                   | Find files matching a glob pattern                    |
| GetEnvTool        | `getenv`          | —                   | Read an env var (secret values masked)                |
| CurrentTimeTool   | `current_time`    | —                   | Current date/time in an IANA timezone                 |
| JsonParseTool     | `json_parse`      | —                   | Parse and pretty-print JSON                           |
| YamlParseTool     | `yaml_parse`      | —                   | Parse YAML, dump as JSON                              |
| KVStoreTool       | `kv_store`        | —                   | Persistent JSON key-value store                       |
| PythonEvalTool    | `python_eval`     | —                   | Safe AST-whitelist evaluation of Python expressions   |
| HttpRequestTool   | `http_request`    | httpx               | Arbitrary HTTP requests (method, headers, body)       |
| SendEmailTool     | `send_email`      | smtplib             | Send email via SMTP                                   |
| SendTelegramTool  | `send_telegram`   | httpx               | Send a message via a Telegram bot                     |

Tools are plain classes, so you can construct them directly with keyword
arguments — `ShellTool(root_dir=..., allowed_commands=[...])`,
`WebSearchTool(provider="google")`, `SQLQueryTool({"db_type": "sqlite",
"path": "./v.db"})`. The registry (used by YAML `tools:` blocks) maps a config
dict onto the constructor: a dict passed to constructors that take a `config`
dict, or keyword arguments for keyword constructors:

```yaml
tools:
  - type: sql_query
    config: { db_type: sqlite, path: ./vectors.db }
  - type: shell
    config: { root_dir: /tmp, allowed_commands: [echo, ls] }
  - type: s3_list
    config: { bucket: my-bucket, region: eu-central-1, verify: false }
```

```python
from draf.tool.registry import default_tool_registry

sql = default_tool_registry.create("sql_query", {"db_type": "sqlite", "path": "./v.db"})
shell = default_tool_registry.create(
    "shell", {"root_dir": "/tmp", "allowed_commands": ["echo"]}
)
```

Security notes: `shell` enforces a blocklist of dangerous commands plus an
optional whitelist; `getenv` masks values whose names hint at credentials
(`TOKEN`, `API_KEY`, `PASSWORD`, `DSN`, …) unless configured with
`mask_secrets: false`; `sql_query` and the other SQL tools are read-only and
reject `INSERT`/`UPDATE`/`DELETE`/DDL; `python_eval` only allows a whitelisted
AST subset (`math.*`, builtins like `len`/`abs`/`sum`, comparisons).

## MCP tools

Connect any [Model Context Protocol](https://modelcontextprotocol.io) server
and use its tools anywhere built-in tools work — LLM nodes, the ReAct agent,
registries. Tools are fetched from the server (schema included) and wrapped
as ordinary `Tool` instances, so `graph.run(state, tools=tools)` needs no
changes. Install `draf[mcp]`; the `mcp` SDK is imported lazily.

```python
from draf import Flow
from draf.tool import mcp_tools

flow = Flow("agent")
flow.react(model="llama3.1:8b", input_key="query", output_key="answer")
graph = flow.compile()

# stdio server: spawn a subprocess
async with mcp_tools(command=["uvx", "mcp-server-git"]) as tools:
    # or a remote server: mcp_tools(url="http://localhost:8000/mcp")
    result = await graph.run(
        {"query": "What changed in the last commit?"},
        tools=tools,
        max_iterations=10,
    )
```

`command` starts a stdio server (split into argv), `url` connects to a
Streamable HTTP endpoint. The session stays open for the `async with` block.
A runnable pair — a tiny server plus a ReAct agent calling it — lives in
[`examples/mcp/`](examples/mcp/).

## RAG (retrieval)

`RAGTool` chunks, embeds, and retrieves documents from a pluggable vector
store (`InMemoryVectorStore`, `SQLite`, `Chroma`, `Qdrant`, `PGVector`,
`FAISS`, `Lance`, `Milvus`, `Weaviate`, `Pinecone`) using raw HTTP embeddings
(`Embedder`: OpenAI, Ollama, Mistral, Voyage, Jina, Together, Groq, or any
OpenAI-compatible `/v1/embeddings` endpoint). Documents load
from CSV, TXT (glob), PDF (`draf[rag-pdf]`), and Excel (`draf[rag-excel]`).
All vector stores (including the embedded/external ones) are installed via
`draf[embedding]`.

```python
from draf import RAGTool

rag = RAGTool(
    {
        "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
        "store": {"type": "sqlite", "path": "vectors.db", "dim": 768},
        "documents": [
            {"type": "txt", "path": "docs/*.txt"},
            {"type": "csv", "path": "meta.csv", "text_column": "content"},
        ],
        "filter": {"topic": "news"},  # metadata filter (DSL below)
        "similarity_threshold": 0.5,  # drop low-score hits
        "max_tokens": 1024,  # context token budget
        "hybrid": True,  # keyword + semantic blend
    }
)
result = await rag.arun("what changed in v2?")
```

Search arguments override the config per call: `arun(query, k, filter=...,
similarity_threshold=..., max_tokens=..., parent_retrieval=...)`.

- **Metadata filters** — `{"category": "news"}` (equality),
  `{"category": ["news", "tech"]}` (membership, or shared element for list
  fields), and `"$and"` / `"$or"` combinators. Honoured by every store.
- **Hybrid search** — blends a lexical keyword score with the embedding score
  (weight `alpha`, default 0.4) in `InMemoryVectorStore`, `SQLiteVectorStore`,
  and the new embedded/external stores (hybrid + metadata filters are applied
  after retrieval there).
- **Small-to-big** — with `parent_chunks: true` every chunk keeps its full
  parent text; `parent_retrieval: true` returns whole deduplicated parent
  documents instead of individual chunks.
- **Token budget** — `max_tokens` truncates the returned context to an
  approximate token count so a RAG call cannot blow up the LLM context window.

Embedder providers (all OpenAI-compatible `/v1/embeddings`; the `model` key is
optional — a per-provider default is used when omitted):

| `provider` | Default `model`                             | API key env var    |
| ---------- | ------------------------------------------- | ------------------ |
| `openai`   | `text-embedding-ada-002`                    | `OPENAI_API_KEY`   |
| `ollama`   | `nomic-embed-text`                          | — (local)          |
| `mistral`  | `mistral-embed`                             | `MISTRAL_API_KEY`  |
| `voyage`   | `voyage-3`                                  | `VOYAGE_API_KEY`   |
| `jina`     | `jina-embeddings-v3`                        | `JINA_API_KEY`     |
| `together` | `togethercomputer/m2-bert-80M-8k-retrieval` | `TOGETHER_API_KEY` |
| `groq`     | `nomic-embed-text-v1.5`                     | `GROQ_API_KEY`     |

Store types and their config keys:

| `type`              | Config                                                                                  | Notes                                          |
| ------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `in_memory`         | `dim`                                                                                   | default; in-process only                       |
| `sqlite`            | `path`, `dim`                                                                           | stdlib file persistence                        |
| `chroma`            | `path`, `collection`                                                                    | embedded                                       |
| `qdrant`            | `host`, `port`, `collection`                                                            | needs a server                                 |
| `pgvector`          | `dsn`, `table`                                                                          | needs PostgreSQL + pgvector                    |
| `faiss`             | `dim`, `path`                                                                           | FAISS flat index + `.meta.json` sidecar        |
| `lance` / `lancedb` | `path`, `table`, `dim`                                                                  | embedded columnar store                        |
| `milvus`            | `uri`, `token`, `collection`, `dim`                                                     | `uri` can be a local `./file.db` (Milvus Lite) |
| `weaviate`          | `collection`, `embedded`, `host`, `http_port`, `grpc_port`, `api_key`, `headers`, `dim` | `embedded: true` for the in-process server     |
| `pinecone`          | `index_name`, `api_key`, `host`, `namespace`, `dim`                                     | API key from `PINECONE_API_KEY`                |

Vector stores expose management operations beyond search:

```python
await store.count()  # number of vectors
await store.entries(limit=100, offset=0)  # (id, metadata) pairs
await store.get(["chunk_0", "chunk_1"])  # by id
await store.update_metadata("chunk_0", {"starred": True})  # merge
await store.clear()  # wipe everything
```

Runnable examples: [`examples/rag_search/`](examples/rag_search/) (CSV corpus,
in-memory store) and [`examples/rag_stores/`](examples/rag_stores/) (same agent
on every vector store).

## Observability (telemetry)

Pass a `RunTracer` to `graph.run()` to collect a JSON-serialisable event log:
node start/end with latency, edge routing, checkpoints, retries, and LLM token
usage. Fold it into a summary afterwards.

```python
from draf import Graph, RunTracer

tracer = RunTracer()
await graph.run(state, tracer=tracer)

print(tracer.to_json())  # {"summary": {...}, "events": [...]}
print(tracer.summary())  # RunSummary(status, total_ms, nodes, tokens, ...)
```

The CLI exposes the same report: `draf -f workflow.yaml --trace`.

## CLI

`draf` runs YAML workflows, validates them, and reports on runs and evals:

```bash
draf -f workflow.yaml                      # run (the default command)
draf -f workflow.yaml --trace              # run + JSON trace to stderr
draf validate workflow.yaml                # validate without running
draf daemon -f workflow.yaml --once        # run one tick of a poll loop
draf daemon -f workflow.yaml --interval 60 # run forever, 60s between ticks
draf eval workflow.yaml --data dataset.jsonl --exact
draf inspect --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id run-1
draf new support-ai                        # scaffold a FastAPI app (default)
draf new support-cli --template cli        # scaffold a terminal-only app
draf new support-worker --template daemon  # scaffold a background worker
draf new support-chat --template fastapi --with postgres,rag,celery  # + variants
draf version
```

`draf daemon` re-runs a workflow on a poll interval (e.g. a GitLab reviewer),
carrying state between ticks via `--checkpoint '{"type":"file","path":"data/cp"}'`.

`draf new` renders a runnable project from one of three templates — `fastapi`
(a service with API-key auth and durable sessions), `cli` (the same supervisor
graph driven from the terminal), or `daemon` (a worker polling a job queue).
Every generated module carries `HOW TO EXTEND` comments and the project's
tests run offline with no API keys.

Feature **variants** are additive overlays enabled with `--with` (comma-separated,
any subset output by `draf new --help`):

- `postgres` — adds a pgvector `deploy/compose.yaml` + `.env.example`; the DSN
  (`DRAF_DATABASE_URL`) points durable sessions (and RAG vectors) at Postgres.
- `rag` — a document catalog over `data/documents/` with RAG search tools wired
  into the writer agent (embedded lazily on the first search, so tests stay
  offline).
- `celery` — a Celery worker + beat pair that re-embeds the catalog whenever
  the seed documents change (requires the `queue` extra).

Every generated app wires its graph, tools, checkpointer and assistant through a
single composition root — `src/core/container.py:build_container` — so the CLI,
server and worker behave identically.

## Docker

Official images are published to Docker Hub for every `v*` tag. One build,
four variants — pick the one that matches how you deploy:

| Image                    | Contents                        | Runs                                             |
| ------------------------ | ------------------------------- | ------------------------------------------------ |
| `bzdvdn/draf`            | core + `draf[tools]`            | the `draf` CLI — run/validate/inspect workflows  |
| `bzdvdn/draf-fastapi`    | core + `draf[fastapi]`          | `uvicorn` — a FastAPI server app                 |
| `bzdvdn/draf-worker`     | core + `draf[queue]`            | `celery` — background workers / beat             |
| `bzdvdn/draf-all`        | every extra except `docs`       | the `draf` CLI with the full optional surface    |

Run a workflow from a mounted `workflow.yaml` (plus an optional `plugins/`
folder) in one shot — plugins are plain Python files loaded at runtime, so a
container isolates untrusted workflow code:

```bash
docker run --rm -v "$PWD:/workflow" \
  bzdvdn/draf:latest run -f /workflow/workflow.yaml
```

Interactive mode, with durable checkpoints:

```bash
docker run --rm -it \
  -v "$PWD/workflow.yaml:/workflow/workflow.yaml" \
  -v "$PWD/plugins:/workflow/plugins" \
  -v draf-checkpoints:/data/checkpoints \
  bzdvdn/draf:latest run -f /workflow/workflow.yaml --interactive
```

Every image runs as a non-root user (UID 65534) with checkpoints under
`/data/checkpoints`. The `fastapi` and `worker` images are base images for your
own app — extend them with your `main.py`, or mount it and override the
command:

```bash
docker run -p 8000:8000 -v "$PWD/app:/app" \
  bzdvdn/draf-fastapi:latest main:app
docker run -v "$PWD:/app" bzdvdn/draf-worker:latest -A src.celery_app worker
```

All CLI subcommands (`run`, `daemon`, `validate`, `graph`, `eval`, `inspect`,
`new`) and flags work inside the container. Local builds use
`docker buildx bake` — see [`Dockerfile`](Dockerfile) and
[`docker-bake.hcl`](docker-bake.hcl).

## Cost & token reports

`RunSummary` folds the trace into cost and token figures. Costs are
estimated from an internal model-price table (exact name match, then
prefix match; unknown and local models cost $0). Secrets are redacted
from every reported value, so API keys never leak into logs or the CLI.

```python
from draf import RunTracer

tracer = RunTracer()
await graph.run(state, tracer=tracer)

summary = tracer.summary()
print(summary.cost_usd)  # estimated spend in USD
print(summary.tokens)  # prompt/completion totals
summary.to_dict()  # JSON-serialisable (redacted)
summary.to_json()
```

### Custom pricing per provider / model

The built-in table cannot cover aggregators and custom providers (OpenRouter,
Kilo, vLLM, …) that keep their **own** rates and use their **own** model names
(e.g. `openai/gpt-4o`). Register custom USD-per-1M-token pricing at runtime;
it takes precedence over the built-in table:

```python
from draf import set_model_pricing, set_provider_pricing, model_pricing

set_model_pricing("openrouter", "openai/gpt-4o", 3.0, 12.0)
set_provider_pricing("kilo", 0.1, 0.4)  # whole-provider default
print(model_pricing("openai/gpt-4o", "openrouter"))  # (3.0, 12.0)
```

Load a whole file at once — `load_pricing("pricing.yaml")` (or a dict):

```yaml
providers:
  openrouter:
    default: { input: 0.1, output: 0.4 }
    models:
      "openai/gpt-4o": { input: 3.0, output: 12.0 }
```

Resolution order: exact `(provider, model)` → provider-prefixed custom entry →
provider-wide default → built-in table → $0 for unknown/local models. When
the cost is computed from a tracer, the provider recorded on each LLM event is
used automatically. `clear_pricing()` resets everything to the built-in table.

## Providers

`Harness` (and the `LLM` / `react()` / `harness()` nodes) speaks the OpenAI
chat-completions format by default and ships presets for several providers —
just pick a provider key and an API key env var:

| `provider`          | API key env var      | Notes                                        |
| ------------------- | -------------------- | -------------------------------------------- |
| `openai`            | `OPENAI_API_KEY`     | default                                      |
| `anthropic`         | `ANTHROPIC_API_KEY`  | responses normalised to OpenAI shape         |
| `deepseek`          | `DEEPSEEK_API_KEY`   |                                              |
| `mistral`           | `MISTRAL_API_KEY`    |                                              |
| `together`          | `TOGETHER_API_KEY`   |                                              |
| `groq`              | `GROQ_API_KEY`       |                                              |
| `openrouter`        | `OPENROUTER_API_KEY` |                                              |
| `gemini`            | `GEMINI_API_KEY`     | Google's OpenAI-compatible endpoint          |
| `openai_compatible` | `OPENAI_API_KEY`     | any custom endpoint (vLLM, LM Studio, Azure) |
| `ollama`            | — (local)            |                                              |

```python
from draf.harness import Harness

harness = Harness(
    model="claude-3-5-sonnet-latest",
    provider="anthropic",  # uses ANTHROPIC_API_KEY
    fallbacks=["gpt-4o"],  # fail over to another provider/model
)
reply = await harness.call([{"role": "user", "content": "hi"}])
print(reply.content, reply.cached, reply.latency_ms)
```

`Harness.from_config(cfg)` builds a harness from a node config dict (the same
keys `LLM` / `ReActAgent` accept), so Python and YAML stay in lockstep. In
YAML the `llm_chat` / `react_agent` nodes map `provider`, `base_url`,
`api_key_env`, `chat_path`, `fallbacks`, `cache`, `max_retries`, and the
other transport keys straight through.

## Response caching

`cache=True` dedupes model calls: the request body is hashed and a cached
reply is returned for identical re-calls, so checkpoint resumes and eval
re-runs never pay for the same request twice. Cached replies report
`ModelReply.cached == True`. A custom mutable mapping can be supplied
instead of a plain dict; streaming responses are not cached.

```python
harness = Harness(model="gpt-4o", cache=True)
first = await harness.call(messages)  # network round-trip
second = await harness.call(messages)  # served from cache
assert first.cached is False and second.cached is True
```

## Concurrency

`set_provider_concurrency(provider, limit)` caps concurrent model calls
globally for a provider — across every `Harness` instance, so parallel
branches (each with its own harness) throttle together instead of blowing
past provider rate limits. The explicit limit is authoritative over a
per-harness `max_parallel`.

```python
from draf.harness import set_provider_concurrency

set_provider_concurrency("openai", 8)  # global cap
set_provider_concurrency("openai", 0)  # remove the cap
```

## Workflow validation & typed errors

Validate a workflow YAML **before** running it — schema checks plus a check
that every node/tool type is registered and every edge target exists.

```python
from draf.yaml_schema import validate_workflow_file, format_errors

errors = validate_workflow_file("workflow.yaml")
if errors:
    print(format_errors(errors, source="workflow.yaml"))
```

```bash
draf validate workflow.yaml    # exits non-zero on errors
```

Loading a broken workflow raises typed errors from a public hierarchy rooted
at `draf.DrafError`, so `except draf.DrafError` catches any library failure.
The subclasses multiple-inherit from builtins for back-compat:

```
DrafError
├── ConfigError             (also KeyError)      — invalid config / unknown types
├── WorkflowError           (also RuntimeError)  — workflow-level failures
│   ├── NodeError                                 — a node raised (carries node_id/type)
│   └── LLMError                                  — a model call failed after retries
├── InterruptError                               — HITL resume misuse
├── GraphInterrupt                               — workflow paused for human input
└── StructuredOutputError   (also ValueError)    — schema validation failed
```

```python
try:
    graph = draf.from_yaml("name: bad\nsteps:\n  - id: s1\n    type: react_agnt\n")
except draf.ConfigError as exc:  # also catchable as KeyError
    print(exc)
```

Transport-level failures (timeouts, HTTP status errors) still propagate as
the underlying `httpx` exceptions so existing `except httpx.XError` handlers
keep working.

## Evaluating workflows (`draf eval`)

Score a workflow against a dataset of examples — exact-match by default,
or an LLM judge for open-ended answers.

```
dataset.jsonl   # one JSON object per line
{"id": "q1", "query": "What is the mascot of Draf?", "expected": "a rocket"}
```

Every key except `id` / `expected` is merged into the workflow state as an
initial override (on top of the workflow's own `state.initial`). `.json`
and `.csv` datasets are also accepted.

```bash
draf eval workflow.yaml --data dataset.jsonl --exact
draf eval workflow.yaml --data dataset.jsonl --judge-model gpt-4o --output report.json
```

In Python:

```python
from draf.yaml import load_workflow
from draf.eval import load_dataset, run_eval, format_report

workflow = load_workflow("workflow.yaml")
dataset = load_dataset("dataset.jsonl")
report = await run_eval(workflow, dataset, exact=True)
print(format_report(report))  # total=… passed=… failed=… unscored=… errors=…
```

`--exact` scores by normalised string equality; otherwise an LLM judge
(`--judge-model`, `--judge-provider`) decides PASS/FAIL per example.
`--output-key` names the state key holding the answer (a heuristic
looks through common keys first), `--max-examples` caps the run.

## Examples

| Example                                                 | What it shows                                                                                                                                  |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [basic_pipeline](examples/basic_pipeline/)              | Minimal YAML pipeline, no API keys                                                                                                             |
| [branching](examples/branching/)                        | Conditional edges + Flow API                                                                                                                   |
| [parallel](examples/parallel/)                          | Concurrent branches + typed `State` reducers                                                                                                   |
| [map_repair_plans](examples/map_repair_plans/)          | Dynamic fan-out (`Map`) + `{key}` prompt templates + typed `State`                                                                             |
| [human_in_loop](examples/human_in_loop/)                | Approve/Edit LLM output via `Interrupt` + `loop()` + resume (Python and YAML)                                                                  |
| [react_agent](examples/react_agent/)                    | ReAct agent loop with a calculator tool and live token streaming                                                                               |
| [harness_agent](examples/harness_agent/)                | `flow.harness()` — parallel tool calls in one round + `__error__` fallback                                                                     |
| [agent_approval](examples/agent_approval/)              | Tool approval (HITL) — every tool call pauses for human sign-off and resumes                                                                   |
| [agent_resilience](examples/agent_resilience/)          | Retries, model failover, context trimming and a token budget (mocked, no API key)                                                              |
| [skills](examples/skills/)                              | Skills folder (`SKILL.md`) — instructions + tool scoping on a harness agent                                                                    |
| [pdf_agent](examples/pdf_agent/)                        | Skill with its own tools — vendored `pdf` skill whose bundled scripts run via the shell tool; mounted on both a harness and a plain `LLM` node |
| [mcp](examples/mcp/)                                    | ReAct agent calling tools from an MCP server (stdio)                                                                                           |
| [streaming](examples/streaming/)                        | Live LLM tokens + graph events via `graph.stream()`                                                                                            |
| [structured_output](examples/structured_output/)        | Schema-validated LLM JSON via `output_type`/`json_schema`                                                                                      |
| [rag_search](examples/rag_search/)                      | RAG over a local CSV, in-memory store                                                                                                          |
| [rag_stores](examples/rag_stores/)                      | Same RAG agent on every vector store (in-memory, sqlite, chroma, faiss, lance, milvus, weaviate, qdrant, pgvector, pinecone)                   |
| [checkpoint_resume](examples/checkpoint_resume/)        | Crash/resume in a few lines                                                                                                                    |
| [checkpoint_stores](examples/checkpoint_stores/)        | Durable workflow on file/sqlite/pg                                                                                                             |
| [release_features](examples/release_features/)          | Release API tour — validation, typed errors, `draf eval`, cost reports, response cache (mocked, no API key)                                    |
| [simple_router](examples/simple_router/)                | Minimal `Flow.route()` supervisor — two agents, a bounded loop (can't hang), offline tests                                                     |
| [repair-ai-chat](examples/applications/repair-ai-chat/) | Full FastAPI app built on `route()` — five agents, tools, RAG, streaming (Russian repair workflow)                                             |

All LLM examples run on local Ollama — no API keys. Most use `llama3.1:8b`
(`ollama pull llama3.1:8b`); [pdf_agent](examples/pdf_agent/) uses
`qwen2.5:7b` for more reliable tool-calling (`ollama pull qwen2.5:7b`).

## Development

```bash
uv sync --all-extras           # install deps (incl. optional extras used by tests)
uv run pytest tests/ -q        # tests
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy .                  # types
uv run mkdocs build            # docs (requires: pip install -e ".[docs]")
```

## CI & release

The repo ships three GitHub Actions workflows in [`.github/workflows/`](.github/workflows/):

| Trigger          | What runs                                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Pull request     | `ci.yml` — lint (`ruff check`, `ruff format --check`, `mypy`) **and** tests across Python 3.11/3.12/3.13                    |
| Push to `master` | `ci.yml` — the same lint and tests as a PR (keeps `master` green without opening one)                                      |
| Tag `v*`         | `release.yml` — tests → build sdist/wheel → PyPI → GitHub Release → **Docker images to Docker Hub** → docs to GitHub Pages |

Tests install the full optional surface (`uv sync --frozen --all-extras`) so the
RAG-store, FastAPI, and scaffold tests exercise real integrations; the suite is
fully offline (no API keys), and the LLM examples that need Ollama are not run.

### Cutting a release

1. Bump the version in `pyproject.toml` **and** `draf/_version.py`
   (they must stay in sync), commit, push to `master`.
2. Tag and push — the tag is what triggers the pipeline:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. `release.yml` runs tests, builds the package, publishes it to PyPI, pushes
   the four Docker images (`draf`, `draf-fastapi`, `draf-worker`, `draf-all`)
   to Docker Hub, attaches the artifacts to a GitHub Release, and deploys the
   docs.

### One-time setup

- **PyPI** — the publish step uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
  so no API token is stored. In your PyPI account add a pending publisher for
  `draf`: owner `bzdvdn`, repository `draf`, workflow `release.yml`, environment
  `release` (the publish job pins `environment: release`).
- **Docker Hub** — the `docker` job logs in with two repository secrets:
  `docker_login` (your Docker Hub username) and `docker_password` (a Docker Hub
  **access token**, not your account password — Account Settings → Security →
  Access Tokens, scope _Read & Write_). The image repositories
  (`draf`, `draf-fastapi`, `draf-worker`, `draf-all`) are created automatically
  on the first push.
- **GitHub Pages** — enable Pages on the repository and set the source to
  _GitHub Actions_; the `docs` job deploys `site/` on every release.

See [CONSTITUTION.md](CONSTITUTION.md) for the framework's principles and
non-negotiable rules.
