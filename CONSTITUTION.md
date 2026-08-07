# Teff Constitution

> "Workflow as data. Agents as graphs."

## Purpose

Teff is a **Python framework** for building durable AI agents and workflows —
an embeddable async library distributed as `pip install teff`. It treats a
workflow as data: a declarative YAML graph that the runtime executes,
checkpoints, interrupts, and resumes — in-process, with no orchestration
servers and no LLM SDKs.

Who it is for:

- Application developers who want agents they can trust, inspect, and run
  inside their own process.
- Teams shipping production systems where retries, checkpoints, human-in-the-loop
  (HITL) interrupts, and observability are requirements — not afterthoughts.

What it is:

- **Graph-as-data.** Topology is YAML/JSON — versionable, diffable, validated by
  jsonschema. Code is never required to describe structure.
- **Durable by default.** Every run is checkpointable (JSON file, SQLite, PG) and
  resumable from an arbitrary checkpoint; state is plain, serializable dicts.
- **Minimal surface.** `httpx` for provider HTTP, `pyyaml`, `jsonschema`, `typer`.
  Zero runtime magic: no implicit behavior, no hidden branching, no magic imports.

Users extend the framework by implementing interfaces: custom node types (`@node` decorator),
custom tools (`Tool` subclasses or the `@tool` decorator), custom LLM providers (HTTP via
`httpx`), custom skills (`SKILL.md` folders), and plugins (Python modules auto-loaded by the
CLI and `load_workflow`). The framework owns graph execution, routing, resilience, and
observability; users own business logic.

## Core Principles

### I. The Graph Owns Behavior

Business logic belongs to the graph. Never inside prompts, nodes, or tools. To
understand the application, read the graph — not the implementation.

- All workflow branching MUST be visible in the graph structure.
- Nodes MUST NOT encode business decisions about routing, retries, or sequencing.
- Conditions belong on edges, not in nodes. Branching is topology, not code.
- Hidden branching is technical debt and MUST NOT be introduced.

### II. Nodes Transform State

Nodes answer only one question: _How should the state change?_ They never control
execution.

- A node MUST NOT decide what runs next, whether to retry, whether to branch, or
  whether execution should pause.
- Nodes communicate exclusively through state — never through direct calls to
  each other.
- Every node MUST be understandable in one sentence: receives state → returns state.
- Nodes are pure async functions: `async def fn(ctx: ExecContext, state: dict) -> dict`.

### III. State Is a Dict

State is `dict`-shaped — plain, serializable, mergeable. A typed `State`
wrapper (per-key schema and reducers) is an optional overlay that must stay
a drop-in for the plain dict: no new runtime behavior, only validation and
merge policy.

- State is `dict` — plain, serializable, mergeable.
- Shallow merge on each node result. `state["key"] = value`.
- No implicit state. The entire workflow snapshot is the state dict.
- Branching decisions read from state keys: `state["mode"] == "search"`.
- Typed `State` (schema + reducers) MUST keep the plain-dict contract:
  serializable, mergable, and usable wherever a plain dict is accepted.

### IV. The Runtime Owns Execution

Execution is infrastructure. The runtime owns scheduling, retries, persistence,
checkpointing, parallelism, resuming, interruptions, streaming, and observability.

- Nodes MUST NOT implement execution infrastructure concerns.
- State MUST be serializable, durable, recoverable.
- Every retry, timeout, branch, checkpoint, and edge MUST be visible.

### V. Workflow as Data

The canonical graph representation is YAML/JSON. Code is not required to describe
workflow topology.

- Every graph can be serialized to YAML and deserialized back with identical topology.
- YAML files are the source of truth for workflow structure.
- The Flow DSL (`Flow`, `Case`, `SubFlow`) produces the same graph as loading from YAML.
- Inline callables are the only exception — they raise on serialization.

### VI. Minimal Dependencies

Teff must remain dependency-light. Every dependency must earn its place.

- Core runtime: `httpx` (provider HTTP), `pyyaml` (workflow YAML), `jsonschema`
  (schema validation), `typer` (CLI), `mcp` (optional MCP tool transport).
- No orchestration servers, no databases, no message brokers.
- LLM providers communicate via raw HTTP (httpx), never through SDKs.
- No FastAPI, no Django, no Flask, no Celery in the core runtime (FastAPI/Celery
  appear only as optional scaffold templates).
- Optional extras MUST be import-error friendly — they never block `import teff`.

### VII. Async by Default

The entire runtime is `asyncio`. Node execution, tool calls, LLM requests — all async.

- Users write `async def` node functions.
- `graph.run()` is `async`.
- No synchronous fallback paths.

### VIII. Framework — Users Import Us, We Do Not Import Users

Teff is an embeddable library, not a platform.

- All extension points are public: `Node`, `Tool`, `NodeRegistry`, `ToolRegistry`,
  `@node` decorator, `@tool` decorator, `Skill`, plugins.
- Users register implementations via `@node`/`@tool` decorators or the
  `NodeRegistry.register()` / `ToolRegistry.register()` methods.
- Plugins are plain Python modules that register nodes/tools; the CLI and
  `load_workflow` auto-load them.
- The framework MUST NOT require forking, configuration files, or code generation
  for extension.

### IX. Observability Is Mandatory

If you cannot inspect execution, you cannot trust execution.

- Every workflow MUST expose: timeline, state transitions, retries, latency,
  token usage, and logs.
- Observability is not optional — it is a first-class feature.

## Runtime Dependencies

Teff MUST keep runtime dependencies absolute minimum:

- `httpx` — async HTTP client for LLM provider communication
- `pyyaml` — YAML serialization/deserialization of workflow graphs
- `jsonschema` — schema validation (workflow YAML, structured output)
- `typer` — CLI
- `mcp` — Model Context Protocol tool transport (lazily imported)
- **NO** Pydantic, no `__init_subclass__` magic, no metaclasses for config
- **NO** LLM SDKs (openai, anthropic, etc.) — raw HTTP only
- **NO** orchestration servers (Celery, Prefect, Airflow) in the core runtime
- **NO** web frameworks (FastAPI, Flask, Django) in the core runtime
- **NO** database drivers in core runtime

Dev/test dependencies (not runtime): ruff, mypy, pytest, uv.

Optional extras (import-error friendly, never block `import teff`):

- `teff[stores-qdrant]` — Qdrant store (qdrant-client)
- `teff[stores-chroma]` — Chroma store (chromadb)
- `teff[stores-pgvector]` — pgvector store (asyncpg, pgvector)
- `teff[stores-faiss]` — FAISS store (faiss-cpu)
- `teff[stores-lance]` — LanceDB store (lancedb)
- `teff[stores-milvus]` — Milvus store (pymilvus)
- `teff[stores-weaviate]` — Weaviate store (weaviate-client)
- `teff[stores-pinecone]` — Pinecone store (pinecone)
- `teff[embedding]` — all stores at once (alias for every `teff[stores-*]`)
- `teff[rag-pdf]` — PDF ingestion (pypdf)
- `teff[rag-excel]` — Excel ingestion (openpyxl)
- `teff[pg-checkpoint]` — PostgreSQL checkpointer (asyncpg)
- `teff[tools]` — extra tools: beautifulsoup4, pypdf, boto3, slack-sdk, psycopg, redis
- `teff[fastapi]` — scaffold web app (fastapi, uvicorn, sse-starlette)
- `teff[observability]` — trace dashboard (fastapi, uvicorn)
- `teff[queue]` — scaffold worker (celery[redis])
- `teff[docs]` — docs build (mkdocs, mkdocs-material, mkdocstrings, mkdocs-gen-files)
- `teff[all]` — everything above except `docs` (MCP ships in the core package)

## Non-Negotiable Rules

- Rules in this section are `MUST` / `MUST NOT` and are enforceable.
- Implementation MUST follow active spec/plan/tasks and remain in declared scope.
- Work MUST NOT proceed from ambiguous or placeholder requirements.
- Public behavior changes MUST be reflected in specs/tasks before merge.
- If implementation conflicts with this constitution, amend constitution first.
- Code MUST NOT introduce vendor lock-in; prefer HTTP, OpenTelemetry, OpenAPI.
- Zero SDK dependencies for LLM providers — raw HTTP via httpx only.
- No `eval()`, `exec()`, or dynamic code generation in the runtime. The safe,
  AST-whitelisted evaluation tools (`calculator`, `python_eval`) are explicit,
  user-invoked exceptions — they never execute during graph execution itself.

## Constraints

- Public API surface MUST live in `teff/__init__.py`.
- LLM providers, node types, and tools MUST be swappable through public interfaces.
- Workflow state MUST be the single source of truth; no implicit state.
- Every abstraction MUST remove more complexity than it introduces.
- The framework MUST NOT require forking, configuration files, or code generation
  for extension.
- YAML round-trip MUST be stable: serialize → deserialize → serialize produces
  identical output (modulo key ordering).

## Tech Stack

- **Language:** Python >= 3.11
- **CLI:** Typer
- **Configuration:** dataclass (stdlib) + YAML + environment variables
- **HTTP Client:** httpx (no SDK wrappers for LLM providers)
- **YAML:** PyYAML
- **Schema validation:** jsonschema
- **Package Manager:** uv
- **Lint:** ruff
- **Types:** mypy
- **Test:** pytest
- **Architecture:** `teff/` package with public subpackages
- **Agent Graphs:** Directed graphs with conditional edges, branch/case/default,
  parallel branches, dynamic fan-out (Map), checkpoints, interrupts (HITL),
  streaming, ReAct loops

## Repository Layout

```
teff/
├── __init__.py            # public API exports
├── graph/                 # Graph, edges, conditions, execution engine, render
├── yaml.py                # from_yaml()/graph_to_yaml(), load_workflow()
├── yaml_schema.py         # workflow validation (jsonschema)
├── errors.py              # typed error hierarchy (TeffError root)
├── trace.py               # RunTracer, RunSummary, cost/token reports
├── stream.py              # StreamEvent types
├── eval.py                # run_eval(), load_dataset()
├── prompt.py              # render_template
├── schema.py              # json_schema_from_type, validate_json
├── skill.py               # Skill loading, core skills
├── plugins.py             # plugin auto-loading
├── logging.py             # stdlib logging with run/session/node correlation
├── testing.py             # pytest plugin
├── cli.py                 # Typer CLI (run/daemon/graph/validate/eval/inspect/new)
├── checkpoint/            # JSONFile/SQLite/PG checkpointer + owner scoping
├── node/                  # Node base, @node, LLM, Transform, ReActAgent,
│                          #   Supervisor, ToolExec, Map, Parallel, Interrupt, Retry
├── flow/                  # Flow DSL, Case, SubFlow, route(), react()/harness(), supervisor()
├── harness/               # Harness, provider presets, concurrency, formats
├── tool/                  # Tool base, @tool, registry, MCP, builtin tools
├── state/                 # State (typed schema + reducers)
├── rag/                   # RAGTool, embedders, chunker, vector stores
└── scaffold/              # `teff new` templates (fastapi/cli/daemon + variants)
docs/
examples/
```

## Development Workflow

- Each feature MUST be developed in a dedicated git branch.
- Feature branches SHOULD follow `feature/<slug>` naming convention.
- Work SHOULD begin from an explicit spec before implementation starts.
- Plans and tasks SHOULD be derived from the active spec and remain aligned with it.
- Implementation, specs, plans, and tasks MUST comply with this constitution.
- If work reveals a conflict with this constitution, the constitution MUST be
  amended before incompatible implementation proceeds.

## Definition of Done

- A task is done only with observable proof: changed files, targeted test output,
  or command result.
- **Docstrings**: All exported declarations (types, functions, methods, constants)
  completed task.
- Verification MUST confirm acceptance-criteria coverage before archive.

## Constitution Metadata

- Version: 1.3.0
- Ratified: 2026-07-31
- Last Amended: 2026-08-07

## Last Updated

2026-08-07 — v1.3.0: Repositioning. Purpose now reads product-first: target
audience (in-process, production-grade agent developers) and core value
propositions (graph-as-data, durable-by-default, minimal surface). Removed
references to other agent frameworks; the constitution stands on its own
technical merits.

2026-08-04 — v1.2.0: Sync with the project. `graph/` is a package (not a single
`graph.py`); added `logging.py`, `testing.py`, and the `tool/builtin/*` /
`rag/stores/*` modules to the layout. MCP is a core dependency only (the
redundant `teff[mcp]` extra was removed); `teff[all]` now excludes `docs`.

2026-08-03 — v1.1.0: Sync with the project. Package layout (node/flow/harness/
tool/state/rag/checkpoint/scaffold), runtime deps (jsonschema, typer, mcp), typed
`State` overlay, Flow DSL, plugins and skills as extension points, AST-tool
exception to the no-eval rule.

2026-07-31 — v1.0.0: First Python version.
