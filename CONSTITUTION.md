# Draf Constitution

> "Workflow as data. Agents as graphs."

## Purpose

Draf is a **Python framework** for building durable AI agents and workflows —
an embeddable async library distributed as `pip install draf`.

Inspired by **LangGraph** (stateful, graph-based agents) and **LangChain** (tool framework,
chains, composable primitives), Draf brings these patterns to Python with first-class
support for simplicity, minimal dependencies, and zero runtime magic.

Users extend the framework by implementing interfaces: custom node types (`@node` decorator),
custom tools (`Tool` subclasses), custom LLM providers (HTTP via `httpx`).
The framework owns graph execution, routing, resilience, and observability;
users own business logic.

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

### III. State Is a Flat Dict

No generics, no type parameters, no wrappers.

- State is `dict` — plain, serializable, mergeable.
- Shallow merge on each node result. `state["key"] = value`.
- No implicit state. The entire workflow snapshot is the state dict.
- Branching decisions read from state keys: `state["mode"] == "search"`.

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
- The Pipeline DSL produces the same graph as loading from YAML.
- Inline lambdas/StepFunc are the only exception — they raise on serialization.

### VI. Minimal Dependencies

Draf must remain dependency-light. Every dependency must earn its place.

- Core runtime: zero dependencies beyond Python stdlib + `pyyaml` + `httpx`.
- No orchestration servers, no databases, no message brokers.
- LLM providers communicate via raw HTTP (httpx), never through SDKs.
- No FastAPI, no Django, no Flask, no Celery.

### VII. Async by Default

The entire runtime is `asyncio`. Node execution, tool calls, LLM requests — all async.

- Users write `async def` node functions.
- `graph.run()` is `async`.
- No synchronous fallback paths.

### VIII. Framework — Users Import Us, We Do Not Import Users

Draf is an embeddable library, not a platform.

- All extension points are public: `Node`, `Tool`, `NodeRegistry`, `@node` decorator.
- Users register implementations via `@node` decorator or `NodeRegistry.register()`.
- The framework MUST NOT require forking, configuration files, or code generation
  for extension.

### IX. Observability Is Mandatory

If you cannot inspect execution, you cannot trust execution.

- Every workflow MUST expose: timeline, state transitions, retries, latency,
  token usage, and logs.
- Observability is not optional — it is a first-class feature.

## Runtime Dependencies

Draf MUST keep runtime dependencies absolute minimum:

- `httpx` — async HTTP client for LLM provider communication
- `pyyaml` — YAML serialization/deserialization of workflow graphs
- **NO** Pydantic, no `__init_subclass__` magic, no metaclasses for config
- **NO** LLM SDKs (openai, anthropic, etc.) — raw HTTP only
- **NO** orchestration servers (Celery, Prefect, Airflow)
- **NO** web frameworks (FastAPI, Flask, Django)
- **NO** database drivers in core runtime

Dev/test dependencies (not runtime): ruff, mypy, pytest, uv.

Optional extras (import-error friendly, never block `import draf`):

- `draf[embedding]` — RAG: qdrant-client, chromadb, asyncpg, sqlalchemy
- `draf[tools]` — extra tools: beautifulsoup4, pypdf, boto3, slack-sdk
- `draf[all]` — everything above

## Non-Negotiable Rules

- Rules in this section are `MUST` / `MUST NOT` and are enforceable.
- Implementation MUST follow active spec/plan/tasks and remain in declared scope.
- Work MUST NOT proceed from ambiguous or placeholder requirements.
- Public behavior changes MUST be reflected in specs/tasks before merge.
- If implementation conflicts with this constitution, amend constitution first.
- Code MUST NOT introduce vendor lock-in; prefer HTTP, OpenTelemetry, OpenAPI.
- Zero SDK dependencies for LLM providers — raw HTTP via httpx only.
- No `eval()`, `exec()`, or dynamic code generation in the runtime.

## Constraints

- Public API surface MUST live in `draf/__init__.py`.
- LLM providers, node types, and tools MUST be swappable through public interfaces.
- Workflow state MUST be the single source of truth; no implicit state.
- Every abstraction MUST remove more complexity than it introduces.
- The framework MUST NOT require forking, configuration files, or code generation
  for extension.
- YAML round-trip MUST be stable: serialize → deserialize → serialize produces
  identical output (modulo key ordering).

## Tech Stack

- **Language:** Python >= 3.11
- **CLI:** Typer (post-MVP)
- **Configuration:** `dataclass` (stdlib) + environment variables
- **HTTP Client:** httpx (no SDK wrappers for LLM providers)
- **YAML:** PyYAML
- **Package Manager:** uv
- **Lint:** ruff
- **Types:** mypy
- **Test:** pytest
- **Architecture:** `draf/` package with public modules
- **Agent Graphs:** Directed graphs with conditional edges, branch/case/default,
  parallel branches (future), checkpointing (future)

## Repository Layout

```
draf/
├── __init__.py            # public API exports
├── node.py                # Node base, ExecContext, registry, @node decorator
├── pipeline.py            # Pipeline builder
├── graph.py               # Graph: nodes + edges, run()
├── yaml.py                # from_yaml(), to_yaml()
├── tool.py                # Tool base, FuncTool
├── executor.py            # runtime execution engine
└── builtin/
    ├── __init__.py         # auto-register builtins
    ├── llm.py              # LLM node
    └── transform.py        # Transform node
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

- Version: 1.0.0
- Ratified: 2026-07-31
- Last Amended: 2026-07-31

## Last Updated

2026-07-31 — v1.0.0: First Python version. Ported from Go draftflow.
Async-first, dict state, YAML-native, httpx-only LLM.
