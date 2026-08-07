# Teff

**Workflow as data. Agents as graphs.**

Teff is a Python framework for building durable AI agents and workflows —
an embeddable async library. Inspired by LangGraph and LangChain, it brings
graph-based, stateful agents to Python with minimal dependencies and zero
runtime magic.

- **Async by default** — nodes, tools, and LLM calls are all `async`.
- **Workflow as data** — the canonical graph is YAML/JSON; code is optional.
- **Durable** — checkpoint/resume across file, SQLite, and PostgreSQL backends.
- **Embeddable** — a library, not a platform. You import us; we never import you.

## What you can build

| Need | Do this | Docs |
| ---- | ------- | ---- |
| A data pipeline | Describe nodes + edges in YAML | [YAML workflows](guide/yaml-workflows.md) |
| Build a graph in code | The full `Flow` builder | [Flow builder](guide/flow-builder.md) |
| An LLM call | `LLM` node with prompt templates | [Nodes](reference/nodes.md) |
| Guaranteed JSON | `output_type` / `json_schema` | [Structured output](guide/structured-output.md) |
| A tool-calling agent | `flow.react()` / `harness()` | [Agents](guide/agents.md) |
| A multi-agent supervisor | `flow.route()` + `agent_step()` | [Supervisors](guide/supervisors.md) |
| Conditional routing | `flow.branch()` + `Case` | [Flow builder](guide/flow-builder.md) |
| Parallel work | `Flow.parallel()` / `Parallel` | [State](guide/state.md) |
| Dynamic fan-out | `Flow.map()` over a state list | [State](guide/state.md) |
| Durable / resumable runs | Checkpointers (file/sqlite/pg) | [Durable](guide/durable.md) |
| Human approval | `Interrupt` + resume | [Durable](guide/durable.md) |
| Retrieval | `RAGTool` over 10+ vector stores | [RAG](guide/rag.md) |
| Tool scoping | Skills (`SKILL.md`) | [Skills](guide/skills.md) |
| Custom node/tool types | Plugins (decorators or classes) | [Plugins](guide/plugins.md) |
| Streaming | `graph.stream()` events | [Streaming](guide/streaming.md) |
| Guaranteed JSON | `output_type` / `json_schema` | [Agents](guide/agents.md) |
| Evaluation | `teff eval` / `run_eval` | [Evaluation](guide/evaluation.md) |
| Observability | `RunTracer` / cost reports | [Streaming](guide/streaming.md) |
| Deployable workflows | `Flow.to_yaml()` export | [YAML workflows](guide/yaml-workflows.md) |
| Scaffold an app | `teff new` (fastapi/cli/daemon) | [CLI](cli.md) |

## Getting started

- [Installation](getting-started/install.md) — get teff on `pip`.
- [Quick start](getting-started/quickstart.md) — your first workflow in five
  minutes.
- [Concepts](getting-started/concepts.md) — state, nodes, graphs, tools, RAG.

## Reference

- [Nodes](reference/nodes.md) — every built-in node type and its config keys.
- [Tools](reference/tools.md) — every built-in tool, config, and security notes.
- [Providers](reference/providers.md) — LLM providers, keys, caching, pricing.
- [API Reference](api/index.md) — the full public surface, generated from
  docstrings.
- [Examples](examples.md) — a runnable example for every feature.

## Development

```bash
uv sync                        # install deps
uv run pytest tests/ -q        # tests
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy .                  # types
```