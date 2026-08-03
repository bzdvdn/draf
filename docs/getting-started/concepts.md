# Core concepts

- **State** — a flat, JSON-serializable dict. Nodes transform state, nothing
  else.
- **Nodes** — pure `async def fn(ctx, state) -> dict` functions (or built-ins:
  `LLM`, `Transform`, `ReActAgent`, `ToolExec`).
- **Graph** — nodes + edges, including conditional edges, branches, and
  `__error__` fallbacks. The graph owns routing and resilience.
- **Tools** — implement `Tool` or use the `@tool` decorator; shareable across
  nodes. A set of built-in tools ships in `draf[tools]`.
- **RAG** — `RAGTool` over pluggable vector stores (`InMemoryVectorStore`,
  `SQLite`, `Chroma`, `Qdrant`, `PGVector`, …).

## State

State is just a dict. Each node is a pure `async` function returning the
changes it wants to apply:

```python
async def my_node(ctx, state):
    state["greeting"] = f"Hello {state.get('name', 'world')}"
    return state
```

## Nodes

A node in YAML is an entry in `steps:` with an `id`, a registered `type`, and
a `config`. In the `Flow` API you instantiate node objects and chain them.
Beyond plain functions, draf ships built-in node types: `transform`, `llm_chat`
(`LLM`), `react_agent` (`ReActAgent`/`Harness`), `tool_exec`, `interrupt`,
`parallel`, `map`.

## Graph & edges

The `Graph` ties nodes together. Plain edges route unconditionally;
conditional edges route on a state key. Branches, cycles (e.g. ReAct loops)
and `__error__` fallbacks all live here.

## Next: the guide

- [YAML workflows](../guide/yaml-workflows.md)
- [State](../guide/state.md)
- [Agents](../guide/agents.md)
- [Durable (checkpoints)](../guide/durable.md)