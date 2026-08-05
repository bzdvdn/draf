# Long-term memory

Checkpoints persist a *single run*; long-term memory persists *knowledge*
across runs. This guide explains the difference and the draf memory
design.

## Checkpoints are not memory

[Checkpoints](durable.md) snapshot the whole workflow state of one run
(`state`, `next_node_id`, `iteration`), scoped by an owner id. They exist
so a run can be resumed, rolled back, or inspected.

They do **not** make a good memory layer:

- **No semantics** — a checkpoint is an opaque dump; you cannot ask "what
  does this user prefer?", only "replay this session".
- **Session-bound** — the snapshot belongs to one `checkpoint_id`; it
  cannot be reused by a different conversation.
- **Unbounded** — the conversation history in `state` grows linearly and
  every LLM call pays for it.
- **No lifecycle** — no aggregation, deduplication, or forgetting.

Long-term memory is a separate concern: *small, relevant, cross-session
facts* that are looked up semantically and evolve over time.

## The three kinds of memory

| Kind | Scope | Example | Persisted in |
| ---- | ----- | ------- | ------------ |
| Short-term (working) | one run | current messages, scratch state | `state` + checkpoints |
| Long-term (facts) | cross-session, per user | "user prefers email over Slack" | memory store |
| Episodic / procedural | cross-session | past successful plans; self-tuned instructions | memory store |

draf's memory layer targets the bottom two rows. The top row is already
handled by `state` reducers and the checkpointer.

## Comparison: LangGraph store

For reference, LangGraph solves the same problem with a separate
`BaseStore`:

- **Namespaces** — hierarchical tuple paths (e.g. `("users", "123",
  "memories")`). The common mistake is rooting the namespace on
  `thread_id`, which recreates per-session silos; the recommended root is
  the **user id**.
- **KV API** — `put(namespace, key, value)`, `get`, `search`, `delete`,
  `list_namespaces`. `value` is a dict; the `text` field is what gets
  embedded.
- **Optional semantic index** — stores can be built with an `index`
  config (`dims` + embed model + `fields`); then `search(query=...)`
  ranks by similarity. Without the index, `search` returns nothing for a
  `query`.
- **TTL** — optional time-to-live per item.
- **LangMem** — a higher-level layer adding `create_manage_memory_tool`
  / `create_search_memory_tool`, plus three memory types (semantic,
  episodic, procedural where the agent rewrites its own system prompt).

draf already has the two hard parts LangGraph had to build: a pluggable
`VectorStore` with semantic search + filter DSL, and an `Embedder`. What
is missing is the namespace/`put`/`search` layer and the agent-facing
tools on top.

## draf design

### `MemoryStore` — a namespace store over `VectorStore`

```python
from draf.memory import MemoryStore
from draf.rag import Embedder, InMemoryVectorStore

memory = MemoryStore(
    store=InMemoryVectorStore(dim=768),
    embedder=Embedder(provider="ollama", model="nomic-embed-text"),
)
```

Namespace → metadata mapping:

- Every item stores its namespace components in metadata
  (`_ns_0`, `_ns_1`, …) and its text under `text`.
- A `search`/`list` call passes a filter over the *prefix* components, so
  `("users", "123")` finds items stored under `("users", "123")` **and**
  `("users", "123", "memories")` — the namespace is a subtree.
- Item ids are stable: `"::".join(namespace) + "::" + key`, so `put` is
  an upsert.

API:

```python
await memory.put(("users", "u1"), "prefs", {"text": "prefers email over Slack", "channel": "email"})
item = await memory.get(("users", "u1"), "prefs")
hits = await memory.search(("users", "u1"), query="how to reach them?", k=5)
await memory.delete(("users", "u1"), "prefs")
keys = await memory.list(("users", "u1"))
```

- `search(query=...)` embeds the query and returns top-k by similarity,
  filtered to the namespace subtree.
- `search(query=None)` returns the most recent items in the namespace
  (metadata carries `updated_at`).
- Optional per-store `ttl`: expired items are skipped on read.

### `MemoryTool` — the agent-facing tool

```python
tools:
  - type: memory
    config:
      store: {type: sqlite, path: ./memory.db, dim: 768}
      embedder: {provider: ollama, model: nomic-embed-text}
      namespace: ["users", "${USER_ID}"]
      default_k: 5
```

Actions exposed to the agent:

- `remember` — upsert one fact; the tool deduplicates: if a *semantically
  close* item already exists in the namespace, the new text overwrites it
  (resolving conflicts by meaning, not just by key).
- `recall` — top-k semantic results formatted for the prompt.
- `forget` — delete by key.
- `list` — enumerate keys (no query → recency order).

### Writing facts

Facts are written one of two ways:

1. **Explicitly** — the agent calls `remember` when it decides something
   is worth keeping.
2. **Extracted** — `MemoryExtractor` runs a lightweight LLM pass that turns
   a conversation into candidate facts, then upserts them. This is the
   LangMem "manage memory" pattern and can be a normal `LLM` node in a
   graph:

   ```python
   from draf.memory import MemoryExtractor

   extractor = MemoryExtractor(model="gpt-4o")
   await extractor.save(memory, conversation, ("users", user_id))
   ```

   Each fact is stored under a stable key derived from its text, so
   re-extracting the same fact updates it in place instead of duplicating
   it.

### Reading facts

Reading is explicit (via `recall`) or injected:

- **Auto-injection** — a `react_agent` or `llm` node accepts a `memory`
  config: on every turn it recalls the top-k memories for the last user
  message and inserts them as a `system` message *before* the LLM call, so
  relevant cross-session facts are visible without the agent having to call
  the tool:

  ```yaml
  react_agent:
    model: gpt-4o
    memory:
      store: {type: sqlite, path: ./memory.db, dim: 768}
      namespace: ["users", "${USER_ID}"]
      k: 5
  ```

  The raw building blocks are `memory_context(store, query, namespace=...)`
  (a ready-to-append block of recalled memories, or `""`) and
  `memory_context_from_config(cfg, state=, ctx=)`, the shared helper both
  nodes use.

  In Python, pass a typed :class:`~draf.memory.context.MemoryConfig` — the
  same shape a config dict or YAML entry carries:

  ```python
  from draf import Flow, MemoryConfig

  flow.react(
      model="gpt-4o",
      memory=MemoryConfig(store=memory, namespace=("users", "ana"), k=5),
  )
  ```

### Per-user memory behind a shared graph

A single compiled graph can serve many end-users: `graph.run(owner=...)`
scopes checkpoints per tenant, and the node's run context carries
`owner` / `session_id` / `checkpoint_id`. Namespace segments may reference
those, so memory becomes per-user without building a graph per request:

```python
flow.react(
    model="gpt-4o",
    memory=MemoryConfig(
        store=memory,
        namespace=["users", "${owner}"],   # resolved from the run context
        k=5,
    ),
)
```

`memory_context_from_config` resolves `${owner}`, `${session_id}` and
`${checkpoint_id}` from the node's context at recall time (falling back to
the process environment, then leaving the reference untouched). Point every
tenant at the same store and keep the root segment — `"users"` — constant
so tenants are just separate sub-namespaces. Facts written for one owner
are never recalled for another; a fact can be shared by putting it under
the common root instead. This is the pattern for chat apps like
`examples/applications/repair-ai-chat`, whose `Assistant.run_turn` already
threads `owner` into `graph.run`.

### Lifecycle

- **Deduplication** — semantic overwrite (see above).
- **TTL / forgetting** — per-item `expires_at`; a `cleanup` method mirrors
  `Checkpointer.cleanup` (`max_age`, `keep_last` per owner/namespace).
- **Ownership** — the namespace root is the user/session id; never the
  checkpoint id, or cross-session memory silently becomes per-session.

### Why this, not more

This is deliberately thin: it reuses `VectorStore` + `Embedder` instead of
building a second storage engine, and it keeps the exact same mental model
as LangGraph's `BaseStore` so patterns learned there transfer. The RAG
layer stays for *documents*; memory is for *facts about the world and the
user*.

## Roadmap

- [x] Design (this document)
- [x] `MemoryStore` (put/get/search/delete/list + subtree namespaces + TTL)
- [x] `MemoryTool` (remember/recall/forget/list)
- [x] Fact-extraction layer (`MemoryExtractor` — LLM summarisation into
  `save`)
- [x] Context injection (`memory_context` + `react_agent` `memory` config)
- [x] Example: `examples/memory_assistant/`