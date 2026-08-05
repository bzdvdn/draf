# memory_chat

Multi-user streaming chat with per-owner long-term memory, run from the console.

The graph is built **once** and serves every tenant. The operator types an
owner id at startup; `owner` is threaded into `graph.run(owner=...)`, and the
`MemoryConfig` namespace `["users", "${owner}"]` is resolved from the run
context at recall time, so each user's facts live in their own sub-namespace
of one shared store (a SQLite vector store, so facts survive process
restarts — start a new session under the same owner and it already knows).

## How it works

- **Streaming** — `graph.run(..., emit=sink)` returns the final state while
  the sink forwards `token` events to the terminal as they are generated.
- **Injection** — before every LLM call the `react` node recalls the top-*k*
  memories for the last user message under `("users", <owner>)` and prepends
  them as a system message. Second session, same owner → everything is known.
- **Extraction** — after each turn a `MemoryExtractor` summarises the
  exchange into durable facts and writes them back under the same namespace
  (stable SHA-1 keys, so re-extracted facts are upserted, not duplicated).

Try it: state a fact under owner `ana`, quit, start again under `ana` — the
assistant already knows it. Start under `bob` — it does not.

## Requirements

Ollama running locally with:

```
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Usage

```
python examples/memory_chat/main.py
python examples/memory_chat/main.py --owner ana
python examples/memory_chat/main.py --no-extract   # chat only, no fact mining
python examples/memory_chat/main.py --db ./mem.sqlite  # custom memory file
```

Memory lives in `./memories.db` (SQLite). Tell `ana` a fact, quit, restart
as `ana` — the assistant already knows it. Start as `bob` — it does not.

The fact-extraction prompt is a `MemoryExtractor(system_prompt=...)`
override (a `None` keeps the built-in few-shot, language-neutral default,
which works for RU and EN). Scope what gets remembered by setting the
`EXTRACTOR_PROMPT` module constant — e.g. "Extract ONLY budget facts" or
"Extract facts only in French".
