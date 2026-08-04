# Logging

Draf ships an operational log stream on top of the standard `logging`
module. At `INFO` it shows the **whole run skeleton** — which nodes ran,
how they routed, which tools fired, and the LLM calls (model + token
counts, no text). Prompt/response **content** is an opt-in `DEBUG`
detail, redacted and truncated, so you can inspect what the model saw
without leaking secrets into your console.

## Quick start

```python
from draf import configure_logging
from draf.flow import Flow
from draf.node import LLM
from draf.provider import ProviderRegistry
import asyncio

configure_logging()  # INFO -> stderr, text

flow = Flow(
    "chat",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.step(LLM(prompt="Скажи привет", output_key="answer"))
graph = flow.compile()

asyncio.run(graph.run({}, checkpoint_id="thread-42"))
```

```
2026-08-03 23:06:13,249 INFO  draf.graph.execution [run=bde23c10 session=thread-42] run_start
2026-08-03 23:06:13,251 INFO  draf.graph.execution [run=bde23c10 session=thread-42 node=start type=log_smoke_hi] node_start
2026-08-03 23:06:13,252 INFO  draf.graph.execution [run=bde23c10 session=thread-42 node=start type=log_smoke_hi] node_end duration_ms=5.3
2026-08-03 23:06:13,252 INFO  draf.graph.execution [run=bde23c10 session=thread-42] run_end status=ok
```

## Levels

| Level     | Events                                                                                        |
|-----------|-----------------------------------------------------------------------------------------------|
| `INFO`    | `run_start`, `run_end`, `node_start`, `node_end`, `edge`, `llm_call` (no text), `tool_call`   |
| `DEBUG`   | Everything above plus `llm_request` / `llm_response` **content**, `checkpoint` activity       |
| `WARNING` | `retry`, `structured`-output validation failures, `interrupt` pauses                          |
| `ERROR`   | `node_error` (and a failing `run_end`)                                                        |

So `INFO` is a readable "skeleton" of the chain — you see every node,
every routing decision and every tool invocation — while prompts and
answers stay out of the picture. Flip to `DEBUG` to add the content.

## Configuring

```python
from draf import configure_logging

configure_logging()  # INFO, text
configure_logging("debug")  # + prompt/answer content
configure_logging("INFO", format="json")  # single-line JSON per record
configure_logging("debug", format="json")  # JSON with content
```

`format="json"` prints one JSON object per line to **stdout** (text goes
to stderr), each carrying `run_id`, `session_id`, `node_id`,
`node_type`, `logger`, `level`, `event`, and the per-event fields:

```json
{"timestamp": "2026-08-03T23:06:13", "level": "INFO", "logger": "draf.graph.execution",
 "event": "node_start", "run_id": "bde23c10", "session_id": "thread-42",
 "node_id": "start", "node_type": "log_smoke_hi"}
```

When `level` is omitted it is read from the `DRAF_LOG_LEVEL` environment
variable (default `INFO`). The LLM content cap is set with
`DRAF_LOG_LLM_CHARS` (default `2000` characters, `0` disables
truncation). `configure_logging` is idempotent — calling it again just
switches the level/formatter of the existing handler.

## Correlation

Every record is tagged with the enclosing run/session/node through
`contextvars`, so concurrent runs never bleed into one another. The ids
come from the run itself: `graph.run()`/`graph.stream()` generate a
`run_id` and reuse your `checkpoint_id` as the `session_id`.

Log your own events inside a run and they inherit the same ids:

```python
from draf import get_logger

log = get_logger(__name__)


@node("check_stock")
async def check_stock(ctx, state):
    log.info("stock for %s", state.get("sku"))
    return {"stock": 12}
```

`get_logger` never attaches handlers — that is solely the job of
`configure_logging`. It simply prefixes `draf.` to your name so the
default filters pick the record up.

## Versus RunTracer

`RunTracer` remains the *per-run telemetry*: a structured event log you
can fold into a `RunSummary` and persist. The logger is the *ops
stream*: what is happening right now, correlated by run, filterable by
level, and greppable in a terminal or log aggregator. Use the tracer
when you need a machine-readable report of a finished run; use logging
when you are watching (or debugging) a live one.

## Entry points

The scaffolded apps wire logging up for you:

```bash
uv run python main.py --log-level debug --log-format json
uv run python daemon.py --log-level info --log-format text
uv run python cli.py run "Hello" --log-level debug
```

Without `--log-level` the `DRAF_LOG_LEVEL` env var (or `INFO`) applies.
