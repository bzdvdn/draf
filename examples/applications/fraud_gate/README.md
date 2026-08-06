# Fraud Gate AI

A production-style **payment screening** FastAPI app built to show
[`Command`](../../../draf/node/command.py) — Draf's dynamic per-node routing.

A payment comes in, an **LLM analyser** reads its free-text note and returns a
structured risk score, and the **router node** picks the next step at runtime
with a `Command` — the decision depends on the *value of the score* the model
just produced, so it cannot be statically wired:

```
POST /api/review ──▶ ingest → analyser (LLM) → router ──(approve)──▶ finalizer (LLM)
                                                     │
                                                     ├─(review)──▶ review_gate ──▶ finalizer
                                                     │              (human analyst,
                                                     │               durable resume)
                                                     └─(deny)────▶ END  (Command.STOP)
```

The three `Command` behaviours on display:

| Route | What the router returns | Why it matters |
| ----- | ----------------------- | -------------- |
| **approve** | `Command(update={...}, goto="finalize")` | jumps to the finalizer even though there is **no** `router → finalize` edge |
| **review** | `Command(update={...})` (no `goto`) | normal `router → review_gate` edge runs; the human analyst's answer is durable and resumes later |
| **deny** | `Command(update={...}, goto=Command.STOP)` | run ends immediately — `finalizer` is **never** reached |

Every `Command` also appends to the event log (`events` reducer), so routing
and state-writing happen in the same return value.

## Run

Requires a local Ollama model (`ollama pull llama3.1:8b`):

```bash
uv sync --extra fastapi
uv run python examples/applications/fraud_gate/main.py
```

Screening requests:

```bash
# auto-approve / deny / review depend on the model's risk score
curl -s -X POST http://127.0.0.1:8001/api/review \
  -H 'Content-Type: application/json' \
  -d '{"tx": {"id": "tx-1", "amount": 1200, "merchant": "amazon.de",
       "country": "DE", "ip_country": "RU",
       "note": "срочно переведу деньги, карту заблокируют"}}'

# a mid-risk payment pauses with waiting:true and a prompt
curl -s -X POST http://127.0.0.1:8001/api/review \
  -H 'Content-Type: application/json' \
  -d '{"tx": {"id": "tx-2", "amount": 60_000, "note": "оплата счёта"}}'

# the analyst resumes the paused review in the same session
curl -s -X POST http://127.0.0.1:8001/api/review/<session_id>/decide \
  -H 'Content-Type: application/json' -d '{"answer": "pass"}'
```

## Layout

```
fraud_gate/
├── main.py                  # server entry point
├── fraud_gate/
│   ├── server.py            # FastAPI factory + Assistant/ReviewService wiring
│   ├── config/config.py     # typed settings (provider, model, thresholds)
│   ├── domain/              # business layer, independent of HTTP
│   │   ├── models.py        # pure value objects (ReviewOutcome / DecideOutcome)
│   │   └── review_service.py# durable sessions + driving the graph
│   ├── graphs/
│   │   ├── build.py         # Ingest / Router nodes + the Command-based flow
│   │   ├── prompts.py       # LLM prompts (analyser + finalizer, structured output)
│   │   └── state.py         # typed state + reducers
│   ├── storage/             # durable checkpoints
│   └── api/                 # thin HTTP layer (schemas + routers)
│       ├── router.py        # /api/health
│       ├── auth/            # X-API-Key dependency
│       └── review/          # /api/review — delegates to ReviewService
└── tests/test_fraud_gate.py # offline: routing + FastAPI, mocked LLM
```

The split keeps the **HTTP layer thin**: `api/review/router.py` only
validates request bodies (pydantic) and delegates to
`domain.review_service.ReviewService`, which owns the durable sessions and
graph orchestration.  The service returns pure `domain/models.py` dataclasses
(`ReviewOutcome` / `DecideOutcome`), and those are **returned directly as
the HTTP response** — FastAPI derives the response schema and serialization
from the handler's return annotation, so pydantic is never needed for
outgoing data and there is no mirroring response DTO.
```

## Offline tests

No LLM, no network — a canned transport serves the analyser's risk score:

```bash
uv run python -m pytest examples/applications/fraud_gate/tests -q
```

Covered: approve → finalizer; mid-risk review pauses and resumes; deny stops
before the finalizer; plus the FastAPI endpoints (`/api/health`,
`/api/review`, `/api/review/{id}/decide`).