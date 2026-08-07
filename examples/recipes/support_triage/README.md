# Support triage (recipe)

A compact, runnable version of the
[Support triage](../../../docs/recipes/support-triage.md) recipe: a
`Flow.route()` supervisor that answers support tickets automatically or
**escalates to a human** with a durable `Interrupt`.

## The graph

```
supervisor ─ next_agent=billing ──► agent_step(billing) ──┐
   ▲                                                     │
   ├ next_agent=tech ──► agent_step(tech) ───────────────┤
   ├ next_agent=human ──► (draft note → Interrupt) ──────┘
   └──────────── supervisor ◀─────────────────────────────┘
   (done_keys filled → finish, bounded by supervisor_rounds)
```

- `supervisor` — a [`Supervisor`](../../../teff/node/supervisor.py) node that
  picks one of two auto-answer specialists or hands the ticket to a human.
- `agent_step(billing)` / `agent_step(tech)` — the framework's
  `teff.flow.agent_step`: context build → ReAct harness → append the reply to
  the shared `messages`. Auto-answers terminate via `done_keys`.
- `human` — a sub-flow that drafts an escalation note, **pauses on an
  `Interrupt`** (`escalate` key), then appends the note to `messages` on
  resume. The run can't hang: `supervisor_rounds` bounds the loop.

## State

```python
ticket: str  # the incoming ticket
messages: Annotated[list, add]  # shared thread across agents
billing: str  # billing specialist answer slot
tech: str  # technical specialist answer slot
human_note: str  # drafted handover note
escalate: str  # operator's answer after resume
```

## Run online (Ollama)

```bash
ollama pull llama3.1:8b
uv run python examples/recipes/support_triage/main.py
```

The run is durable: `JSONFileCheckpointer` + a fixed `checkpoint_id` mean a
paused escalation resumes across processes. If the router says `human`, the
script prints the escalation prompt and waits for the operator's answer.

## Offline tests (no LLM)

```bash
uv run python -m pytest examples/recipes/support_triage/tests -q
```

Covered with a mocked transport:

- the `next_agent=billing|tech|human` edges exist and every sub-flow loops
  back to the supervisor;
- a billing ticket is answered and the loop terminates
  (`supervisor_rounds == 2`, one supervisor LLM call);
- a supervisor that never says `finish` cannot hang the graph;
- escalation **pauses** on the `escalate` interrupt and **resumes** from the
  checkpoint, with the note reaching the shared conversation.

## Structure

```
support_triage/
├── main.py                     # one durable, interactive ticket
├── support_triage/
│   └── graphs/
│       ├── build.py            # supervisor + route + human sub-flow
│       ├── prompts.py          # router / billing / tech / handover prompts
│       └── state.py            # typed state + reducers
└── tests/test_support_triage.py
```

## Going further

- **Ground answers in your docs**: add a `RAGTool` (see
  [`rag_search`](../../rag_search/) and
  [`RAG guide`](../../../docs/guide/rag.md)) and allowlist it on the
  specialists via `agent_step(..., use_tools=["rag_search"])`.
- **More queues**: add `route_keys` + one `agent_step` per queue.
- **Drive escalation off confidence**: return a `Command` from the supervisor
  instead of a route key (see the
  [`fraud_gate`](../../applications/fraud_gate/) example).
- **Ship it**: wrap the run in FastAPI like
  [`repair-ai-chat`](../../applications/repair-ai-chat/) and use a SQLite/PG
  checkpointer for multi-instance sessions.