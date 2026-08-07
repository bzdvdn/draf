# Troubleshooting & gotchas

Real failure modes we hit building the runnable examples, with the root cause
and the fix. Scan this before you debug.

## 1. "An interrupt inside a sub-flow never resumes"

**Symptom:** a workflow with an `Interrupt` *inside* a `SubFlow` (or embedded
`Flow.add_flow`) pauses once, but every resume pauses again immediately.

**Root cause:** when the interrupt fires, the pending resume-point stores the
*outer* node id of the sub-flow (e.g. `subflow_2`). On resume the runtime
re-enters the sub-flow from its entry — so it walks straight back into the
same interrupt.

**Fix:** keep the `Interrupt` at the **top level** of the graph. Put the
escalation *chain* in the outer graph (e.g. `route(..., human=[LLM, Interrupt,
AppendAssistant])`) instead of nesting the interrupt inside the sub-flow.

## 2. "The supervisor never calls the model"

**Symptom:** a `Supervisor` routing node decides nothing / returns `finish`
immediately.

**Root cause:** `Supervisor` only consults the model when there is a **user
message** in the state's `messages` list.

**Fix:** seed the initial state with the user input:

```python
state["messages"] = [{"role": "user", "content": ticket}]
```

## 3. "`graph.run(state=...)` rejects my TypedDict"

**Symptom:** `run()`'s `state` parameter is typed `dict | State`, and a
`TypedDict` instance (`TriageState`) doesn't type-check.

**Root cause:** a `TypedDict` is a `dict` at runtime but not at the type level.

**Fix:** wrap it in the framework's typed-state wrapper, which also carries the
per-key reducers (and then `reducers=` is ignored):

```python
from teff.state import State

state = State(TriageState, dict(initial_state()))
result = await graph.run(state=state, checkpointer=cp, checkpoint_id="ticket-1")
```

## 4. "Resume works in-process but not across processes"

**Symptom:** a run paused on an interrupt resumes fine in one test, but a fresh
process can't continue it.

**Root cause:** resume needs all three to match: the **same checkpointer
backend**, the **same `checkpoint_id`**, and the **same `owner`** (defaults to
`"default"`).

**Fix:** persist `checkpoint_id`/`owner` with the session, and read the
checkpoint with `load()`, not `get()`:

```python
cp = JSONFileCheckpointer("data/checkpoints")
saved = cp.load(owner="default", checkpoint_id="ticket-1")  # None if absent
```

## 5. "State I mutated before the interrupt is gone after resume"

**Symptom:** assertions on state set *before* `GraphInterrupt` fail after
resume.

**Root cause:** on interrupt, the local `state` dict you hold is a copy; the
real state is the checkpoint. The pre-interrupt object is not updated.

**Fix:** read post-resume values from the **resumed result**, not the
pre-interrupt dict:

```python
try:
    result = await graph.run(...)
except GraphInterrupt:
    ...
    result = await graph.run(..., resume={"escalate": answer})
assert result["human_note"] == "..."  # read from the resumed run
```

## 6. "Tests fail only after `uv sync` without extras"

**Symptom:** a handful of tests fail locally but pass in CI.

**Root cause:** `teff`'s heavy extras (e.g. `chromadb`, `pydantic_settings`)
are optional; plain `uv sync` skips them.

**Fix:**

```bash
uv sync --all-extras
```

## 7. "`ruff format --check .` fails on my Markdown"

**Symptom:** CI's format job fails, but no `.py` file changed.

**Root cause:** ruff formats Python blocks **inside Markdown** too, and CI
formats the whole repo (`ruff format --check .`).

**Fix:**

```bash
uv run ruff format docs/recipes/*.md
```

## 8. Offline testing without a model

To test flows that call an LLM with no network, patch the httpx transport and
fabricate the responses the provider expects — both the plain `post` and the
`stream` path (the streaming call passes `"POST"` as its first argument and
returns a context manager with `aiter_lines`). See the mocked transport in
`examples/recipes/support_triage/tests/test_support_triage.py`.