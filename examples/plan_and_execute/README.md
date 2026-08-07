# Plan-and-execute: plan → fan out → review → replan if needed

The classic LangGraph "plan-and-execute" pattern: an LLM plans a JSON
object of steps, the steps fan out to a `Map` that executes each one (here
a per-step LLM call), a reviewer validates the results and either accepts
them or sends control back to the planner for a revised plan.

```
planner -> extract_steps -> [Map] executor -> reviewer -> extract
                                                       |-- verdict=pass   --> final
                                                       |-- verdict!=pass  --> replan -> extract_steps -> [Map] executor
```

- `planner` returns a JSON object `{"steps": [...]}` via an explicit
  `json_schema` (a bare top-level array is unreliable with local models);
  `extract_steps` pulls the array into `state["steps"]` (raw `json_get`).
- `Map` fans the list out to parallel executor calls
  (`input_keys=["steps"]`, results collected into `state["results"]`).
- `reviewer` returns `{"verdict": "pass"|"needs_work", "feedback": "..."}`.
- On `needs_work` the `replan` node rewrites the plan from the reviewer's
  feedback and the same `Map` runs again.

## Files

| File            | What it shows                                                                  |
| --------------- | ------------------------------------------------------------------------------ |
| `graph.py`      | The loop wired by hand with the low-level `Graph` API — every node, edge and condition explicit |
| `flow.py`       | The same loop with the `Flow` builder — `flow.map()` replaces the `Map` node and `flow.loop()` replaces the back edge |
| `workflow.yaml` | The same workflow as pure YAML (no graph-building code)                         |

Both Python files are fully self-contained (each duplicates its mock), so
either can be read and run on its own. Compare the two to see how the
`Flow` API hides graph plumbing.

## Run it

By default the LLM transport is mocked with a scripted model — the
planner's first plan is rejected once, then accepted. No API key, no
Ollama, no network:

```bash
uv run python examples/plan_and_execute/graph.py
uv run python examples/plan_and_execute/flow.py
```

Expected output (both):

```
Verdict    : pass
planning rounds: 2  review rounds: 2
status: ok  llm_calls: 10
```

To run against a real model, start Ollama with `llama3.1:8b` and set
`TEFF_LIVE=1`:

```bash
ollama pull llama3.1:8b
TEFF_LIVE=1 uv run python examples/plan_and_execute/graph.py
```

The YAML workflow runs against real Ollama too:

```bash
ollama pull llama3.1:8b
uv run teff run examples/plan_and_execute/workflow.yaml
```
