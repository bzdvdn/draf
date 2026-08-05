# Self-refine: generate → critique → revise → repeat

The classic LangGraph "reflection" pattern: an LLM drafts, a second LLM
reviews the draft as structured JSON, and if the review is
`needs_work` a third call rewrites the draft — then the critic reviews
again. The loop stops the moment the critic says `pass`.

```
draft --[draft]--> critic --[critique]--> extract
                                           |
                        verdict=pass ------> final
                        verdict!=pass ------> fix -> critic
```

- `draft` writes a first version.
- `critic` returns `{"verdict": ..., "issues": [...]}` via `output_type`
  — schema-validated, re-asked on malformed JSON.
- `extract` (`Transform action="json_get"`) turns the review object into a
  plain `verdict` string the edge conditions read.
- `fix` rewrites the draft and loops back to the critic.

## Files

| File            | What it shows                                                                  |
| --------------- | ------------------------------------------------------------------------------ |
| `graph.py`      | The loop wired by hand with the low-level `Graph` API — every node, edge and condition explicit |
| `flow.py`       | The same loop with the `Flow` builder — one `flow.loop(key, until, done, body)` call replaces the back edge |
| `workflow.yaml` | The same workflow as pure YAML (no graph-building code)                         |

Both Python files are fully self-contained (each duplicates its mock), so
either can be read and run on its own. Compare the two to see how the
`Flow` API hides graph plumbing.

## Run it

By default the LLM transport is mocked with a scripted model — the critic
rejects the first draft, accepts the second — so exactly one revision
happens. No API key, no Ollama, no network:

```bash
uv run python examples/self_refine/graph.py
uv run python examples/self_refine/flow.py
```

Expected output (both):

```
Verdict: pass
revision rounds (critic calls): 2
status: ok  llm_calls: 5
```

To run against a real model, start Ollama with `llama3.1:8b` and set
`DRAF_LIVE=1`:

```bash
ollama pull llama3.1:8b
DRAF_LIVE=1 uv run python examples/self_refine/graph.py
```

The YAML workflow runs against real Ollama too:

```bash
ollama pull llama3.1:8b
uv run draf run examples/self_refine/workflow.yaml
```
