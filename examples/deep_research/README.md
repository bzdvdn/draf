# Deep research: plan → web research → synthesize → review

A research pipeline inspired by the "deep research" agents (LangGraph Deep
Research, OpenAI Deep Research):

```
planner -> extract_topics -> research_agent <-> tool_exec (web_search / fetch_url)
                     |
                     v
writer -> reviewer -> extract
                        |-- verdict=pass   --> final
                        |-- verdict!=pass  --> fix -> reviewer
```

- `planner` breaks the topic into a JSON object `{"topics": [...]}`;
  `extract_topics` pulls the array into `state["topics"]`.
- `research_agent` is a ReAct agent that searches the web (`web_search`)
  and reads pages (`fetch_url`) until it can answer, then writes its notes.
- `writer` synthesises a report; `reviewer` returns a structured JSON
  review; on `needs_work` `fix` rewrites the report and the review loop
  repeats.

## Files

| File            | What it shows                                                                  |
| --------------- | ------------------------------------------------------------------------------ |
| `graph.py`      | The pipeline wired by hand with the low-level `Graph` API — explicit `ReActAgent`/`ToolExec` cycle and back edges |
| `flow.py`       | The same pipeline with the `Flow` builder — `flow.react()` replaces the agent/tool cycle and `flow.loop()` replaces the review/fix back edge |
| `workflow.yaml` | The same workflow as pure YAML (no graph-building code)                         |

Both Python files are fully self-contained (each duplicates its mock and
fake tools), so either can be read and run on its own. Compare the two to
see how the `Flow` API hides graph plumbing.

## Run it

By default the run is fully offline: the LLM transport is mocked and the
`web_search`/`fetch_url` tools are stand-ins returning canned snippets.
No API key, no Ollama, no network:

```bash
uv run python examples/deep_research/graph.py
uv run python examples/deep_research/flow.py
```

Expected output (both):

```
Verdict   : pass
review rounds: 2
status: ok  llm_calls: 8
```

To run against a real model and the real DuckDuckGo search + URL fetch
tools, start Ollama with `llama3.1:8b` and set `DRAF_LIVE=1`:

```bash
ollama pull llama3.1:8b
DRAF_LIVE=1 uv run python examples/deep_research/graph.py
```

The YAML workflow runs against real Ollama and real web tools too:

```bash
ollama pull llama3.1:8b
uv run draf run examples/deep_research/workflow.yaml
```
