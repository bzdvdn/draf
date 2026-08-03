# Nodes reference

Every node is a `steps:` entry in YAML (`type: <name>`) and a class in the
`Flow` API. All nodes share the same contract: `async def execute(ctx, state)
-> dict`. Plain functions work too — see [Core concepts](../getting-started/concepts.md).

## Built-in node types

| `type` | Flow class | Purpose |
| ------ | ---------- | ------- |
| `transform` | `Transform` | String transforms on state values |
| `llm_chat` | `LLM` | A single LLM call with prompts, structured output, tools |
| `react_agent` | `ReActAgent` / `Harness` | Tool-calling agent loop (LLM + `tool_exec`) |
| `tool_exec` | `ToolExec` | Execute tool calls signalled by an agent, in parallel |
| `tool_call` | `ToolCall` | Invoke a registered tool by name with fixed args |
| `interrupt` | `Interrupt` | Pause for human input; resume via checkpoint |
| `parallel` | `Parallel` | Run branch chains concurrently, merge with reducers |
| `map` | `Map` | Dynamic fan-out of a state list into parallel branches |
| `context_builder` | `ContextBuilder` | Compose a scratch prompt from state + conversation |
| `append_assistant` | `AppendAssistant` | Append the result as an assistant message |
| `supervisor` | `Supervisor` | Ask a model "which agent next" + deterministic guards |

## Wrapper nodes

### `Retry`

Wrap any node with retry logic — retries the inner node up to `max_retries`
times with an optional `delay` (seconds) between attempts. On final failure
the last exception propagates:

```python
from draf import Retry, LLM

flow.step(Retry(LLM(model="gpt-4", output_key="answer"), max_retries=3, delay=1.0))
```

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `node` | Node | — | The inner node to retry |
| `max_retries` | int | `3` | Attempts (including the first) |
| `delay` | float | `0.0` | Seconds to wait between attempts |

Retries are recorded on the tracer (`tracer.retry(...)`) and surfaced in
`graph.stream()` events.

## `transform`

String transforms. Actions: `uppercase`, `lowercase`, `trim`, `count_lines`,
`value` (set a literal), `json_get` (extract a field from a dict).

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `action` | str | — | One of the actions above |
| `input_key` | str | `""` | State key to read from |
| `output_key` | str | `""` | State key to write to |
| `value` | str | `None` | Literal value (`action: value`, or `json_get` input) |
| `field` | str | `None` | Field to extract with `action: json_get` |

## `llm_chat`

One model call. Provider, sampling, caching and retry keys are shared with
`react_agent`.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `model` | str | — | Model name (required) |
| `provider` | str | `openai` | Provider key (see [Providers](providers.md)) |
| `system` | str | — | System prompt (supports `{key}` templates) |
| `prompt` | str | — | User prompt with `{key}` templates |
| `input_key` | str | — | Read a single state key as the user message |
| `output_key` | str | `"output"` | Where the reply lands |
| `json_schema` / `output_type` | dict/type | — | Structured output validation |
| `parse` | bool | `False` | Parse the reply as JSON into a dict (no validation) |
| `use_tools` | bool/list | — | Tool scope: `True`, `False`, or a list of tool names |
| `skills` / `skill_dir` | list/str | — | Mount [skills](../guide/skills.md) |
| `temperature` / `max_tokens` | float/int | — | Sampling knobs |
| `max_retries` / `fallbacks` | int/list | — | Retry + model failover |
| `cache` | bool | `False` | Dedupe identical calls |
| `max_tool_rounds` | int | `10` | Max model calls per visit |

## `react_agent` / `tool_exec`

The ReAct loop is two nodes: the agent (`react_agent`) proposes tool calls and
the executor (`tool_exec`) runs them in parallel, then routes back on
`_tool_call_name !=`.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `input_key` / `output_key` | str | — | Entry question / final answer |
| `messages_key` | str | `"messages"` | Conversation state key |
| `tool_call_key` | str | `"_tool_call_name"` | Signal key for routing |
| `use_tools` | bool/list | `True` | Tool scope for the agent |
| `tool_error_mode` | str | `"message"` | `"message"` (model sees the error) or `"raise"` |
| `tool_timeout` / `tool_retries` | float/int | — | Bound and retry each tool call |
| `tool_approval` | str/callable | `"auto"` | `"auto"`, `"deny"`, `"interactive"`, or callable |
| `parse_text_tool_calls` | bool | `True` | Decode tool calls from plain text (local models) |
| `max_tool_rounds` | int | `10` | Max model calls per graph visit |
| `max_total_tokens` | int | — | Token budget for the whole agent run |
| `max_context_tokens` / `trim_messages` | int/bool | — | Trim the conversation before each call |

See [Agents](../guide/agents.md) for the full harness surface.

## `tool_call`

Invoke one registered tool with explicit arguments (no model involved).

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `tool` | str | — | Registered tool name |
| `args` | dict | — | Tool args; values support `{key}` templates |
| `output_key` | str | `"output"` | State key for the result |
| `on_error` | str | `"raise"` | `"raise"` or `"message"` |
| `max_chars` | int | — | Truncate the result |

## `interrupt`

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `key` | str | — | State key the resume value lands in |
| `prompt` | str | — | Question shown to the operator |
| `messages_key` | str | `"messages"` | Conversation state key |
| `reset_keys` | list | — | Scratch keys to clear before resume |

Pauses raise `GraphInterrupt`; resume with the same `checkpoint_id` plus
`resume={key: answer}`. Requires a checkpointer. See
[Durable execution](../guide/durable.md).

## `parallel` / `map`

- `parallel` — config `branches`: list of branch chains (each a node or list
  of nodes, or an embedded sub-flow). Branches merge with per-key reducers.
- `map` — config `input_keys` (state list keys, zipped), `output_key` (list of
  per-item results), `result_key` (per-item result key), `chunk_size` (items
  per branch, default 1), `max_concurrency` (cap on branches).

See [State](../guide/state.md).

## `context_builder` / `append_assistant`

Compose the scratch prompt from named state sections and the conversation:

- `context_builder` — `sections` (state key → section label map),
  `messages_key` (default `"messages"`), `output_key` (default `"input"`),
  `reset_keys`.
- `append_assistant` — `output_key` (default `"draft"`), `messages_key`.

## `supervisor`

The decider for a [supervisor loop](../guide/supervisors.md): ask the model
for a one-word route, then apply deterministic guards. Wired with
`flow.supervisor()` or used directly with `flow.route()`.

| Key | Default | Description |
| --- | ------- | ----------- |
| `model` / `provider` | — | LLM model and provider for the harness. |
| `system` | `""` | System prompt (list the reply values + `finish`). |
| `output_key` | `"next_agent"` | State key that receives the chosen route. |
| `sections` | `{}` | State key → label map rendered into the prompt as progress. |
| `route_keys` | `{}` | Map route value → output slot; a picked agent whose slot already has content is not re-routed. |
| `done_keys` / `done_mode` | `{}` / `"all"` | When these output slots are filled, return `finish` with no model call (`"any"` = just one). |
| `fallback_agent` | `""` | Route to this agent when `finish` is picked before anything is produced. |
| `rounds_key` / `max_rounds` | `"supervisor_rounds"` / `6` | Force `finish` once the counter reaches `max_rounds`. |
| `messages_key` | `"messages"` | Source of the user message; `""` means always consult the model. |
| `agents` | — | Explicit reply vocabulary (default: `route_keys ∪ {"finish"} ∪ {fallback_agent}`). |

See [Supervisors — a ready-made decider](../guide/supervisors.md#a-ready-made-decider-supervisor)
for the guards and the `_needs_model` / `decide` override hooks.

## Registering custom types

Use decorators or subclasses — see [Plugins](../guide/plugins.md). The
current registry:

```python
from draf.node.registry import default_registry

print(default_registry.list())
```