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
| `gate` | `Gate` | Turn a verdict object into a loop decider + retry budget |
| `validate` | `Validate` | Decode an interrupt answer (raw or verdict) into a loop decider; capture a value |

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
`value` (set a literal), `render` (render a `{key}` template into a scalar),
`json_get` (extract a field from a dict), `append` (render a template and
accumulate into a list).

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `action` | str | — | One of the actions above |
| `input_key` | str | `""` | State key to read from |
| `output_key` | str | `""` | State key to write to |
| `value` | str | `None` | Literal value (`action: value`, or `json_get` input) |
| `field` | str | `None` | Field to extract with `action: json_get` |
| `template` | str | `None` | `{key}` template for `action: render` / `action: append` |
| `raw` | bool | `False` | Keep `json_get` values without stringifying |

## `llm_chat`

One model call. Provider, sampling, caching and retry keys are shared with
`react_agent`.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `model` | str | — | Model name (required, or `default_model` on the graph) |
| `provider` | str | — | Provider key (must be declared in `providers=`; see [Providers](providers.md)) |
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

To validate the answer (instead of comparing it verbatim) and capture a
value, pair the interrupt with an `Ask` strategy — see
[`validate`](#validate) and [`Ask`](#ask) below.

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

## `gate`

Turn a verdict object (typically structured JSON from an `LLM`) into the
discriminator value a `flow.loop` / `flow.branch` switches on — the
"approve or fix" loop behind QA and review cycles. Each evaluation
increments `rounds_key`; once it reaches `max_rounds` the gate is forced to
`pass_value` so the loop terminates instead of raising a `max_iterations`
error.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `input_key` | str | `"verdict"` | State key holding the verdict object (`LLM(json_schema=...)` output). |
| `ok_field` | str | `"ok"` | Field of the verdict treated as the pass flag. |
| `output_key` | str | `"decision"` | State key receiving `pass_value` / `fail_value`. |
| `pass_value` | str | `"yes"` | Written on pass (the value `loop` compares *until* against). |
| `fail_value` | str | `"fix"` | Written when the verdict fails. |
| `rounds_key` | str | `"rounds"` | Evaluation counter, incremented each run. |
| `max_rounds` | int | `3` | After this many evaluations the gate is forced to `pass_value`. |
| `message_field` | str | `"message"` | Field of the verdict with the remarks. |
| `message_key` | str | `""` | State key receiving the remarks (cleared on a pass); empty disables. |
| `missing_is_ok` | bool | `True` | A missing / non-dict verdict counts as a pass. |

```python
from draf.node import Gate, LLM

flow.step(qa_llm)          # LLM(json_schema=QaVerdict) -> state["qa_verdict"]
flow.step(Gate(input_key="qa_verdict", output_key="qa_ok", rounds_key="qa_rounds"))
flow.loop(
    key="qa_ok", until="yes",
    done=finalize,
    body=[planner, estimator, qa_llm],
)
```

## `validate`

Like `gate`, but built for interrupt answers: it turns the raw answer (or a
classifier verdict) into the discriminator value a `flow.loop` /
`flow.branch` switches on, and can **capture an arbitrary value** (a
discount code, a date, …) into `value_key`. Each evaluation increments
`rounds_key`; once it reaches `max_rounds` the node is forced to
`pass_value` so the loop terminates instead of raising a `max_iterations`
error.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `input_key` | str | `"answer"` | State key holding the raw answer, or the verdict object for a `model` Ask. |
| `strategy` | str | `""` | `"equals"` / `"any_of"` / `"regex"` / `"check"` for raw answers; `"model"` when `input_key` holds a verdict. |
| `equals` / `any_of` / `regex` / `check` | — | — | Raw-answer matching: exact (normalized) value, a set of values, a regex, or a callable `fn(value) -> bool` / `(bool, extracted)`. |
| `verdict_key` / `ok_field` | str | `"verdict"` / `"ok"` | Where a `model` classifier's verdict object lives and its pass flag. |
| `output_key` | str | `"decision"` | State key receiving `pass_value` / `fail_value`. |
| `pass_value` | str | `"да"` | Written on pass (the value `loop` compares *until* against). |
| `fail_value` | str | `"нет"` | Written when the answer fails. |
| `value_key` | str | `""` | State key receiving the captured value (cleared on a fail); empty disables. |
| `value_field` | str | `""` | Verdict field (for `model`) captured into `value_key`. |
| `rounds_key` | str | `"rounds"` | Evaluation counter, incremented each run. |
| `max_rounds` | int | `100` | After this many evaluations the node is forced to `pass_value`. |
| `missing_is_ok` | bool | `False` | A missing / empty answer counts as a pass. |

## `Ask`

`Ask` is not a node — it's the declarative strategy an interrupt uses to
decide pass/fail and capture a value. `flow.interrupt(key, prompt,
accept=Ask(...))` validates a single answer; `flow.interrupt_loop(key,
accept=Ask(...), body=..., done=...)` re-asks until it passes. Use the
classmethod constructors:

```python
from draf.flow import Flow
from draf.node import Ask, LLM, Transform

# exact (normalized) match
Ask.equals("да", decision_key="plan_ok")

# any of several values
Ask.any_of("да", "ок", "конечно", decision_key="plan_ok")

# regex + capture the value into state["discount_code"]
Ask.regex(r"^[A-Z]{2}-[0-9]{4}$", decision_key="code_ok", value_key="discount_code")

# callable: fn(value) -> bool, or (bool, extracted)
Ask.check(lambda v: len(v) >= 8, value_key="password")

# LLM classifier normalizes free-form answers into {ok: bool, ...}
Ask.model(
    system="Ты классифицируешь ответ пользователя...",
    user="Ответ пользователя:\n{approved}\n\nОдобрил?",
    schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    model="llama3.1:8b", provider="ollama",
    verdict_key="verdict", decision_key="approved_ok",
)

flow.interrupt_loop(
    key="code",
    prompt="Введите промокод (формат XX-1234):",
    accept=Ask.regex(r"^[A-Z]{2}-[0-9]{4}$", decision_key="code_ok", value_key="discount_code"),
    body=Transform(action="value", value="неверный код", output_key="total"),
    done=Transform(action="value", value="скидка применена", output_key="total"),
)
```

`Ask` is auto-detected from the constructor kwargs, so
`Ask(equals="да")` and `Ask(regex=..., value_key="code")` work too. See the
[runnable example](../examples.md) `ask_strategies` for all three strategies
in one checkout flow.

## Registering custom types

Use decorators or subclasses — see [Plugins](../guide/plugins.md). The
current registry:

```python
from draf.node.registry import default_registry

print(default_registry.list())
```