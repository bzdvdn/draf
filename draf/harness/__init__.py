"""Agent harness — reusable model↔tool loop.

A harness owns the transport and provider plumbing for one model and
drives the agent loop: call the model, execute requested tools, feed
the results back into the conversation.  It is shared by the
:class:`~draf.node.llm.LLM` node (internal multi-round loop) and the
:class:`~draf.node.agent.ReActAgent` (one step per graph round, so the
loop stays visible as topology).  Tools are ordinary
:class:`~draf.tool.Tool` instances keyed by name, so MCP tools and
built-in tools work unchanged.

    Behaviour can be parameterised through the constructor / ``from_config``:

    - ``max_rounds`` — stop the ``run()`` loop after this many model calls.
    - ``stop_when(messages)`` — extra termination predicate.
    - ``parse_text_tool_calls`` — decode tool calls embedded in plain text
      (local models often skip the structured ``tool_calls`` field).
    - ``tool_error_mode`` — ``"message"`` (default, errors become tool
      messages) or ``"raise"`` (a tool failure propagates, e.g. into an
      ``__error__`` edge).
    - ``tool_timeout`` — per-tool execution timeout in seconds.
    - ``tool_retries`` — extra attempts per tool call after a failure.
    - ``max_retries`` — retry failed HTTP requests (429/5xx/timeouts).
    - ``retry_on`` — status codes / error types worth retrying.
    - ``fallbacks`` — list of fallback model names used when the primary
      transport fails (provider failover).
    - ``max_total_tokens`` — stop the loop once total prompt+completion
      tokens exceed this budget.
    - ``max_context_tokens`` / ``max_context_messages`` — trim the
      conversation history before each model call to fit these limits.
    - ``cache`` — cache model responses keyed by request so re-runs /
      checkpoint resumes do not pay for the same call twice.
    - ``on_tool_call`` — async hook ``(name, args) -> Awaitable[None]``
      invoked before each tool executes (approval/auditing).
    - ``on_step`` / ``on_llm`` / ``on_token`` — observability hooks.
"""

from draf.harness.context import ContextLimitError, trim_messages
from draf.harness.formats import (
    extract_content,
    extract_message,
    extract_usage,
    parse_text_tool_call,
)
from draf.harness.loop import Harness, ModelReply, Step
from draf.harness.providers import (
    PROVIDER_DEFAULTS,
    provider_concurrency,
    resolve_provider,
    set_provider_concurrency,
)
from draf.harness.schema import tool_to_schema
from draf.harness.tools import execute_tool_calls, resolve_approval

__all__ = [
    "Harness",
    "ModelReply",
    "Step",
    "PROVIDER_DEFAULTS",
    "provider_concurrency",
    "set_provider_concurrency",
    "resolve_provider",
    "tool_to_schema",
    "extract_message",
    "extract_content",
    "extract_usage",
    "parse_text_tool_call",
    "execute_tool_calls",
    "resolve_approval",
    "trim_messages",
    "ContextLimitError",
]
