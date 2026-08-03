"""High-level agent helpers built on :class:`~draf.flow.Flow`.

:func:`agent_step` is the shared recipe behind every routed agent in the
scaffolds and examples: compose an ``input`` from shared state, run a
ReAct harness (LLM + optional tools) into an output slot, then append the
final reply to the shared conversation.  Wrapped as a
:class:`~draf.flow.SubFlow`, it plugs straight into
:meth:`draf.flow.Flow.route`.
"""

from __future__ import annotations

from draf.flow.flow import Flow
from draf.flow.sub_flow import SubFlow
from draf.node.context import AppendAssistant, ContextBuilder


def agent_step(
    system: str,
    output_key: str,
    *,
    model: str,
    provider: str = "",
    sections: dict[str, str] | None = None,
    messages_key: str = "messages",
    use_tools: str | list[str] | None = None,
    stream: bool = True,
    id: str | None = None,
    **config,
) -> SubFlow:
    """One routed agent: context builder → ReAct harness → append to conversation.

    Builds a small ``Flow`` wrapped as a :class:`~draf.flow.SubFlow`::

        ContextBuilder ──► ReAct harness ──► AppendAssistant

    * The context builder composes a plain-text ``input`` from the shared
      state sections (plus the latest user message) and resets the agent's
      scratch keys, so each run starts clean.
    * The harness runs the model against that ``input`` with *use_tools*,
      writing its final answer to *output_key*.
    * ``AppendAssistant`` copies that answer into the shared conversation.

    The agent's scratch conversation lives in a private ``_<output_key>_messages``
    state slot (reset by the context builder); only the final reply reaches
    *messages_key*.

    Args:
        system: System prompt for the agent.
        output_key: State key that receives the agent's final answer.
        model: LLM model name (e.g. ``llama3.1:8b``).
        provider: Provider name (e.g. ``ollama``).
        sections: Shared state key → label mapping rendered into the agent's
            context.  Defaults to ``{output_key: output_key.capitalize()}``.
        messages_key: State key holding the shared conversation.
        use_tools: ``None``/``[]`` (no tools, default), ``"all"`` (everything
            the pool offers), or a list of tool names the agent may call.
            Prefer an explicit allowlist; the ``True``/``False`` bool
            shorthands are kept only for backwards compatibility.
        stream: Emit tokens as stream events (live rendering).
        **config: Extra kwargs for the ReAct harness / ``ToolExec``.

    Returns:
        A ``SubFlow`` node usable with :meth:`draf.flow.Flow.route` /
        :meth:`draf.flow.Flow.step`.
    """
    scratch_key = f"_{output_key}_messages"
    inner = Flow(f"agent-{output_key}")
    inner.step(
        ContextBuilder(
            sections=sections or {output_key: output_key.capitalize()},
            reset_keys=(output_key, "input", scratch_key),
        )
    )
    inner.harness(
        model=model,
        system=system,
        input_key="input",
        output_key=output_key,
        messages_key=scratch_key,
        use_tools=use_tools,
        provider=provider,
        stream=stream,
        **config,
    )
    inner.step(AppendAssistant(output_key=output_key, messages_key=messages_key))
    return SubFlow(inner.compile(), id_prefix=id or "")


#: Backwards-compatible alias — ``agent_step`` is the preferred name.
agent_chain = agent_step
