"""AskHuman — pause a ReAct run and wait for an operator's answer.

Human-in-the-loop for tool-calling agents: the agent calls
``ask_human(question=...)`` when it truly needs a person's input (a
preference, a decision, data it cannot compute or fetch).  The ReAct
``ToolExec`` node intercepts the call *before* execution and pauses the
run as a :class:`~teff.node.interrupt.GraphInterrupt` (which requires a
checkpointer).  The operator's reply is supplied via ``resume=`` and
delivered back to the agent as the tool result.

The tool's own ``arun`` is never invoked inside a ReAct loop — the
interception happens before ``execute_tool_calls`` — so this class
exists to carry the LLM-facing schema and to participate in
``use_tools`` scoping.
"""

from teff.tool.tool import Tool


class AskHuman(Tool):
    """Pause the workflow and ask the human operator a question.

    Args:
        question: The question to ask the operator.  The run pauses and
            the operator's free-form reply is returned as the tool result.

    Requires a checkpointer on ``graph.run()`` (the pause is a
    ``GraphInterrupt``); resume with the answer under the ``ask_human``
    state key::

        graph.run(state, checkpointer=cp, resume={"ask_human": "42"})
    """

    name = "ask_human"
    description = (
        "Pause the workflow and ask the human operator a question; "
        "returns the operator's free-form reply.  Use only when you truly "
        "need information only a person can provide (a preference, a "
        "decision, or data you cannot compute or fetch yourself)."
    )
    schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the operator.",
            }
        },
        "required": ["question"],
    }

    async def arun(  # type: ignore[override]
        self, question: str = ""
    ) -> str:
        raise NotImplementedError(
            "ask_human is intercepted by the ToolExec node (it pauses the "
            "run as an interrupt); it cannot be invoked directly"
        )


__all__ = ["AskHuman"]
