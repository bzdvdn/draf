"""Custom ``failing`` node used by the checkpoint-store examples.

Raises once on the first execution (simulating a transient crash — a
network blip or timeout), then succeeds.  Declared as a plugin in each
backend's ``workflow.yaml`` so the graph loads standalone via
``load_workflow`` without any runner-side registration.
"""

from draf.node import node

# Simulates a transient external failure.  Lives outside the workflow
# state because state is restored to the pre-node checkpoint on resume.
_crash_once = {"armed": True}


@node("failing")
async def failing_node(ctx, state):
    """Raises on the first execution, succeeds afterwards."""
    if _crash_once["armed"]:
        _crash_once["armed"] = False
        raise RuntimeError("simulated transient failure")
    state["recovered"] = True
    return state
