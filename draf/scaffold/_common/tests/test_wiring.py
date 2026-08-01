"""Offline wiring tests — no LLM, no network.

Proves the supervisor Flow built by ``src/graphs/build.py`` is wired
correctly: the ``route()`` edges exist, the entry point is the supervisor,
and the tool set is built.
"""

from src.graphs.build import build_flow


def test_supervisor_route_edges():
    flow, _ = build_flow(model="gpt-4o")
    edges = {(e.source_id, e.target_id, e.condition) for e in flow.compile().edges}

    assert ("supervisor_1", "subflow_2", "next_agent=planner") in edges
    assert ("supervisor_1", "subflow_3", "next_agent=writer") in edges
    assert ("supervisor_1", "subflow_4", "next_agent=reviewer") in edges
    assert ("subflow_2", "supervisor_1", None) in edges  # planner loops back
    assert ("subflow_3", "supervisor_1", None) in edges  # writer loops back
    assert ("subflow_4", "supervisor_1", None) in edges  # reviewer loops back


def test_tools_built():
    flow, tools = build_flow(model="gpt-4o")
    names = {t.name for t in tools}
    assert "current_date" in names


def test_reducers_from_typed_state():
    from src.graphs.state import STATE_REDUCERS, initial_state

    assert "messages" in STATE_REDUCERS  # append reducer
    assert "plan" not in STATE_REDUCERS  # plain keys overwrite, no reducer
    assert initial_state()["next_agent"] == ""
    assert initial_state()["supervisor_rounds"] == 0


def test_supervisor_carries_round_budget():
    """The decider writes a bounded round counter that resets to 0."""
    from src.nodes.supervisor import Supervisor

    node = Supervisor()
    assert node.config["max_rounds"] >= 1
    assert node.config["rounds_key"] == "supervisor_rounds"


def test_flow_entry_point_is_supervisor():
    flow, _ = build_flow(model="gpt-4o")
    g = flow.compile()
    assert g.entry_point == "supervisor_1"
    assert len(g.nodes) == 4  # supervisor + 3 agent subflows


def test_context_builder_keeps_input_and_resets_scratch():
    """The composed ``input`` must survive; scratch keys are cleared."""
    import asyncio

    from src.nodes.context import ContextBuilder

    builder = ContextBuilder(
        sections={"plan": "Plan"},
        reset_keys=("plan", "input", "_plan_messages"),
    )

    async def run() -> dict:
        return await builder.execute(
            None,
            {
                "plan": "previous plan",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    out = asyncio.run(run())
    assert "Plan:\nprevious plan" in out["input"]
    assert "User: hi" in out["input"]
    assert out["plan"] == []  # scratch cleared
    assert out["_plan_messages"] == []  # scratch conversation cleared
