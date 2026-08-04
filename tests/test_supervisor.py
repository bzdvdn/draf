import asyncio

import pytest

from draf.flow import Flow


def _run(coro):
    return asyncio.run(coro)


def _supervisor(**kw):
    from draf.node import Supervisor

    base = dict(
        model="test-model",
        provider="fake",
        sections={"plan": "План"},
        route_keys={"planner": "plan", "reviewer": "review"},
        done_keys={"plan", "review"},
        messages_key="messages",
    )
    base.update(kw)
    return Supervisor(**base)


class TestConfig:
    def test_defaults(self):
        from draf.node import Supervisor

        s = Supervisor()
        assert s.type == "supervisor"
        assert s.config["output_key"] == "next_agent"
        assert s.config["rounds_key"] == "supervisor_rounds"
        assert s.config["max_rounds"] == 6
        assert s.config["messages_key"] == "messages"
        assert s.config["done_mode"] == "all"

    def test_agents_derived_from_route_keys_plus_finish_and_fallback(self):
        s = _supervisor(fallback_agent="direct")
        assert s._agents() == {"planner", "reviewer", "finish", "direct"}

    def test_explicit_agents_override(self):
        s = _supervisor(agents={"a", "b"})
        assert s._agents() == {"a", "b"}


class TestParse:
    def test_parse_returns_last_known_word(self):
        s = _supervisor()
        assert s._parse_agent("go planner") == "planner"
        assert s._parse_agent("..finish..") == "finish"
        assert s._parse_agent("Estimator.") == ""  # not in vocabulary
        assert s._parse_agent("Approved.") == ""  # free text never matches


class TestDecide:
    def test_done_keys_all_filled_short_circuits(self):
        s = _supervisor()
        state = {"plan": "x", "review": "y", "messages": []}
        assert s.decide(state, "planner") == "finish"

    def test_done_mode_any(self):
        s = _supervisor(done_mode="any")
        state = {"plan": "x", "review": "", "messages": []}
        assert s.decide(state, "planner") == "finish"

    def test_fallback_when_finish_before_anything(self):
        s = _supervisor(fallback_agent="direct")
        state = {"plan": "", "review": "", "messages": []}
        assert s.decide(state, "finish") == "direct"

    def test_no_rerun_guard(self):
        s = _supervisor()
        state = {"plan": "x", "review": "", "messages": []}
        assert s.decide(state, "planner") == "finish"

    def test_valid_proposal_passes(self):
        s = _supervisor()
        state = {"plan": "", "review": "", "messages": []}
        assert s.decide(state, "reviewer") == "reviewer"


class TestNeedsModel:
    def test_no_user_message_skips_model(self):
        s = _supervisor()
        assert s._needs_model({"plan": "", "review": "", "messages": []}) is False

    def test_user_message_consults_model(self):
        s = _supervisor()
        state = {
            "plan": "",
            "review": "",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert s._needs_model(state) is True

    def test_done_keys_filled_skips_model(self):
        s = _supervisor()
        state = {
            "plan": "x",
            "review": "y",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert s._needs_model(state) is False

    def test_empty_messages_key_always_consults(self):
        s = _supervisor(messages_key="")
        assert s._needs_model({"plan": "", "review": ""}) is True


class TestExecute:
    def test_bounded_loop_forces_finish(self):
        s = _supervisor()
        state = {"messages": [], "supervisor_rounds": 5}  # max_rounds=6
        result = _run(s.execute(None, dict(state)))
        assert result["next_agent"] == "finish"
        assert result["supervisor_rounds"] == 6

    def test_no_model_needed_routes_from_decide(self):
        s = _supervisor()
        state = {"plan": "", "review": "", "messages": []}
        result = _run(s.execute(None, dict(state)))
        assert result["next_agent"] == "finish"  # no user message to route

    def test_model_proposal_is_used(self):
        s = _supervisor()

        async def fake_ask(ctx, state, *, rounds, max_rounds):
            return "reviewer"

        s._ask_model = fake_ask
        state = {
            "plan": "",
            "review": "",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = _run(s.execute(None, dict(state)))
        assert result["next_agent"] == "reviewer"
        assert result["supervisor_rounds"] == 1


class TestAskModelGraph:
    @pytest.mark.asyncio
    async def test_ask_model_uses_default_model(self, monkeypatch):
        from draf.graph import Graph
        from draf.provider import ProviderRegistry

        bodies = []

        class _Resp:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": self.content}}
                    ]
                }

        async def mock_post(self, url, headers=None, json=None):
            bodies.append(json)
            return _Resp("planner")

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        s = _supervisor(
            model=None,
            provider="openai",
            route_keys={"planner": "plan", "reviewer": "review"},
            done_keys=set(),
        )
        g = Graph(
            nodes={"decider": s},
            edges=[],
            entry_point="decider",
            providers=ProviderRegistry.from_presets("openai"),
            default_model="m-default",
        )
        result = await g.run(
            {"messages": [{"role": "user", "content": "hi"}], "plan": "", "review": ""}
        )
        assert result["next_agent"] == "planner"
        assert bodies[0]["model"] == "m-default"


class TestFlowHelper:
    def test_supervisor_helper_wires_node(self):
        from draf.node import LLM

        flow = Flow("x")
        flow.supervisor(
            model="m",
            provider="p",
            sections={"plan": "План"},
            id="decider",
        ).route(
            "next_agent",
            planner=LLM(model="m", output_key="plan"),
            finish=LLM(model="m", output_key="final"),
        )
        graph = flow.compile()
        assert "decider" in graph.nodes
        assert graph.nodes["decider"].type == "supervisor"

    def test_helper_rejects_instance_plus_config(self):
        from draf.node import Supervisor

        with pytest.raises(TypeError):
            Flow().supervisor(Supervisor(), model="m")

    def test_helper_rejects_wrong_type(self):
        from draf.node import LLM

        with pytest.raises(TypeError):
            Flow().supervisor(LLM(model="m"))


class TestRegistry:
    def test_registered_and_creatable(self):
        from draf.node import Supervisor, default_registry

        assert "supervisor" in default_registry.list()
        node = default_registry.create("supervisor", {"model": "m"})
        assert isinstance(node, Supervisor)
