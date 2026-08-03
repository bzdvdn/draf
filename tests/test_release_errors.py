"""Tests for the public exception hierarchy and secret redaction."""

import pytest


class TestErrorHierarchy:
    def test_all_errors_derive_from_draf_error(self):
        from draf.errors import (
            ConfigError,
            DrafError,
            InterruptError,
            LLMError,
            NodeError,
            WorkflowError,
        )

        for cls in (ConfigError, WorkflowError, NodeError, LLMError, InterruptError):
            assert issubclass(cls, DrafError)

    def test_backwards_compatible_builtin_bases(self):
        from draf import GraphInterrupt, StructuredOutputError
        from draf.errors import ConfigError, DrafError, WorkflowError

        assert issubclass(ConfigError, KeyError)
        assert issubclass(WorkflowError, RuntimeError)
        assert issubclass(GraphInterrupt, DrafError)
        assert issubclass(StructuredOutputError, ValueError)

    def test_unknown_node_type_raises_config_error(self):
        from draf.errors import ConfigError
        from draf.node.registry import default_registry

        with pytest.raises(ConfigError, match="unknown node type"):
            default_registry.create("not_a_node")

    def test_max_iterations_raises_workflow_error(self):
        import asyncio

        from draf.errors import WorkflowError
        from draf.graph import Edge, Graph
        from draf.node.node import Node

        class Loop(Node):
            type = "loop"

            async def execute(self, ctx, state: dict) -> dict:  # type: ignore[override]
                return {}

        graph = Graph(
            nodes={"n": Loop()},
            edges=[Edge("n", "n")],
            entry_point="n",
        )
        with pytest.raises(WorkflowError, match="max_iterations"):
            asyncio.run(graph.run({}, max_iterations=1))

    def test_context_limit_error_is_workflow_error(self):
        from draf.errors import WorkflowError
        from draf.harness import ContextLimitError

        assert issubclass(ContextLimitError, WorkflowError)

    def test_invalid_yaml_raises_config_error(self):
        from draf.errors import ConfigError
        from draf.yaml import from_yaml

        with pytest.raises(ConfigError, match="unknown node type 'react_agnt'"):
            from_yaml(
                """
name: bad
steps:
  - id: s1
    type: react_agnt
"""
            )


class TestRedact:
    def test_redacts_dict_values_by_key(self):
        from draf.errors import redact

        out = redact(
            {
                "Authorization": "Bearer sk-secret",
                "Content-Type": "application/json",
                "ok": 1,
            }
        )
        assert out["Authorization"] == "***"
        assert out["Content-Type"] == "application/json"
        assert out["ok"] == 1

    def test_redacts_key_value_pairs_in_strings(self):
        from draf.errors import redact

        assert redact("https://api.example.com/?api_key=sk-abc123&a=1") == (
            "https://api.example.com/?api_key=***&a=1"
        )
        assert redact("Authorization: Bearer sk-longsecret123") == "Authorization: ***"

    def test_redacts_nested_structures(self):
        from draf.errors import redact

        out = redact(
            {"headers": {"x-api-key": "k" * 8}, "list": [{"token": "t" * 8}, 1]}
        )
        assert out["headers"]["x-api-key"] == "***"
        assert out["list"][0]["token"] == "***"
        assert out["list"][1] == 1

    def test_leaves_plain_text_alone(self):
        from draf.errors import redact

        assert (
            redact("plain message without secrets") == "plain message without secrets"
        )
