"""Tests for workflow YAML validation."""

import pytest


def _yaml_doc(steps: list, edges: list | None = None, tools: list | None = None):
    return {
        "name": "test",
        "steps": steps,
        "edges": edges or [],
        "tools": tools or [],
    }


class TestValidateWorkflow:
    def test_valid_workflow_passes(self):
        from teff.yaml_schema import validate_workflow

        doc = _yaml_doc(
            steps=[
                {
                    "id": "s1",
                    "type": "transform",
                    "config": {
                        "action": "uppercase",
                        "input_key": "x",
                        "output_key": "y",
                    },
                }
            ],
            edges=[{"from": "s1", "to": "s1"}],
        )
        assert validate_workflow(doc) == []

    def test_unknown_node_type(self):
        from teff.yaml_schema import validate_workflow

        errs = validate_workflow(_yaml_doc(steps=[{"id": "s1", "type": "react_agnt"}]))
        assert len(errs) == 1
        assert "react_agnt" in errs[0]["message"]
        assert errs[0]["path"] == "steps[0].type"

    def test_missing_step_id(self):
        from teff.yaml_schema import validate_workflow

        errs = validate_workflow(_yaml_doc(steps=[{"type": "transform"}]))
        assert any("'id' is a required property" in e["message"] for e in errs)

    def test_unknown_tool_type(self):
        from teff.yaml_schema import validate_workflow

        errs = validate_workflow(
            _yaml_doc(
                steps=[{"id": "s1", "type": "transform"}], tools=[{"type": "nope"}]
            )
        )
        assert any("unknown tool type 'nope'" in e["message"] for e in errs)

    def test_edge_references_unknown_step(self):
        from teff.yaml_schema import validate_workflow

        doc = _yaml_doc(
            steps=[{"id": "s1", "type": "transform"}],
            edges=[{"from": "s1", "to": "ghost"}],
        )
        errs = validate_workflow(doc)
        assert any("ghost" in e["message"] for e in errs)

    def test_custom_node_types_are_respected(self):
        from teff.yaml_schema import validate_workflow

        doc = _yaml_doc(steps=[{"id": "s1", "type": "custom_thing"}])
        assert validate_workflow(doc, node_types=["custom_thing"]) == []

    def test_format_errors(self):
        from teff.yaml_schema import format_errors

        text = format_errors(
            [{"path": "steps[0].type", "message": "boom"}], source="wf.yaml"
        )
        assert "wf.yaml: steps[0].type: boom" in text

    def test_validate_workflow_file_missing(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml_schema import validate_workflow_file

        with pytest.raises(ConfigError, match="not found"):
            validate_workflow_file(str(tmp_path / "nope.yaml"))

    def test_validate_workflow_file_invalid_yaml(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml_schema import validate_workflow_file

        path = tmp_path / "wf.yaml"
        path.write_text("steps: [\n  ::: not yaml")
        with pytest.raises(ConfigError, match="invalid YAML"):
            validate_workflow_file(str(path))

    def test_load_workflow_rejects_invalid(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text("name: bad\nsteps:\n  - id: s1\n    type: react_agnt\n")
        with pytest.raises(ConfigError):
            load_workflow(str(path))

    def test_round_trip_from_graph_to_yaml(self):
        from teff.yaml import from_yaml, graph_to_yaml

        graph = from_yaml(
            "name: rt\nsteps:\n  - id: s1\n    type: transform\n    config: "
            "{action: uppercase, input_key: x, output_key: y}\n"
        )
        text = graph_to_yaml(graph)
        reloaded = from_yaml(text)
        assert reloaded.entry_point == "s1"
        assert "type: transform" in text
