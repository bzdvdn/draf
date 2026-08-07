"""Offline tests for the ``hello_workflow`` example — no LLM, no network.

Proves all three abstractions (YAML, Flow DSL, low-level Graph) describe the
same deterministic workflow and route single- vs multi-line input correctly.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1]
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from low_level import build as build_low_level  # noqa: E402

from teff.yaml import load_workflow  # noqa: E402


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(graph, state: dict) -> dict:
    return asyncio.run(graph.run(state=state))


def _graphs():
    yaml_graph, _, _, _ = load_workflow(_EXAMPLE / "workflow.yaml")
    flow = _load_module("flow_dsl").build().compile()
    return yaml_graph, flow, build_low_level()


def test_all_abstractions_agree_on_single_line():
    for graph in _graphs():
        result = _run(graph, {"text": "only"})
        assert result["lines"] == "1"
        assert result["note"] == "single-line note"
        assert result["status"] == "done"


def test_all_abstractions_agree_on_multi_line():
    for graph in _graphs():
        result = _run(graph, {"text": "two\nlines\nhere"})
        assert result["lines"] == "3"
        assert result["note"] == "multi-line note"
        assert result["status"] == "done"


def test_cli_can_run_the_yaml():
    import subprocess

    completed = subprocess.run(
        ["teff", "run", "--file", str(_EXAMPLE / "workflow.yaml")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "single-line note" in completed.stdout
