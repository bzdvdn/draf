import pytest

from draf.graph import Graph
from draf.node import Transform

ROOT = __file__.rsplit("/tests/", 1)[0]
EXAMPLE = f"{ROOT}/examples/applications/gitlab-reviewer/workflow.yaml"
REPO_HEALTH = f"{ROOT}/examples/applications/repo-health/workflow.yaml"


class TestEnvInterpolation:
    def test_interpolates_env_in_tool_config(self, tmp_path, monkeypatch):
        from draf.yaml import load_workflow

        monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
        monkeypatch.setenv("GITLAB_TOKEN", "secret-token")
        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: env-workflow
tools:
  - type: gitlab_list_open_mrs
    config:
      url: "${GITLAB_URL}"
      token: "${GITLAB_TOKEN}"
steps:
  - id: s
    type: transform
    config: {action: trim, input_key: x, output_key: y}
"""
        )
        _, tools, _, _ = load_workflow(str(path))
        assert len(tools) == 1
        assert tools[0].url == "https://gitlab.example.com"
        assert tools[0].token == "secret-token"

    def test_missing_env_stays_as_placeholder(self, tmp_path, monkeypatch):
        from draf.yaml import load_workflow

        monkeypatch.delenv("GITLAB_URL", raising=False)
        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: env-workflow
tools:
  - type: gitlab_list_open_mrs
    config:
      url: "${GITLAB_URL}"
      token: "static"
steps:
  - id: s
    type: transform
    config: {action: trim, input_key: x, output_key: y}
"""
        )
        _, tools, _, _ = load_workflow(str(path))
        assert tools[0].url == "${GITLAB_URL}"
        assert tools[0].token == "static"

    def test_works_without_tools(self, tmp_path):
        from draf.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: plain
steps:
  - id: s
    type: transform
    config: {action: uppercase, input_key: t, output_key: o}
"""
        )
        graph, tools, initial, reducers = load_workflow(str(path))
        assert len(tools) == 0
        assert graph.entry_point == "s"


class TestJsonGet:
    @pytest.mark.asyncio
    async def test_extracts_field_from_state_dict(self):
        node = Transform(
            action="json_get", input_key="payload", field="verdict", output_key="v"
        )
        out = await node.execute(None, {"payload": {"verdict": "approve", "score": 3}})
        assert out == {"v": "approve"}

    @pytest.mark.asyncio
    async def test_missing_field_raises(self):
        node = Transform(
            action="json_get", input_key="payload", field="nope", output_key="v"
        )
        with pytest.raises(KeyError, match="nope"):
            await node.execute(None, {"payload": {"verdict": "approve"}})

    @pytest.mark.asyncio
    async def test_non_dict_raises(self):
        node = Transform(
            action="json_get", input_key="payload", field="v", output_key="v"
        )
        with pytest.raises(ValueError, match="dict"):
            await node.execute(None, {"payload": "not a dict"})


class TestNumericConditions:
    def _run(self, condition: str, state: dict) -> bool:
        return Graph({}, [], "")._evaluate(condition, state)

    def test_gte(self):
        assert self._run("diff_lines>=2", {"diff_lines": "2"})
        assert self._run("diff_lines>=2", {"diff_lines": 3})
        assert not self._run("diff_lines>=2", {"diff_lines": 1})

    def test_lte(self):
        assert self._run("diff_lines<=10", {"diff_lines": "10"})
        assert not self._run("diff_lines<=10", {"diff_lines": 11})

    def test_gt_lt(self):
        assert self._run("x>5", {"x": "6"})
        assert not self._run("x>5", {"x": 5})
        assert self._run("x<5", {"x": 4})
        assert not self._run("x<5", {"x": 5})

    def test_missing_key_is_false(self):
        assert not self._run("diff_lines>0", {})

    def test_non_numeric_is_false(self):
        assert not self._run("x>5", {"x": "abc"})

    def test_string_conditions_still_work(self):
        assert self._run("verdict=approve", {"verdict": "approve"})
        assert self._run("verdict!=approve", {"verdict": "comment"})


class TestGitlabReviewerExample:
    def test_example_loads(self):
        from draf.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(EXAMPLE)
        names = {t.name for t in tools}
        assert {
            "gitlab_list_open_mrs",
            "gitlab_get_mr_changes",
            "gitlab_post_note",
            "gitlab_approve",
            "send_telegram",
            "kv_store",
        } <= names
        assert graph.entry_point == "reset"
        assert {"reset", "reviewer", "tool_exec"} <= set(graph.nodes)
        assert initial["project_ids"] == ["group/repo1", "group/repo2"]

    def test_example_validates(self):
        from draf.yaml_schema import validate_workflow_file

        assert validate_workflow_file(EXAMPLE) == []


GITHUB_EXAMPLE = f"{ROOT}/examples/applications/github-reviewer/workflow.yaml"


class TestGithubReviewerExample:
    def test_example_loads(self):
        from draf.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(GITHUB_EXAMPLE)
        names = {t.name for t in tools}
        assert {
            "github_list_open_prs",
            "github_get_pr_changes",
            "github_post_comment",
            "github_approve",
            "send_telegram",
            "kv_store",
        } <= names
        assert graph.entry_point == "reset"
        assert {"reset", "reviewer", "tool_exec"} <= set(graph.nodes)
        assert initial["repo_ids"] == ["owner/repo1", "owner/repo2"]

    def test_example_validates(self):
        from draf.yaml_schema import validate_workflow_file

        assert validate_workflow_file(GITHUB_EXAMPLE) == []


class TestRepoHealthExample:
    def test_example_loads(self):
        from draf.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(REPO_HEALTH)
        names = {t.name for t in tools}
        assert {
            "git",
            "csv_query",
            "redis",
            "lock",
            "wait_for",
            "send_telegram",
        } <= names
        assert graph.entry_point == "reset"
        assert {"reset", "agent", "tool_exec"} <= set(graph.nodes)
        assert initial["priority_csv"] == "data/priority.csv"

    def test_example_validates(self):
        from draf.yaml_schema import validate_workflow_file

        assert validate_workflow_file(REPO_HEALTH) == []

    def test_flow_py_compiles_equivalent_structure(self):
        import importlib.util

        path = f"{ROOT}/examples/applications/repo-health/flow.py"
        spec = importlib.util.spec_from_file_location("repo_health_flow", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        graph = mod.build_flow().compile()
        node_types = {node.type for node in graph.nodes.values()}
        assert node_types == {"context_builder", "react_agent", "tool_exec"}

        tools = mod.build_tools()
        names = {t.name for t in tools}
        assert {
            "git",
            "csv_query",
            "redis",
            "lock",
            "wait_for",
            "send_telegram",
        } <= names


class TestContextBuilderListRendering:
    @pytest.mark.asyncio
    async def test_renders_lists_one_per_line(self):
        from draf.node import ContextBuilder

        node = ContextBuilder(
            sections={"project_ids": "Projects to review"},
            messages_key="messages",
            output_key="input",
        )
        out = await node.execute(
            None, {"project_ids": ["group/a", "group/b"], "messages": []}
        )
        assert out["input"] == "Projects to review:\ngroup/a\ngroup/b"


class TestDaemonOnce:
    YAML = """\
name: daemon-workflow
state:
  initial:
    text: "a\\nb\\nc"
steps:
  - id: bump
    type: transform
    config: {action: count_lines, input_key: text, output_key: lines}
"""

    def test_daemon_once_runs_single_tick(self, tmp_path, capsys):
        from draf.cli import daemon as daemon_cmd

        path = tmp_path / "wf.yaml"
        path.write_text(self.YAML)
        daemon_cmd(
            str(path),
            interval=0,
            once=True,
            trace=False,
            checkpoint=None,
            checkpoint_id="daemon",
            checkpoint_owner="test",
            node_timeout=None,
            max_iterations=None,
        )
        captured = capsys.readouterr()
        assert '"lines": "3"' in captured.out
