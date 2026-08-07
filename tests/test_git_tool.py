"""Offline tests for the git tool.

``subprocess.run`` is faked so tests never touch a real git binary or
repository; only the tool's argument building, output handling and
error surfacing are exercised.
"""

import pytest


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestGitTool:
    def _tool(self, **cfg):
        from teff.tool.builtin import GitTool

        return GitTool(cfg or {"path": "/repo"})

    def _patch(self, monkeypatch, responder):
        from teff.tool.builtin import git as gmod

        calls: list[list] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return responder(cmd)

        monkeypatch.setattr(gmod.subprocess, "run", fake_run)
        return calls

    def test_status_clean(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda cmd: FakeProc(""))
        assert self._tool().run(action="status") == "clean working tree"
        assert calls[0][:3] == ["git", "-C", "/repo"]

    def test_status_dirty(self, monkeypatch):
        self._patch(monkeypatch, lambda cmd: FakeProc(" M file.py\n?? new.txt\n"))
        assert self._tool().run(action="status") == "M file.py\n?? new.txt"

    def test_log(self, monkeypatch):
        calls = self._patch(
            monkeypatch,
            lambda cmd: FakeProc("abc123 Add feature\ncdef45 Fix bug\n"),
        )
        result = self._tool().run(action="log", limit=5)
        assert result == "abc123 Add feature\ncdef45 Fix bug"
        assert calls[0][3] == "log"
        assert calls[0][4:6] == ["-n", "5"]

    def test_diff(self, monkeypatch):
        calls = self._patch(
            monkeypatch, lambda cmd: FakeProc("@@ -1 +1 @@\n-old\n+new\n")
        )
        result = self._tool().run(action="diff")
        assert "+new" in result
        assert calls[0][3:] == ["diff"]

    def test_diff_with_ref_and_path(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda cmd: FakeProc("patch"))
        self._tool().run(action="diff", ref="main", path="src/a.py")
        assert calls[0][3:] == ["diff", "main", "--", "src/a.py"]

    def test_ls_files(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda cmd: FakeProc("a.py\nb.py\n"))
        assert self._tool().run(action="ls_files") == "a.py\nb.py"
        assert calls[0][3:] == ["ls-files"]

    def test_ls_files_with_path(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda cmd: FakeProc("src/a.py\n"))
        assert self._tool().run(action="ls_files", path="src") == "src/a.py"
        assert calls[0][3:] == ["ls-files", "src"]

    def test_branch(self, monkeypatch):
        self._patch(monkeypatch, lambda cmd: FakeProc("* main\n  dev\n"))
        assert "* main" in self._tool().run(action="branch")

    def test_show_defaults_to_head(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda cmd: FakeProc("commit contents"))
        assert self._tool().run(action="show") == "commit contents"
        assert calls[0][3:] == ["show", "HEAD"]

    def test_git_error_raises(self, monkeypatch):
        self._patch(
            monkeypatch, lambda cmd: FakeProc("", "fatal: not a git repository", 128)
        )
        with pytest.raises(ValueError, match="not a git repository"):
            self._tool().run(action="status")

    def test_unknown_action_rejected(self, monkeypatch):
        self._patch(monkeypatch, lambda cmd: FakeProc(""))
        with pytest.raises(ValueError, match="unknown action: commit"):
            self._tool().run(action="commit")

    def test_action_required(self, monkeypatch):
        self._patch(monkeypatch, lambda cmd: FakeProc(""))
        with pytest.raises(ValueError, match="action is required"):
            self._tool().run(action="")

    def test_schema_action_required(self):
        from teff.harness import tool_to_schema
        from teff.tool.builtin import GitTool

        schema = tool_to_schema(GitTool({}))["function"]["parameters"]
        assert "action" in schema["required"]

    def test_registered(self):
        from teff.tool.registry import default_tool_registry

        assert "git" in default_tool_registry.list()
