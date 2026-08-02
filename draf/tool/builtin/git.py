"""Git tools — read-only inspection of a git repository.

A single tool with an ``action`` selector (status, log, diff, ls_files,
branch, show) that shells out to ``git`` via ``subprocess`` (no shell
interpolation, fixed argument lists, and a whitelist of read-only
actions so a workflow cannot mutate a repository).
"""

import subprocess

from draf.tool.tool import Tool


class GitTool(Tool):
    """Inspect a git repository without mutating it.

    Args:
        action: ``status`` | ``log`` | ``diff`` | ``ls_files``
            | ``branch`` | ``show``.
        limit: Max commits for ``log`` (default 20).
        ref: Commit/ref for ``diff``/``show`` (default ``HEAD`` for
            ``show``; empty for ``diff`` means working tree).
        path: Restrict ``diff``/``ls_files`` to a path.
        max_chars: Cap on returned output (default 20000).

    Args (config):
        path: Repository directory (default ``.``).
    """

    name = "git"
    description = (
        "Inspect a git repository read-only (status, log, diff, ls_files, "
        "branch, show)"
    )

    _READ_ONLY_ACTIONS = ("status", "log", "diff", "ls_files", "branch", "show")

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.path = cfg.get("path", ".")

    def _git(self, *args: str) -> str:
        cmd = ["git", "-C", self.path, *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise ValueError(f"git {' '.join(args)} failed: {stderr[:500]}")
        return proc.stdout

    def run(  # type: ignore[override]
        self,
        action: str,
        limit: int = 20,
        ref: str = "",
        path: str = "",
        max_chars: int = 20000,
    ) -> str:
        if not action:
            raise ValueError("action is required (status, log, diff, ...)")
        a = action.lower()
        if a not in self._READ_ONLY_ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if a == "status":
            out = self._git("status", "--short")
            return (out.strip()[:max_chars] or "clean working tree")
        if a == "log":
            out = self._git(
                "log", "-n", str(int(limit)), "--oneline", "--decorate"
            )
            return out.strip()[:max_chars] or "no commits"
        if a == "diff":
            args = ["diff"]
            if ref:
                args.append(ref)
            if path:
                args.extend(["--", path])
            out = self._git(*args)
            return out[:max_chars] or "no changes"
        if a == "ls_files":
            args = ["ls-files"]
            if path:
                args.append(path)
            out = self._git(*args)
            return out.strip()[:max_chars] or "no files"
        if a == "branch":
            out = self._git("branch", "-a")
            return out.strip()[:max_chars] or "no branches"
        if a == "show":
            out = self._git("show", ref or "HEAD")
            return out[:max_chars] or "nothing to show"
        raise ValueError(f"unknown action: {action}")


__all__ = ["GitTool"]
