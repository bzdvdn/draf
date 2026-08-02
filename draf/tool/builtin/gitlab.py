"""GitLab tools — list merge requests, fetch diffs, post notes, approve.

A small, purpose-built REST client over the GitLab v4 API so a workflow
can review merge requests without hand-rolling ``http_request`` calls:
auth headers, project path encoding and error surfacing are handled here.

All tools read ``url`` (the GitLab base URL, e.g. ``https://gitlab.com``)
and ``token`` (a personal access token) from config.  A ``project`` is
either a numeric id or a URL-encoded path like ``group/subgroup/repo`` —
path-style ids are URL-encoded automatically.
"""

from __future__ import annotations

import json
from urllib.parse import quote

from draf.tool.tool import Tool


class _GitLabBase(Tool):
    """Shared API client setup for GitLab tools.

    Args:
        config: Optional dict with ``url`` (base URL) and ``token``
            (personal access token).  Both can also be read from the
            environment via ``GITLAB_URL`` / ``GITLAB_TOKEN`` when not
            given in config.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.url = cfg.get("url", "").rstrip("/")
        self.token = cfg.get("token", "")

    def _require(self) -> None:
        if not self.url:
            raise ValueError("gitlab tools require 'url' in config (or GITLAB_URL)")
        if not self.token:
            raise ValueError("gitlab tools require 'token' in config (or GITLAB_TOKEN)")

    @staticmethod
    def _project_id(project: str) -> str:
        if not project:
            raise ValueError("project is required")
        if project.isdigit():
            return project
        return quote(project, safe="")

    async def _request(self, method: str, path: str) -> str:
        self._require()
        import httpx

        url = f"{self.url}/api/v4{path}"
        headers = {"PRIVATE-TOKEN": self.token}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers)
            if response.status_code >= 400:
                raise ValueError(
                    f"GitLab {method} {path} -> HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            return response.text


class GitLabListOpenMRsTool(_GitLabBase):
    """List open merge requests for a project.

    Args:
        project: Project id or URL-encoded path (``group/repo``).
        limit: Maximum number of MRs to return (default 50).
        state: MR state filter (default ``opened``).
    """

    name = "gitlab_list_open_mrs"
    description = "List open merge requests for a GitLab project"

    async def arun(  # type: ignore[override]
        self, project: str, limit: int = 50, state: str = "opened"
    ) -> str:
        pid = self._project_id(project)
        path = f"/projects/{pid}/merge_requests?state={state}&per_page={limit}"
        text = await self._request("GET", path)
        data = json.loads(text)
        lines = []
        for mr in data:
            lines.append(
                f"!{mr['iid']}\t{mr.get('state', '')}\t"
                f"{mr.get('title', '')}\t(mr_id={mr.get('id')})"
            )
        return "\n".join(lines) if lines else "no open merge requests"


class GitLabGetMRChangesTool(_GitLabBase):
    """Fetch a merge request's diff (changed files + line-level changes).

    Args:
        project: Project id or URL-encoded path.
        iid: Merge request internal id (the ``!N`` number).
        max_chars: Cap on the returned diff text (default 20000).
    """

    name = "gitlab_get_mr_changes"
    description = "Fetch the diff of a GitLab merge request"

    async def arun(  # type: ignore[override]
        self, project: str, iid: str, max_chars: int = 20000
    ) -> str:
        pid = self._project_id(project)
        path = f"/projects/{pid}/merge_requests/{iid}/changes"
        text = await self._request("GET", path)
        data = json.loads(text)
        changes = data.get("changes", [])
        out = [f"# MR !{iid}: {data.get('title', '')}"]
        out.append(f"state={data.get('state', '')}  target_branch={data.get('target_branch', '')}")
        for change in changes:
            out.append(f"\n== {change.get('new_path', '')} "
                       f"(+{change.get('new_file', False)} "
                       f"-{change.get('deleted_file', False)})")
            diff = change.get("diff", "")
            out.append(diff[:max_chars])
        return "\n".join(out)


class GitLabPostNoteTool(_GitLabBase):
    """Post a note (comment) on a merge request.

    Args:
        project: Project id or URL-encoded path.
        iid: Merge request internal id.
        body: The note text to post.
    """

    name = "gitlab_post_note"
    description = "Post a comment/note on a GitLab merge request"

    async def arun(  # type: ignore[override]
        self, project: str, iid: str, body: str
    ) -> str:
        pid = self._project_id(project)
        path = f"/projects/{pid}/merge_requests/{iid}/notes"
        import httpx

        self._require()
        url = f"{self.url}/api/v4{path}"
        headers = {"PRIVATE-TOKEN": self.token}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url, headers=headers, json={"body": body}
            )
            if response.status_code >= 400:
                raise ValueError(
                    f"GitLab POST {path} -> HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            data = response.json()
        return f"note posted on !{iid} (note_id={data.get('id')})"


class GitLabApproveTool(_GitLabBase):
    """Approve a merge request.

    Args:
        project: Project id or URL-encoded path.
        iid: Merge request internal id.
    """

    name = "gitlab_approve"
    description = "Approve a GitLab merge request"

    async def arun(  # type: ignore[override]
        self, project: str, iid: str
    ) -> str:
        pid = self._project_id(project)
        path = f"/projects/{pid}/merge_requests/{iid}/approve"
        import httpx

        self._require()
        url = f"{self.url}/api/v4{path}"
        headers = {"PRIVATE-TOKEN": self.token}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers)
            if response.status_code >= 400:
                raise ValueError(
                    f"GitLab POST {path} -> HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
        return f"approved MR !{iid}"


__all__ = [
    "GitLabListOpenMRsTool",
    "GitLabGetMRChangesTool",
    "GitLabPostNoteTool",
    "GitLabApproveTool",
]
