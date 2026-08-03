"""GitHub tools — list pull requests, fetch diffs, post comments, approve.

A small, purpose-built REST client over the GitHub API so a workflow can
review pull requests without hand-rolling ``http_request`` calls: auth
headers, endpoint paths and error surfacing are handled here.

All tools read ``token`` (a personal access token) from config and default
to the public ``https://api.github.com`` base URL — override it with
``url`` for GitHub Enterprise.  A ``repo`` is ``owner/repo``.  PRs are
addressed by their ``number`` (the ``#N`` from the web UI).
"""

from __future__ import annotations

import json

from draf.tool.tool import Tool


class _GitHubBase(Tool):
    """Shared API client setup for GitHub tools.

    Args:
        config: Optional dict with ``url`` (base URL, default
            ``https://api.github.com``) and ``token`` (personal access
            token, e.g. a fine-grained token with ``pull_requests: write``
            permission).
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.url = (cfg.get("url", "") or "https://api.github.com").rstrip("/")
        self.token = cfg.get("token", "")

    def _require(self) -> None:
        if not self.token:
            raise ValueError("github tools require 'token' in config (or GITHUB_TOKEN)")

    @staticmethod
    def _repo(repo: str) -> str:
        if not repo or "/" not in repo:
            raise ValueError("repo is required as 'owner/repo'")
        return repo

    async def _request(
        self, method: str, path: str, json_body: dict | None = None
    ) -> str:
        self._require()
        import httpx

        url = f"{self.url}/repos{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(
                method, url, headers=headers, json=json_body
            )
            if response.status_code >= 400:
                raise ValueError(
                    f"GitHub {method} {path} -> HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            return response.text


class GitHubListOpenPRsTool(_GitHubBase):
    """List open pull requests for an ``owner/repo``.

    Args:
        repo: ``owner/repo``.
        limit: Maximum number of PRs to return (default 50).
        state: PR state filter (default ``open``).
    """

    name = "github_list_open_prs"
    description = "List open pull requests for a GitHub repository"

    async def arun(  # type: ignore[override]
        self, repo: str, limit: int = 50, state: str = "open"
    ) -> str:
        r = self._repo(repo)
        path = f"/{r}/pulls?state={state}&per_page={limit}"
        text = await self._request("GET", path)
        data = json.loads(text)
        lines = []
        for pr in data:
            lines.append(
                f"#{pr['number']}\t{pr.get('state', '')}\t"
                f"{pr.get('title', '')}\t(pr_id={pr.get('id')})"
            )
        return "\n".join(lines) if lines else "no open pull requests"


class GitHubGetPRChangesTool(_GitHubBase):
    """Fetch a pull request's diff (changed files + per-file patches).

    Args:
        repo: ``owner/repo``.
        number: Pull request number (the ``#N``).
        max_chars: Cap on the returned diff text (default 20000).
    """

    name = "github_get_pr_changes"
    description = "Fetch the diff of a GitHub pull request"

    async def arun(  # type: ignore[override]
        self, repo: str, number: str, max_chars: int = 20000
    ) -> str:
        r = self._repo(repo)
        path = f"/{r}/pulls/{number}/files"
        text = await self._request("GET", path)
        data = json.loads(text)
        out = [f"# PR #{number}"]
        for change in data:
            out.append(
                f"\n== {change.get('filename', '')} "
                f"(+{change.get('additions', 0)} "
                f"-{change.get('deletions', 0)} {change.get('status', '')})"
            )
            patch = change.get("patch", "")
            out.append(patch[:max_chars])
        return "\n".join(out) if data else f"no changes for PR #{number}"


class GitHubPostCommentTool(_GitHubBase):
    """Post a comment on a pull request (as an issue comment).

    Args:
        repo: ``owner/repo``.
        number: Pull request number.
        body: The comment text to post.
    """

    name = "github_post_comment"
    description = "Post a comment on a GitHub pull request"

    async def arun(  # type: ignore[override]
        self, repo: str, number: str, body: str
    ) -> str:
        r = self._repo(repo)
        path = f"/{r}/issues/{number}/comments"
        text = await self._request("POST", path, json_body={"body": body})
        data = json.loads(text)
        return f"comment posted on #{number} (comment_id={data.get('id')})"


class GitHubApproveTool(_GitHubBase):
    """Approve a pull request by submitting an APPROVE review.

    Args:
        repo: ``owner/repo``.
        number: Pull request number.
    """

    name = "github_approve"
    description = "Approve a GitHub pull request"

    async def arun(  # type: ignore[override]
        self, repo: str, number: str
    ) -> str:
        r = self._repo(repo)
        path = f"/{r}/pulls/{number}/reviews"
        await self._request(
            "POST", path, json_body={"event": "APPROVE", "body": "Approved"}
        )
        return f"approved PR #{number}"


__all__ = [
    "GitHubListOpenPRsTool",
    "GitHubGetPRChangesTool",
    "GitHubPostCommentTool",
    "GitHubApproveTool",
]
