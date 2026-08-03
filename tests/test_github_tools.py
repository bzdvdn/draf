import json

import pytest


class TestGitHubTools:
    @pytest.fixture(autouse=True)
    def _mock_httpx(self, monkeypatch):
        """Route httpx.AsyncClient to an in-memory fake for every test here."""
        import httpx

        calls: list[dict] = []

        def make_client(canned_responses: list[dict]):
            class FakeResponse:
                status_code = 200

                def __init__(self, payload):
                    self._payload = payload

                @property
                def text(self):
                    return json.dumps(self._payload)

                def json(self):
                    return self._payload

            class FakeClient:
                def __init__(self, *a, **k):
                    self._idx = 0

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

                async def request(self, method, url, headers=None, json=None):
                    calls.append(
                        {
                            "method": method,
                            "url": url,
                            "headers": headers or {},
                            "json": json,
                        }
                    )
                    payload = canned_responses[self._idx]
                    self._idx += 1
                    return FakeResponse(payload)

            monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
            return calls

        self._calls = calls
        self._make_client = make_client

    async def test_list_open_prs(self):
        from draf.tool.builtin import GitHubListOpenPRsTool

        self._make_client(
            [[{"number": 5, "id": 123, "state": "open", "title": "Fix flaky test"}]]
        )
        tool = GitHubListOpenPRsTool({"token": "tok"})
        result = await tool.arun(repo="owner/repo")
        assert "#5" in result
        assert "Fix flaky test" in result
        assert self._calls[0]["method"] == "GET"
        assert "/repos/owner/repo/pulls?state=open" in self._calls[0]["url"]
        assert self._calls[0]["headers"]["Authorization"] == "Bearer tok"

    async def test_list_open_prs_empty(self):
        from draf.tool.builtin import GitHubListOpenPRsTool

        self._make_client([[]])
        tool = GitHubListOpenPRsTool({"token": "tok"})
        assert await tool.arun(repo="owner/repo") == "no open pull requests"

    async def test_get_pr_changes(self):
        from draf.tool.builtin import GitHubGetPRChangesTool

        self._make_client(
            [
                [
                    {
                        "filename": "src/app.py",
                        "status": "modified",
                        "additions": 1,
                        "deletions": 1,
                        "patch": "@@ -1 +1 @@\n- old\n+ new",
                    }
                ]
            ]
        )
        tool = GitHubGetPRChangesTool({"token": "tok"})
        result = await tool.arun(repo="owner/repo", number="5")
        assert "src/app.py" in result
        assert "+ new" in result
        assert self._calls[0]["url"].endswith("/repos/owner/repo/pulls/5/files")

    async def test_post_comment(self):
        from draf.tool.builtin import GitHubPostCommentTool

        self._make_client([{"id": 99}])
        tool = GitHubPostCommentTool({"token": "tok"})
        result = await tool.arun(repo="owner/repo", number="5", body="Please fix this")
        assert "comment posted on #5" in result
        assert self._calls[0]["method"] == "POST"
        assert self._calls[0]["json"] == {"body": "Please fix this"}
        assert self._calls[0]["url"].endswith("/repos/owner/repo/issues/5/comments")

    async def test_approve(self):
        from draf.tool.builtin import GitHubApproveTool

        self._make_client([{"id": 1}])
        tool = GitHubApproveTool({"token": "tok"})
        result = await tool.arun(repo="owner/repo", number="5")
        assert "approved PR #5" in result
        assert self._calls[0]["method"] == "POST"
        assert self._calls[0]["json"]["event"] == "APPROVE"
        assert self._calls[0]["url"].endswith("/repos/owner/repo/pulls/5/reviews")

    def test_default_base_url(self):
        from draf.tool.builtin import GitHubListOpenPRsTool

        tool = GitHubListOpenPRsTool({"token": "tok"})
        assert tool.url == "https://api.github.com"

    def test_custom_base_url(self):
        from draf.tool.builtin import GitHubListOpenPRsTool

        tool = GitHubListOpenPRsTool(
            {"url": "https://ghe.example.com/", "token": "tok"}
        )
        assert tool.url == "https://ghe.example.com"

    async def test_requires_token(self):
        from draf.tool.builtin import GitHubListOpenPRsTool

        with pytest.raises(ValueError, match="token"):
            await GitHubListOpenPRsTool({}).arun(repo="owner/repo")

    async def test_requires_repo(self):
        from draf.tool.builtin import GitHubListOpenPRsTool

        with pytest.raises(ValueError, match="owner/repo"):
            await GitHubListOpenPRsTool({"token": "tok"}).arun(repo="bad")

    async def test_http_error_surfaces(self, monkeypatch):
        import httpx

        from draf.tool.builtin import GitHubListOpenPRsTool

        class FakeResponse:
            status_code = 401
            text = "Bad credentials"

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def request(self, *a, **k):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
        with pytest.raises(ValueError, match="HTTP 401"):
            await GitHubListOpenPRsTool({"token": "tok"}).arun(repo="owner/repo")

    def test_schema_required_args(self):
        from draf.harness import tool_to_schema
        from draf.tool.builtin import (
            GitHubApproveTool,
            GitHubGetPRChangesTool,
            GitHubListOpenPRsTool,
            GitHubPostCommentTool,
        )

        for tool in (
            GitHubListOpenPRsTool({"token": "t"}),
            GitHubGetPRChangesTool({"token": "t"}),
            GitHubPostCommentTool({"token": "t"}),
            GitHubApproveTool({"token": "t"}),
        ):
            params = tool_to_schema(tool)["function"]["parameters"]
            assert "repo" in params["required"]
        approve_required = tool_to_schema(GitHubApproveTool({"token": "t"}))[
            "function"
        ]["parameters"]["required"]
        assert "number" in approve_required
