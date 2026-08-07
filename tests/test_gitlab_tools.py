import json

import pytest


class TestGitLabTools:
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

                async def request(self, method, url, headers=None):
                    calls.append({"method": method, "url": url})
                    payload = canned_responses[self._idx]
                    self._idx += 1
                    return FakeResponse(payload)

                async def post(self, url, headers=None, json=None):
                    calls.append({"method": "POST", "url": url, "json": json})
                    payload = canned_responses[self._idx]
                    self._idx += 1
                    return FakeResponse(payload)

            monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
            return calls

        self._calls = calls
        self._make_client = make_client

    async def test_list_open_mrs(self):
        from teff.tool.builtin import GitLabListOpenMRsTool

        self._make_client(
            [
                [
                    {
                        "iid": 5,
                        "id": 123,
                        "state": "opened",
                        "title": "Fix flaky test",
                    }
                ]
            ]
        )
        tool = GitLabListOpenMRsTool(
            {"url": "https://gitlab.example.com", "token": "tok"}
        )
        result = await tool.arun(project="group/repo")
        assert "!5" in result
        assert "Fix flaky test" in result
        assert self._calls[0]["method"] == "GET"
        assert "/projects/group%2Frepo/merge_requests" in self._calls[0]["url"]

    async def test_list_open_mrs_numeric_project_id(self):
        from teff.tool.builtin import GitLabListOpenMRsTool

        self._make_client([[]])
        tool = GitLabListOpenMRsTool(
            {"url": "https://gitlab.example.com", "token": "tok"}
        )
        result = await tool.arun(project="42")
        assert result == "no open merge requests"
        assert "/projects/42/merge_requests" in self._calls[0]["url"]

    async def test_get_mr_changes(self):
        from teff.tool.builtin import GitLabGetMRChangesTool

        self._make_client(
            [
                {
                    "iid": 5,
                    "title": "Fix flaky test",
                    "state": "opened",
                    "target_branch": "main",
                    "changes": [
                        {
                            "new_path": "src/app.py",
                            "new_file": False,
                            "deleted_file": False,
                            "diff": "@@ -1 +1 @@\n- old\n+ new",
                        }
                    ],
                }
            ]
        )
        tool = GitLabGetMRChangesTool(
            {"url": "https://gitlab.example.com", "token": "tok"}
        )
        result = await tool.arun(project="group/repo", iid="5")
        assert "src/app.py" in result
        assert "+ new" in result
        assert "/merge_requests/5/changes" in self._calls[0]["url"]

    async def test_post_note(self):
        from teff.tool.builtin import GitLabPostNoteTool

        self._make_client([{"id": 99}])
        tool = GitLabPostNoteTool({"url": "https://gitlab.example.com", "token": "tok"})
        result = await tool.arun(project="group/repo", iid="5", body="Please fix this")
        assert "note posted on !5" in result
        assert self._calls[0]["method"] == "POST"
        assert self._calls[0]["json"] == {"body": "Please fix this"}

    async def test_approve(self):
        from teff.tool.builtin import GitLabApproveTool

        self._make_client([{}])
        tool = GitLabApproveTool({"url": "https://gitlab.example.com", "token": "tok"})
        result = await tool.arun(project="group/repo", iid="5")
        assert "approved MR !5" in result
        assert self._calls[0]["method"] == "POST"
        assert self._calls[0]["url"].endswith("/merge_requests/5/approve")

    async def test_requires_url(self):
        from teff.tool.builtin import GitLabListOpenMRsTool

        with pytest.raises(ValueError, match="url"):
            await GitLabListOpenMRsTool({"token": "tok"}).arun(project="g/r")

    async def test_requires_token(self):
        from teff.tool.builtin import GitLabListOpenMRsTool

        with pytest.raises(ValueError, match="token"):
            await GitLabListOpenMRsTool({"url": "https://x"}).arun(project="g/r")

    async def test_requires_project(self):
        from teff.tool.builtin import GitLabListOpenMRsTool

        with pytest.raises(ValueError, match="project"):
            await GitLabListOpenMRsTool({"url": "https://x", "token": "tok"}).arun(
                project=""
            )

    async def test_http_error_surfaces(self, monkeypatch):
        import httpx

        from teff.tool.builtin import GitLabListOpenMRsTool

        class FakeResponse:
            status_code = 500
            text = "boom"

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
        tool = GitLabListOpenMRsTool({"url": "https://x", "token": "tok"})
        with pytest.raises(ValueError, match="HTTP 500"):
            await tool.arun(project="g/r")

    async def test_ascii_schema_required_args(self):
        from teff.harness import tool_to_schema
        from teff.tool.builtin import (
            GitLabApproveTool,
            GitLabGetMRChangesTool,
            GitLabListOpenMRsTool,
            GitLabPostNoteTool,
        )

        for tool in (
            GitLabListOpenMRsTool({"url": "u", "token": "t"}),
            GitLabGetMRChangesTool({"url": "u", "token": "t"}),
            GitLabPostNoteTool({"url": "u", "token": "t"}),
            GitLabApproveTool({"url": "u", "token": "t"}),
        ):
            params = tool_to_schema(tool)["function"]["parameters"]
            assert "project" in params["required"]
        assert (
            "iid"
            in tool_to_schema(GitLabApproveTool({"url": "u", "token": "t"}))[
                "function"
            ]["parameters"]["required"]
        )
