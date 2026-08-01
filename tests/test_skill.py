import pytest


@pytest.fixture
def analyst_skill(tmp_path):
    d = tmp_path / "analyst"
    d.mkdir()
    (d / "SKILL.md").write_text(
        """---
name: analyst
description: Answer questions over tabular data
allowed-tools:
  - calc
disallowed-tools: [secret]
---

You are a data analyst. Always use calc first.
""",
        encoding="utf-8",
    )
    return d


def test_load_skill_parses_frontmatter(analyst_skill):
    from draf.skill import load_skill

    s = load_skill(analyst_skill)
    assert s.name == "analyst"
    assert s.description == "Answer questions over tabular data"
    assert s.allowed_tools == ["calc"]
    assert s.disallowed_tools == ["secret"]
    assert "data analyst" in s.instructions
    assert s.path == analyst_skill


def test_load_skill_defaults_name_to_folder(tmp_path):
    from draf.skill import load_skill

    d = tmp_path / "writer"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\ndescription: Writes\n---\n\nWrite well.", encoding="utf-8"
    )
    s = load_skill(d)
    assert s.name == "writer"
    assert s.allowed_tools is None
    assert s.disallowed_tools == []


def test_load_skill_missing_raises(tmp_path):
    from draf.skill import load_skill

    with pytest.raises(FileNotFoundError):
        load_skill(tmp_path / "nope")


def test_load_skill_accepts_skill_md_path(analyst_skill):
    from draf.skill import load_skill

    s = load_skill(analyst_skill / "SKILL.md")
    assert s.name == "analyst"


def test_resolve_skills_accepts_names_paths_and_objects(tmp_path, analyst_skill):
    from draf.skill import Skill, resolve_skills

    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "SKILL.md").write_text(
        "---\n---\n\nPlain body.", encoding="utf-8"
    )

    skills = resolve_skills({"skills": [analyst_skill, "other"], "skill_dir": tmp_path})
    assert [s.name for s in skills] == ["analyst", "other"]

    skills = resolve_skills({"skills": [Skill(name="inline", instructions="x")]})
    assert skills[0].name == "inline"


def test_skills_instructions_renders_blocks(analyst_skill):
    from draf.skill import load_skill, skills_instructions

    text = skills_instructions([load_skill(analyst_skill)])
    assert "### Skill: analyst" in text
    assert "You are a data analyst." in text


def _tools():
    from draf.tool import Tool

    class Calc(Tool):
        name = "calc"
        description = "Compute"

        def run(self, expression: str = "") -> str:  # type: ignore[override]
            return "0"

    class Secret(Tool):
        name = "secret"
        description = "Hidden"

        def run(self, **kwargs) -> str:  # type: ignore[override]
            return "s"

    class Read(Tool):
        name = "read"
        description = "Read"

        def run(self, path: str = "") -> str:  # type: ignore[override]
            return "r"

    return {"calc": Calc(), "secret": Secret(), "read": Read()}


def test_scope_tools_list_narrows_pool():
    from draf.skill import scope_tools

    pool = _tools()
    scoped = scope_tools(pool, {"use_tools": ["calc"]})
    assert list(scoped) == ["calc"]


def test_scope_tools_false_is_empty():
    from draf.skill import scope_tools

    assert scope_tools(_tools(), {"use_tools": False}) == {}


def test_scope_tools_skill_allowed_intersects(analyst_skill):
    from draf.skill import load_skill, scope_tools

    skills = [load_skill(analyst_skill)]
    # skill allows only [calc], so even with all tools enabled we get calc
    scoped = scope_tools(_tools(), {"use_tools": True}, skills)
    assert list(scoped) == ["calc"]

    # node list intersects with skill allowed-tools
    scoped = scope_tools(_tools(), {"use_tools": ["calc", "read"]}, skills)
    assert list(scoped) == ["calc"]


def test_scope_tools_disallowed_removes(analyst_skill):
    from draf.skill import load_skill, scope_tools

    skills = [load_skill(analyst_skill)]
    scoped = scope_tools(_tools(), {"use_tools": ["calc", "secret"]}, skills)
    # allowed -> calc (intersect), then disallowed removes secret
    assert list(scoped) == ["calc"]


class TestSkillIntegration:
    @pytest.mark.asyncio
    async def test_llm_merges_instructions_and_scopes_tools(
        self, monkeypatch, tmp_path
    ):
        from draf.node import ExecContext, LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        d = tmp_path / "analyst"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: analyst\ndescription: Tabular expert\nallowed-tools: [calc]\n---\n\n"
            "You are a data analyst. Use calc.",
            encoding="utf-8",
        )

        captured = {}

        async def mock_post(*a, **kw):
            captured["body"] = kw.get("json")

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "done"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM(
            {
                "model": "gpt-4",
                "system": "Base system",
                "output_key": "out",
                "skills": [str(d)],
                "use_tools": True,
            }
        )
        ctx = ExecContext(state={}, tools=_tools())
        await node.execute(ctx, {"input": "hi"})

        messages = captured["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert "Base system" in messages[0]["content"]
        assert "### Skill: analyst" in messages[0]["content"]
        assert "You are a data analyst." in messages[0]["content"]

        names = [t["function"]["name"] for t in captured["body"]["tools"]]
        assert names == ["calc"]

    @pytest.mark.asyncio
    async def test_llm_skill_instructions_with_braces(self, monkeypatch, tmp_path):
        """Skill bodies may contain ``{...}`` (e.g. code samples) — they must
        not be interpolated as template placeholders on a plain LLM node."""
        from draf.node import ExecContext, LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        d = tmp_path / "pdf"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: pdf\ndescription: PDF processing\n---\n\n"
            "Extract text:\n"
            "```python\n"
            "from pypdf import PdfReader\n"
            "reader = PdfReader('document.pdf')\n"
            "print(f'Pages: {len(reader.pages)}')\n"
            "```\n",
            encoding="utf-8",
        )

        captured = {}

        async def mock_post(*a, **kw):
            captured["body"] = kw.get("json")

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "done"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM(
            {
                "model": "gpt-4",
                "system": "Base system",
                "output_key": "out",
                "skills": [str(d)],
            }
        )
        ctx = ExecContext(state={}, tools=_tools())
        await node.execute(ctx, {"input": "hi"})

        content = captured["body"]["messages"][0]["content"]
        assert "Base system" in content
        assert "### Skill: pdf" in content
        assert "len(reader.pages)" in content

    @pytest.mark.asyncio
    async def test_react_agent_scopes_tools_with_skill(self, monkeypatch, tmp_path):
        from draf.node import ExecContext
        from draf.node.agent import ReActAgent

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        d = tmp_path / "analyst"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: analyst\nallowed-tools: [calc]\n---\n\nUse calc.",
            encoding="utf-8",
        )

        captured = {}

        async def mock_post(*a, **kw):
            captured["body"] = kw.get("json")

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "final"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = ReActAgent(
            {
                "model": "gpt-4",
                "input_key": "query",
                "output_key": "answer",
                "skills": [str(d)],
            }
        )
        ctx = ExecContext(state={}, tools=_tools())
        result = await node.execute(ctx, {"query": "hi"})

        assert result["answer"] == "final"
        assert "### Skill: analyst" in captured["body"]["messages"][0]["content"]
        names = [t["function"]["name"] for t in captured["body"]["tools"]]
        assert names == ["calc"]

    @pytest.mark.asyncio
    async def test_flow_harness_passes_skills_to_nodes(self, tmp_path):
        from draf.flow import Flow
        from draf.node.agent import ReActAgent, ToolExec

        d = tmp_path / "analyst"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: analyst\nallowed-tools: [calc]\n---\n\nUse calc.",
            encoding="utf-8",
        )

        flow = Flow("test")
        flow.harness(
            model="gpt-4",
            input_key="query",
            output_key="answer",
            skills=[str(d)],
            use_tools=["calc", "read"],
        )
        g = flow.compile()

        agent = next(n for n in g.nodes.values() if isinstance(n, ReActAgent))
        assert agent.config["skills"] == [str(d)]
        assert agent.config["use_tools"] == ["calc", "read"]

        tool_exec = next(n for n in g.nodes.values() if isinstance(n, ToolExec))
        assert tool_exec.config["skills"] == [str(d)]
        assert tool_exec.config["use_tools"] == ["calc", "read"]


def test_skill_scopes_foreign_mcp_tools(tmp_path):
    """Scoping works over tools described outside the framework (MCP)."""
    pytest.importorskip("mcp")
    from draf.skill import load_skill, scope_tools
    from draf.tool import McpTool
    from mcp.types import Tool as McpToolSpec

    d = tmp_path / "repo-helper"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\n"
        "name: repo-helper\n"
        "allowed-tools: [get_me, search_repositories, get_file_contents]\n"
        "disallowed-tools: [create_issue, create_pull_request]\n"
        "---\n\n"
        "Read-only GitHub assistant.",
        encoding="utf-8",
    )
    skill = load_skill(d)

    specs = [
        McpToolSpec(
            name="get_me",
            description="my profile",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
        McpToolSpec(
            name="search_repositories",
            description="search repos",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
        McpToolSpec(
            name="get_file_contents",
            description="read a file",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
        McpToolSpec(
            name="create_issue",
            description="create an issue",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
    ]
    pool = {s.name: McpTool(object(), s) for s in specs}

    visible = scope_tools(pool, {"use_tools": True}, [skill])
    assert sorted(visible) == ["get_file_contents", "get_me", "search_repositories"]

    # explicit node scope intersects with the skill's allowed-tools
    visible = scope_tools(pool, {"use_tools": ["get_me", "get_file_contents"]}, [skill])
    assert sorted(visible) == ["get_file_contents", "get_me"]


@pytest.mark.asyncio
async def test_react_agent_body_scopes_foreign_tools(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    from draf.node import ExecContext
    from draf.node.agent import ReActAgent
    from draf.tool import McpTool
    from mcp.types import Tool as McpToolSpec

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    d = tmp_path / "repo-helper"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\n"
        "name: repo-helper\n"
        "allowed-tools: [get_me, get_file_contents]\n"
        "disallowed-tools: [create_issue]\n"
        "---\n\n"
        "Read-only GitHub assistant.",
        encoding="utf-8",
    )

    specs = [
        McpToolSpec(
            name="get_me",
            description="me",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
        McpToolSpec(
            name="get_file_contents",
            description="read",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
        McpToolSpec(
            name="create_issue",
            description="create",
            input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
        ),
    ]
    pool = {s.name: McpTool(object(), s) for s in specs}

    captured = {}

    async def mock_post(*a, **kw):
        captured["body"] = kw.get("json")

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "final"}}]}

        return MockResponse()

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    node = ReActAgent(
        {
            "model": "gpt-4",
            "input_key": "query",
            "output_key": "answer",
            "skills": [str(d)],
        }
    )
    ctx = ExecContext(state={}, tools=pool)
    result = await node.execute(ctx, {"query": "hi"})

    assert result["answer"] == "final"
    names = [t["function"]["name"] for t in captured["body"]["tools"]]
    assert sorted(names) == ["get_file_contents", "get_me"]
