"""Tests for the ``rag_ingest`` write tool and the ``teff chat`` terminal loop."""

import textwrap

import pytest

from teff.tool.registry import default_tool_registry


class _FakeEmbedder:
    dim = 4

    async def embed_many(self, chunks):
        return [[0.1] * self.dim for _ in chunks]

    async def embed(self, text):
        return [0.1] * self.dim


def _make_tool(**config):
    import teff.tool.builtin  # noqa: F401 — registers rag_ingest

    cfg = {
        "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
        "store": {"type": "in_memory", "dim": 4},
    }
    cfg.update(config)
    tool = default_tool_registry.create("rag_ingest", cfg)
    tool.embedder = _FakeEmbedder()
    return tool


class TestRAGIngestTool:
    def test_registered(self):
        import teff.tool.builtin  # noqa: F401

        assert "rag_ingest" in default_tool_registry.list()

    async def test_ingest_text_then_search(self):
        from teff.rag.tool import RAGTool

        tool = _make_tool()
        out = await tool.arun(text="Ванна чугунная 170 см в наличии", source_id="p1")
        assert "ingested 1 document" in out

        rag = RAGTool(store=tool.store, embedder=_FakeEmbedder())
        result = await rag.arun(query="чугунная ванна")
        assert "чугунная" in result.lower()

    async def test_ingest_requires_text_or_path(self):
        tool = _make_tool()
        with pytest.raises(ValueError, match="text.*path"):
            await tool.arun()

    async def test_ingest_csv_file(self, tmp_path):
        p = tmp_path / "items.csv"
        p.write_text(
            "name,desc\nВанна,Чугунная 170 см\nРаковина,Керамическая 60 см\n",
            encoding="utf-8",
        )
        tool = _make_tool(type="csv", text_column="desc")
        out = await tool.arun(path=str(p))
        assert "2 document" in out

    async def test_metadata_passed_through(self):
        tool = _make_tool()
        await tool.arun(text="some text", metadata={"topic": "catalog"})
        chunks = await tool.store.search([0.1] * 4, k=5)
        assert any("catalog" in (m or {}).values() for _, _, m in chunks)


class TestChatCommand:
    def test_chat_echoes_reply(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from teff.cli import app

        p = tmp_path / "chat.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: chat-test
                state:
                  schema:
                    messages: {{reducer: append, type: list}}
                  initial:
                    messages: []
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                checkpoint:
                  type: file
                  path: {cp}
                steps:
                  - id: reply
                    type: llm_chat
                    config:
                      system: "Reply with one word."
                      model: llama3.1:8b
                      provider: ollama
                      output_key: answer
                      messages_key: messages
                edges: []
                """
            ).format(cp=tmp_path / "cp"),
            encoding="utf-8",
        )

        import teff.testing  # noqa: F401 — registers the mock_llm fixture
        from teff.harness.loop import Harness
        from teff.testing import MockLLM

        mock = MockLLM(content="ok")
        monkeypatch.setattr(Harness, "_post", mock._post)
        monkeypatch.setattr(Harness, "_post_stream", mock._post_stream)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["chat", str(p), "--session", "s1"],
            input="hello\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "ok" in result.output
