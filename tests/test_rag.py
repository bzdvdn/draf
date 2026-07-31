import pytest


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_abc_enforces_contract(self):
        from draf.rag import VectorStore

        with pytest.raises(TypeError):

            class Bad(VectorStore):
                pass

            Bad()

    @pytest.mark.asyncio
    async def test_inmemory_store_and_search(self):
        from draf.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        await store.add([("d1", [1, 0, 0, 0], {"text": "hello"})])
        results = await store.search([1, 0, 0, 0], k=1)
        assert results[0][0] == "d1"
        assert results[0][1] > 0.99

    @pytest.mark.asyncio
    async def test_empty_search_returns_empty(self):
        from draf.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        results = await store.search([1, 0, 0, 0], k=5)
        assert results == []


class TestEmbedder:
    def test_ollama_needs_no_api_key(self):
        from draf.rag import Embedder

        e = Embedder(provider="ollama", model="nomic-embed-text")
        assert e._api_key == ""
        assert e._base_url == "http://localhost:11434/v1"

    def test_openai_requires_api_key(self, monkeypatch):
        from draf.rag import Embedder

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="API key"):
            Embedder(provider="openai")


class TestSQLiteVectorStore:
    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_path):
        from draf.rag.stores import SQLiteVectorStore

        db = str(tmp_path / "v.db")
        s1 = SQLiteVectorStore(path=db, dim=3)
        await s1.add([("d1", [1.0, 0, 0], {"text": "hello"})])

        s2 = SQLiteVectorStore(path=db, dim=3)
        results = await s2.search([1.0, 0, 0], k=1)
        assert results[0][0] == "d1"
        assert results[0][2]["text"] == "hello"
        s1.close()
        s2.close()

    @pytest.mark.asyncio
    async def test_ranked_search(self, tmp_path):
        from draf.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        await s.add(
            [
                ("a", [1.0, 0, 0], {}),
                ("b", [0.0, 1.0, 0], {}),
            ]
        )
        results = await s.search([1.0, 0, 0], k=2)
        assert [r[0] for r in results] == ["a", "b"]
        s.close()

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self, tmp_path):
        from draf.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        with pytest.raises(ValueError, match="dim"):
            await s.add([("d1", [1.0, 0], {})])
        s.close()

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        from draf.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        await s.add([("d1", [1.0, 0, 0], {})])
        await s.delete(["d1"])
        assert await s.search([1.0, 0, 0], k=5) == []
        s.close()

    def test_rag_tool_with_sqlite_store(self, tmp_path):
        from draf.rag import RAGTool

        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {"type": "sqlite", "path": str(tmp_path / "rag.db"), "dim": 8},
                "documents": [{"id": "d1", "text": "hello"}],
            }
        )
        assert type(rag.store).__name__ == "SQLiteVectorStore"

    def test_unknown_store_type_raises(self):
        from draf.rag import RAGTool

        with pytest.raises(ValueError, match="unsupported store type"):
            RAGTool(
                {
                    "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                    "store": {"type": "bogus"},
                }
            )

    def test_external_store_missing_dep_raises(self):
        import importlib.util

        from draf.rag import RAGTool

        deps = {
            "chroma": "chromadb",
            "qdrant": "qdrant_client",
            "pgvector": "asyncpg",
        }
        for stype, dep in deps.items():
            if importlib.util.find_spec(dep) is not None:
                pytest.skip(f"{dep} is installed")
            with pytest.raises(ImportError):
                RAGTool(
                    {
                        "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                        "store": {"type": stype},
                    }
                )

    def test_sqlite_store_from_config(self, tmp_path):
        from draf.rag import RAGTool

        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {
                    "type": "sqlite",
                    "path": str(tmp_path / "rag.db"),
                    "dim": 8,
                },
            }
        )
        assert type(rag.store).__name__ == "SQLiteVectorStore"


class TestRAGTool:
    def test_constructs_with_all_deps(self):
        from draf.rag import RAGTool, Embedder, Chunker
        from draf.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        embedder = Embedder.__new__(Embedder)
        embedder._api_key = "test"
        embedder._base_url = "http://test"
        embedder.provider = "test"
        embedder.model = "test"
        chunker = Chunker(strategy="fixed", chunk_size=50)
        rag = RAGTool(store=store, embedder=embedder, chunker=chunker)
        assert rag.name == "rag"
        assert "search" in rag.description.lower()

    def test_config_with_inline_documents(self, tmp_path):
        from draf.rag import RAGTool

        cfg = {
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": {"type": "in_memory", "dim": 8},
            "documents": [
                {"id": "d1", "topic": "a", "text": "first doc text"},
                {"id": "d2", "topic": "b", "text": "second doc text"},
            ],
        }
        rag = RAGTool(cfg)
        assert rag.store.dim == 8
        assert rag.embedder.provider == "ollama"
        assert rag._documents == [
            ("first doc text", {"id": "d1", "topic": "a"}),
            ("second doc text", {"id": "d2", "topic": "b"}),
        ]

    def test_config_with_csv_file(self, tmp_path):
        from draf.rag import RAGTool

        csv_path = tmp_path / "docs.csv"
        csv_path.write_text('id,topic,text\nd1,a,hello world\nd2,b,"two, words"\n')
        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {"type": "in_memory", "dim": 8},
                "documents": str(csv_path),
            }
        )
        assert rag._documents == [
            ("hello world", {"id": "d1", "topic": "a"}),
            ("two, words", {"id": "d2", "topic": "b"}),
        ]

    def test_config_with_csv_dict(self, tmp_path):
        from draf.rag import RAGTool

        csv_path = tmp_path / "docs.tsv"
        csv_path.write_text("id\tcontent\nd1\thello\n")
        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {"type": "in_memory", "dim": 8},
                "documents": {
                    "file": str(csv_path),
                    "text_column": "content",
                    "delimiter": "\t",
                },
            }
        )
        assert rag._documents == [("hello", {"id": "d1"})]

    def test_lazy_seeding_seeds_once(self, tmp_path):
        from draf.rag import RAGTool
        from draf.rag.stores import InMemoryVectorStore

        class FakeEmbedder:
            async def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0, 0.0]

            async def embed_many(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            documents=[("hello world", {"id": "d1"})],
        )
        import asyncio

        async def go():
            await rag._ensure_seeded()
            await rag._ensure_seeded()

        asyncio.run(go())
        assert len(store._vectors) == 1

    def test_mixed_loader_config(self, tmp_path):
        from draf.rag import RAGTool

        (tmp_path / "a.txt").write_text("first text file")
        (tmp_path / "b.txt").write_text("second text file")
        csv_path = tmp_path / "c.csv"
        csv_path.write_text("id,text\nd1,csv row\n")
        cfg = {
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": {"type": "in_memory", "dim": 8},
            "documents": [
                {"type": "txt", "path": str(tmp_path / "*.txt")},
                {"type": "csv", "path": str(csv_path)},
            ],
        }
        rag = RAGTool(cfg)
        texts = [t for t, _ in rag._documents]
        assert "first text file" in texts
        assert "second text file" in texts
        assert "csv row" in texts

    def test_unknown_document_type_raises(self):
        from draf.rag import RAGTool

        with pytest.raises(ValueError, match="unsupported document type"):
            RAGTool(
                {
                    "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                    "store": {"type": "in_memory", "dim": 8},
                    "documents": [{"type": "nope", "path": "x"}],
                }
            )


class TestDocumentLoaders:
    def test_csv_loader(self, tmp_path):
        from draf.rag.tool import load_documents_csv

        p = tmp_path / "d.csv"
        p.write_text("id,topic,text\nd1,a,hello world\n")
        assert load_documents_csv(str(p)) == [
            ("hello world", {"id": "d1", "topic": "a"})
        ]

    def test_txt_loader_single_and_glob(self, tmp_path):
        from draf.rag.tool import load_documents_txt

        (tmp_path / "one.txt").write_text("first")
        (tmp_path / "two.txt").write_text("second")
        assert load_documents_txt(str(tmp_path / "one.txt")) == [
            ("first", {"id": "one", "path": str(tmp_path / "one.txt")})
        ]
        assert len(load_documents_txt(str(tmp_path / "*.txt"))) == 2

    def test_pdf_loader(self, tmp_path):
        from draf.rag.tool import load_documents_pdf

        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        p = tmp_path / "d.pdf"
        with open(p, "wb") as f:
            w.write(f)
        assert load_documents_pdf(str(p)) == []

    def test_excel_loader(self, tmp_path):
        from draf.rag.tool import load_documents_excel

        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["id", "text"])
        ws.append(["e1", "row one"])
        wb.save(tmp_path / "d.xlsx")
        assert load_documents_excel(str(tmp_path / "d.xlsx")) == [
            ("row one", {"id": "e1"})
        ]

    def test_excel_loader_custom_column(self, tmp_path):
        from draf.rag.tool import load_documents_excel

        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["id", "content"])
        ws.append(["e1", "row one"])
        wb.save(tmp_path / "d.xlsx")
        docs = load_documents_excel(str(tmp_path / "d.xlsx"), text_column="content")
        assert docs == [("row one", {"id": "e1"})]

    def test_pdf_without_pypdf_raises_helpful_error(self, monkeypatch):
        from draf.rag.tool import load_documents_pdf

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf" or name.startswith("pypdf."):
                raise ImportError("no pypdf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="rag-pdf"):
            load_documents_pdf("whatever.pdf")
