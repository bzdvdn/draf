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
