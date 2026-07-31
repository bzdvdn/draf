"""Chroma vector store — requires ``chromadb``."""

from draf.rag.base import VectorStore


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB (persistent).

    Requires the ``chromadb`` package (install via ``draf[embedding]``).
    """

    def __init__(self, path: str = "./chroma", collection: str = "draf"):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("install chromadb for ChromaVectorStore") from e
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection)

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        ids = [v[0] for v in vectors]
        embeddings = [v[1] for v in vectors]
        metadatas = [v[2] for v in vectors]
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    async def search(
        self, query: list[float], k: int = 10
    ) -> list[tuple[str, float, dict]]:
        results = self._collection.query(query_embeddings=[query], n_results=k)
        out = []
        for i in range(len(results["ids"][0])):
            out.append(
                (
                    results["ids"][0][i],
                    results["distances"][0][i] if results.get("distances") else 0.0,
                    results["metadatas"][0][i] if results.get("metadatas") else {},
                )
            )
        return out

    async def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
