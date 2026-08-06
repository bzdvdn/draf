"""Tests for the external vector-store backends.

Every backend of ``draf[embedding]`` must be verified, not just the stdlib
ones.  Engines that can run fully offline — FAISS, LanceDB, ChromaDB, and
Qdrant's embedded ``:memory:`` mode — are exercised against the real
library.  Managed/remote services (Milvus, Weaviate, Pinecone) and
PostgreSQL run against a lightweight in-memory fake of their client, so the
adapter code we ship — filter translation, ID handling, response shaping —
is exercised deterministically in CI without provisioning a live server.
"""

import json
import re

import pytest

from draf.rag.base import cosine_similarity

DIM = 3


async def _noop(*a, **k):
    return None


def _feed():
    return [
        ("d1", [1.0, 0, 0], {"text": "first", "cat": "x"}),
        ("d2", [0.0, 1.0, 0.0], {"text": "second", "cat": "y"}),
        ("d3", [0.0, 0.0, 1.0], {"text": "third", "cat": "x"}),
    ]


async def _search(store, query=(1.0, 0, 0), k=3, **kw):
    """Return the sorted result IDs for a store."""
    return [r[0] for r in await store.search(list(query), k=k, **kw)]


def _expect(ids):
    assert ids[0] == "d1"  # d1 is the nearest to [1,0,0]
    assert set(ids) <= {"d1", "d2", "d3"}


# ---------------------------------------------------------------------------
# Real, offline engines
# ---------------------------------------------------------------------------


class TestFaissStore:
    @pytest.mark.asyncio
    async def test_roundtrip_ranked(self):
        pytest.importorskip("faiss")
        from draf.rag.stores import FAISSVectorStore

        s = FAISSVectorStore(dim=DIM)
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))
        assert await s.get(["d1"]) == [("d1", _feed()[0][2])]

    @pytest.mark.asyncio
    async def test_delete_and_entries(self):
        pytest.importorskip("faiss")
        from draf.rag.stores import FAISSVectorStore

        s = FAISSVectorStore(dim=DIM)
        await s.add(_feed())
        await s.delete(["d2"])
        assert await s.count() == 2
        assert [i for i, _ in await s.entries()] == ["d1", "d3"]

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self):
        pytest.importorskip("faiss")
        from draf.rag.stores import FAISSVectorStore

        s = FAISSVectorStore(dim=DIM)
        with pytest.raises(ValueError, match="dim"):
            await s.add([("bad", [1.0, 0], {})])


class TestLanceStore:
    @pytest.mark.asyncio
    async def test_roundtrip(self, tmp_path):
        pytest.importorskip("lancedb")
        from draf.rag.stores import LanceVectorStore

        s = LanceVectorStore(path=str(tmp_path / "l"), dim=DIM)
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))
        assert await s.get(["d1"]) == [("d1", _feed()[0][2])]

    @pytest.mark.asyncio
    async def test_filter(self, tmp_path):
        pytest.importorskip("lancedb")
        from draf.rag.stores import LanceVectorStore

        s = LanceVectorStore(path=str(tmp_path / "l"), dim=DIM)
        await s.add(_feed())
        assert set(await _search(s, filter={"cat": "x"})) == {"d1", "d3"}


class TestChromaStore:
    @pytest.mark.asyncio
    async def test_roundtrip(self, tmp_path):
        pytest.importorskip("chromadb")
        from draf.rag.stores import ChromaVectorStore

        s = ChromaVectorStore(path=str(tmp_path / "c"), collection="draf_coll")
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))
        assert await s.get(["d1"]) == [("d1", _feed()[0][2])]


class TestQdrantStore:
    @pytest.mark.asyncio
    async def test_roundtrip_embedded(self):
        qc = pytest.importorskip("qdrant_client")
        from draf.rag.stores import QdrantVectorStore

        s = QdrantVectorStore(client=qc.QdrantClient(":memory:"), collection="qd")
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))

    @pytest.mark.asyncio
    async def test_delete(self):
        qc = pytest.importorskip("qdrant_client")
        from draf.rag.stores import QdrantVectorStore

        s = QdrantVectorStore(client=qc.QdrantClient(":memory:"), collection="qd")
        await s.add(_feed())
        await s.delete(["d2"])
        assert await s.count() == 2


# ---------------------------------------------------------------------------
# Managed / remote engines — deterministic fake clients
# ---------------------------------------------------------------------------


# --- Pinecone fake ----------------------------------------------------------


def _pinecone_filter_ok(filter, meta):
    if not filter:
        return True
    for k, cond in filter.items():
        if k in ("$and", "$or"):
            subs = [cond] if not isinstance(cond, list) else cond
            fn = all if k == "$and" else any
            if not fn(_pinecone_filter_ok(sub, meta) for sub in subs):
                return False
        elif isinstance(cond, dict) and "$in" in cond:
            if meta.get(k) not in cond["$in"]:
                return False
        elif meta.get(k) != cond:
            return False
    return True


class _PineconeMatch:
    __slots__ = ("id", "score", "metadata")

    def __init__(self, id, score, metadata):
        self.id = id
        self.score = score
        self.metadata = metadata


class _PineconeQueryResult:
    __slots__ = ("matches",)

    def __init__(self, matches):
        self.matches = matches


class FakePineconeIndex:
    def __init__(self):
        self._rows = {}  # (namespace, vid) -> (vec, meta)

    def upsert(self, vectors, namespace=""):
        for vid, vec, meta in vectors:
            self._rows[(namespace, vid)] = (vec, meta)

    def describe_index_stats(self):
        return type("S", (), {"total_vector_count": len(self._rows)})()

    def query(self, vector, top_k, include_metadata=True, namespace="", filter=None):
        scored = [
            (cosine_similarity(vector, vec), vid, meta)
            for (ns, vid), (vec, meta) in self._rows.items()
            if ns == namespace and _pinecone_filter_ok(filter, meta)
        ]
        scored.sort(key=lambda t: -t[0])
        return _PineconeQueryResult(
            [_PineconeMatch(vid, sc, meta) for sc, vid, meta in scored[:top_k]]
        )

    def fetch(self, ids, namespace=""):
        got = {}
        for vid in ids:
            if (namespace, vid) in self._rows:
                got[vid] = _PineconeMatch(vid, 1.0, self._rows[(namespace, vid)][1])
        return type("F", (), {"vectors": got})()

    def delete(self, ids, namespace=""):
        for vid in ids:
            self._rows.pop((namespace, vid), None)

    def update(self, id, set_metadata, namespace=""):
        if (namespace, id) in self._rows:
            vec, meta = self._rows[(namespace, id)]
            self._rows[(namespace, id)] = (vec, {**meta, **set_metadata})


class TestPinecone:
    @pytest.fixture
    def pc(self, monkeypatch):
        pytest.importorskip("pinecone")
        import pinecone

        index = FakePineconeIndex()

        class FakePC:
            def __init__(self, *a, **k):
                self._index = index

            def Index(self, name, host=""):
                return index

        monkeypatch.setitem(pinecone.__dict__, "Pinecone", FakePC)
        return index

    @pytest.mark.asyncio
    async def test_roundtrip(self, pc):
        from draf.rag.stores import PineconeVectorStore

        s = PineconeVectorStore(api_key="test", namespace="n")
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))
        assert await s.get(["d1"]) == [("d1", _feed()[0][2])]
        await s.delete(["d2"])
        assert await s.count() == 2

    @pytest.mark.asyncio
    async def test_filter(self, pc):
        from draf.rag.stores import PineconeVectorStore

        s = PineconeVectorStore(api_key="test", namespace="n")
        await s.add(_feed())
        assert set(await _search(s, filter={"cat": "x"})) == {"d1", "d3"}


# --- Milvus fake ------------------------------------------------------------


def _milvus_expr_ok(expr: str, row) -> bool:
    """Return True when ``row`` satisfies a Milvus filter expression fragment.

    Tests only rely on ``id in [...]``; anything else is treated as matching.
    """
    if not expr:
        return True
    ids = re.findall(r"id in \[(.*?)\]", expr)
    if ids:
        wanted = json.loads("[" + ids[0] + "]")
        return row["id"] in wanted
    return True


class _MilvusSchema:
    def __init__(self):
        self._fields = []

    def add_field(self, *a, **k):
        self._fields.append((a, k))


class _MilvusIndexParams:
    def add_index(self, **k):
        pass


class FakeMilvusClient:
    def __init__(self, uri=None, token=None):
        self._uri = uri
        self._rows = {}  # collection -> {vid: (vec, meta)}
        self._created = set()

    def has_collection(self, name):
        return name in self._created

    @staticmethod
    def create_schema(auto_id=False, enable_dynamic_field=True):
        return _MilvusSchema()

    @staticmethod
    def prepare_index_params():
        return _MilvusIndexParams()

    def create_collection(self, name, schema=None, index_params=None):
        self._created.add(name)
        self._rows.setdefault(name, {})

    def insert(self, collection, rows):
        store = self._rows.setdefault(collection, {})
        for row in rows:
            store[row["id"]] = (
                row["vector"],
                {k: v for k, v in row.items() if k not in ("id", "vector")},
            )

    def search(self, collection, data, limit, output_fields=None, filter=""):
        query = data[0]
        scored = [
            (cosine_similarity(query, vec), vid, meta)
            for vid, (vec, meta) in self._rows.get(collection, {}).items()
        ]
        scored.sort(key=lambda t: -t[0])
        return [
            [
                {"id": vid, "distance": sc, "entity": meta}
                for sc, vid, meta in scored[:limit]
            ]
        ]

    def delete(self, collection, filter=""):
        store = self._rows.get(collection, {})
        to_drop = [vid for vid in store if _milvus_expr_ok(filter, {"id": vid})]
        for vid in to_drop:
            store.pop(vid, None)

    def get_collection_stats(self, collection):
        return {"row_count": len(self._rows.get(collection, {}))}

    def query(self, collection, filter="", output_fields=None, limit=100, offset=0):
        store = self._rows.get(collection, {})
        rows = [
            {"id": vid, "vector": vec, **meta}
            for vid, (vec, meta) in sorted(store.items())
            if _milvus_expr_ok(filter, {"id": vid})
        ]
        return rows[offset : offset + limit]

    def upsert(self, collection, rows):
        store = self._rows.setdefault(collection, {})
        for row in rows:
            store[row["id"]] = (
                row["vector"],
                {k: v for k, v in row.items() if k not in ("id", "vector")},
            )

    def drop_collection(self, name):
        self._created.discard(name)
        self._rows.pop(name, None)


class TestMilvus:
    @pytest.mark.asyncio
    async def test_roundtrip(self, monkeypatch):
        pytest.importorskip("pymilvus")
        import pymilvus

        monkeypatch.setattr(pymilvus, "MilvusClient", FakeMilvusClient)
        monkeypatch.setattr(
            pymilvus, "DataType", type("Dt", (), {"VARCHAR": 1, "FLOAT_VECTOR": 2})
        )

        from draf.rag.stores import MilvusVectorStore

        s = MilvusVectorStore(uri="./tmp/db", collection="coll", dim=DIM)
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))
        assert await s.get(["d1"]) == [("d1", _feed()[0][2])]
        await s.delete(["d2"])
        assert await s.count() == 2


# --- Weaviate fake ----------------------------------------------------------


class _WeaviateObj:
    __slots__ = ("uuid", "properties", "metadata")

    def __init__(self, uuid, properties, distance):
        self.uuid = uuid
        self.properties = properties
        self.metadata = type("M", (), {"distance": distance})()


class _WeaviateCollection:
    def __init__(self, fake, name):
        self._fake = fake
        self._name = name

    @property
    def batch(self):
        return _WeaviateBatch(self)

    def add(self, vid, props, vector):
        self._fake._rows.setdefault(self._name, {})[vid] = (props, vector)

    @property
    def query(self):
        return self

    def near_vector(self, near_vector, limit, return_metadata=None):
        self._fake._query = near_vector
        self._fake._limit = limit
        return self

    @property
    def objects(self):
        query = self._fake._query
        scored = [
            (cosine_similarity(query, vec), vid, props)
            for vid, (props, vec) in self._fake._rows.get(self._name, {}).items()
        ]
        scored.sort(key=lambda t: -t[0])
        return [
            _WeaviateObj(vid, props, 1.0 - sc)
            for sc, vid, props in scored[: self._fake._limit]
        ]

    @property
    def aggregate(self):
        return self

    def over_all(self, **kw):
        self._fake._count = len(self._fake._rows.get(self._name, {}))
        return self

    @property
    def total_count(self):
        return self._fake._count


class _WeaviateBatch:
    def __init__(self, collection):
        self._coll = collection

    def fixed_size(self, n):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def add_object(self, uuid, properties, vector):
        self._coll.add(properties["doc_id"], properties, vector)


class _WeaviateCollectionsFacade:
    def __init__(self, fake):
        self._fake = fake

    def exists(self, name):
        return name in self._fake._collections

    def create(self, name, **kwargs):
        self._fake._collections.add(name)
        self._fake._rows.setdefault(name, {})

    def get(self, name):
        return _WeaviateCollection(self._fake, name)

    def delete(self, name):
        self._fake._collections.discard(name)
        self._fake._rows.pop(name, None)


class FakeWeaviate:
    def __init__(self):
        self._collections = set()
        self._rows = {}
        self._query = None
        self._limit = 3
        self._count = 0
        self.collections = _WeaviateCollectionsFacade(self)


class TestWeaviate:
    @pytest.mark.asyncio
    async def test_roundtrip(self):
        pytest.importorskip("weaviate")
        from draf.rag.stores import WeaviateVectorStore

        s = WeaviateVectorStore(client=FakeWeaviate(), collection="wcv")
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))


# --- PostgreSQL fake ----------------------------------------------------------


class _FakePgConn:
    def __init__(self):
        self._rows = {}  # doc_id -> (vec, meta)

    async def execute(self, sql, *args):
        if "ANY($1)" in sql:
            for vid in args[0]:
                self._rows.pop(vid, None)
        elif sql.startswith("DELETE"):
            self._rows.clear()
        return None

    async def executemany(self, sql, rows):
        for vid, vec, meta in rows:
            self._rows[vid] = (vec, json.loads(meta))

    async def fetch(self, sql, *args):
        if "ANY($1)" in sql:
            return [
                {"doc_id": vid, "metadata": json.dumps(meta, ensure_ascii=False)}
                for vid, (vec, meta) in self._rows.items()
                if vid in args[0]
            ]
        query = args[0]
        scored = [
            (cosine_similarity(query, vec), vid, meta)
            for vid, (vec, meta) in self._rows.items()
        ]
        scored.sort(key=lambda t: -t[0])
        return [
            {
                "doc_id": vid,
                "score": sc,
                "metadata": json.dumps(meta, ensure_ascii=False),
            }
            for sc, vid, meta in scored
        ]

    async def fetchval(self, sql, *args):
        return len(self._rows)

    async def fetchrow(self, sql, *args):
        if args[0] not in self._rows:
            return None
        return {"metadata": json.dumps(self._rows[args[0]][1], ensure_ascii=False)}

    async def close(self):
        pass


class TestPGVectorStore:
    @pytest.mark.asyncio
    async def test_roundtrip(self, monkeypatch):
        pytest.importorskip("asyncpg")
        import asyncpg
        import pgvector.asyncpg  # noqa: F401  ensure submodule attributes are registered

        conn = _FakePgConn()

        async def fake_connect(dsn, *a, **k):
            return conn

        monkeypatch.setattr(asyncpg, "connect", fake_connect)
        monkeypatch.setattr(pgvector.asyncpg, "register_vector", _noop)

        from draf.rag.stores import PGVectorStore

        s = PGVectorStore(dsn="postgresql://mock/mock", dim=DIM)
        await s.add(_feed())
        assert await s.count() == 3
        _expect(await _search(s))
        assert await s.get(["d1"]) == [("d1", _feed()[0][2])]
        await s.delete(["d2"])
        assert await s.count() == 2
