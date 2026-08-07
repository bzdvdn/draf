"""Dependencies — the knowledge base shared by all specialists.

One :class:`~service_desk.rag.knowledge.KnowledgeBase` over a durable SQLite vector
store, seeded from the CSVs in ``data/knowledge/``.  Each specialist gets a
domain-scoped tool over the same store.
"""

from __future__ import annotations

from pathlib import Path

from service_desk.rag.knowledge import KnowledgeBase
from teff.rag.embedder import Embedder
from teff.rag.stores import SQLiteVectorStore

#: Vector-store dimension for ``nomic-embed-text`` (the default embedder).
EMBED_DIM = 768

#: Durable SQLite store for the knowledge base (generated, gitignored).
DEFAULT_KB_DB = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "kb.db"

#: (csv, domain, text_column) sources seeded into the knowledge base.
KNOWLEDGE_SOURCES = (
    ("incidents.csv", "incidents", "symptom"),
    ("billing.csv", "billing", "topic"),
    ("deploy.csv", "deploy", "scenario"),
)

_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"


def build_deps(provider: str = "ollama", embedder=None, store=None) -> KnowledgeBase:
    """Build the knowledge base bound to an embedder (default per *provider*).

    Pass *embedder* / *store* to inject stubs in offline tests; otherwise a
    real :class:`~teff.rag.embedder.Embedder` and a durable SQLite store are
    used and documents embed lazily on the first search (never at build time).
    """
    if embedder is None:
        embedder = Embedder(provider=provider)
    if store is None:
        store = SQLiteVectorStore(path=str(DEFAULT_KB_DB), dim=EMBED_DIM)
    knowledge = KnowledgeBase(embedder=embedder, store=store)
    for filename, domain, column in KNOWLEDGE_SOURCES:
        knowledge.add_csv(
            str(_KNOWLEDGE_DIR / filename), domain=domain, text_column=column
        )
    knowledge.resume()
    return knowledge
