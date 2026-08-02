# Sample document for the {{PROJECT_NAME}} RAG catalog.
#
# Files dropped into this folder (`.txt`, `.md`, `.csv`) are indexed
# automatically by the catalog wiring (`src/rag/wiring.py`).  Documents are
# embedded lazily on the first search, so building the catalog never touches
# the network — only an actual query requires a configured embedding provider.

Welcome to the {{PROJECT_NAME}} knowledge base.

This is a sample document used to prove the RAG pipeline works end to end.
Delete it (and replace it with your own files) when you start populating
real content.

The agents can answer questions about anything found here — for example,
they can quote this paragraph back when asked about the sample document.
