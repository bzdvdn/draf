CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS draf_vectors (
    doc_id text,
    embedding vector,
    metadata jsonb
);
