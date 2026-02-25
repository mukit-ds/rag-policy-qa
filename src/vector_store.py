"""
In-memory vector store using numpy cosine similarity.
Hybrid search: dense (OpenAI embeddings) + sparse (BM25) with Reciprocal Rank Fusion.
Includes ingestion audit log stored in SQLite.
"""

import os
import hashlib
import sqlite3
import datetime
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# In-memory store
_chunks: list[dict] = []
_embeddings: list[list[float]] = []
_seen_hashes: set[str] = set()
_bm25: BM25Okapi = None

# SQLite audit log
DB_PATH = "ingest_audit.db"


def _init_db():
    """Initialize SQLite audit log table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            documents_processed INTEGER,
            chunks_indexed INTEGER,
            embedding_model TEXT,
            estimated_token_cost REAL
        )
    """)
    conn.commit()
    conn.close()


def _log_ingest(documents_processed: int, chunks_indexed: int, tokens_used: int):
    """Write one audit row to SQLite."""
    cost = round(tokens_used * 0.00000002, 6)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO ingest_log (timestamp, documents_processed, chunks_indexed, embedding_model, estimated_token_cost)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.datetime.utcnow().isoformat(),
        documents_processed,
        chunks_indexed,
        EMBEDDING_MODEL,
        cost
    ))
    conn.commit()
    conn.close()


def get_ingest_status() -> list[dict]:
    """Return all audit log entries."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp, documents_processed, chunks_indexed, embedding_model, estimated_token_cost
        FROM ingest_log ORDER BY id DESC
    """).fetchall()
    conn.close()
    return [
        {
            "timestamp": r[0],
            "documents_processed": r[1],
            "chunks_indexed": r[2],
            "embedding_model": r[3],
            "estimated_token_cost_usd": r[4]
        }
        for r in rows
    ]


def _chunk_text(text: str, doc_id: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_hash = hashlib.md5(f"{doc_id}_{idx}_{chunk_text[:50]}".encode()).hexdigest()
        chunks.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk_text,
            "hash": chunk_hash
        })
        idx += 1
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI."""
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]


def _build_bm25():
    """Build BM25 index from current chunks."""
    global _bm25
    tokenized = [c["text"].lower().split() for c in _chunks]
    _bm25 = BM25Okapi(tokenized)


def ingest_documents(data_dir: str) -> dict:
    """Load, chunk, embed and store all documents. Idempotent."""
    from pathlib import Path
    global _bm25
    _init_db()

    all_new_chunks = []
    docs_processed = 0

    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        doc_chunks = _chunk_text(text, doc_id=path.name)
        new_chunks = [c for c in doc_chunks if c["hash"] not in _seen_hashes]
        if new_chunks:
            all_new_chunks.extend(new_chunks)
            docs_processed += 1

    if not all_new_chunks:
        return {
            "status": "ok",
            "documents_processed": 0,
            "chunks_indexed": len(_chunks)
        }

    # Embed in batches of 100
    texts = [c["text"] for c in all_new_chunks]
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(_embed(batch))

    for chunk, emb in zip(all_new_chunks, embeddings):
        _chunks.append(chunk)
        _embeddings.append(emb)
        _seen_hashes.add(chunk["hash"])

    # Rebuild BM25 index
    _build_bm25()

    # Estimate tokens
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars // 4
    _log_ingest(docs_processed, len(_chunks), estimated_tokens)

    return {
        "status": "ok",
        "documents_processed": docs_processed,
        "chunks_indexed": len(_chunks)
    }


def query_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: dense (OpenAI embeddings) + sparse (BM25).
    Merged using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum(1 / (k + rank(d)))
    where k=60 (standard constant), rank is 1-based position in each list.
    """
    if not _chunks:
        return []

    n = len(_chunks)
    k = 60  # RRF constant

    # ── Dense retrieval ──────────────────────────────────────────
    q_emb = _embed([question])[0]
    q_vec = np.array(q_emb, dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)

    store_matrix = np.array(_embeddings, dtype=np.float32)
    norms = np.linalg.norm(store_matrix, axis=1, keepdims=True)
    store_matrix = store_matrix / norms
    dense_scores = store_matrix @ q_vec
    dense_ranking = np.argsort(dense_scores)[::-1]

    # ── Sparse retrieval (BM25) ───────────────────────────────────
    tokenized_query = question.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    sparse_ranking = np.argsort(bm25_scores)[::-1]

    # ── Reciprocal Rank Fusion ────────────────────────────────────
    rrf_scores = np.zeros(n)
    for rank, idx in enumerate(dense_ranking):
        rrf_scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ranking):
        rrf_scores[idx] += 1.0 / (k + rank + 1)

    top_indices = np.argsort(rrf_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = _chunks[idx].copy()
        chunk["score"] = float(rrf_scores[idx])
        results.append(chunk)

    return results