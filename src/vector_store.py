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

_chunks: list[dict] = []
_embeddings: list[list[float]] = []
_seen_hashes: set[str] = set()
_bm25: BM25Okapi = None

DB_PATH = "ingest_audit.db"


def _init_db():
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
    # $0.02 per 1M tokens for text-embedding-3-small
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
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def _rebuild_bm25():
    global _bm25
    tokenized = [c["text"].lower().split() for c in _chunks]
    _bm25 = BM25Okapi(tokenized)


def ingest_documents(data_dir: str) -> dict:
    from pathlib import Path
    global _bm25
    _init_db()

    new_chunks = []
    docs_processed = 0
    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        doc_chunks = _chunk_text(text, doc_id=path.name)
        unseen = [c for c in doc_chunks if c["hash"] not in _seen_hashes]
        if unseen:
            new_chunks.extend(unseen)
            docs_processed += 1

    if not new_chunks:
        return {"status": "ok", "documents_processed": 0, "chunks_indexed": len(_chunks)}

    texts = [c["text"] for c in new_chunks]
    embeddings = []
    for i in range(0, len(texts), 100):
        embeddings.extend(_embed(texts[i:i + 100]))

    for chunk, emb in zip(new_chunks, embeddings):
        _chunks.append(chunk)
        _embeddings.append(emb)
        _seen_hashes.add(chunk["hash"])

    _rebuild_bm25()

    estimated_tokens = sum(len(t) for t in texts) // 4
    _log_ingest(docs_processed, len(_chunks), estimated_tokens)

    return {"status": "ok", "documents_processed": docs_processed, "chunks_indexed": len(_chunks)}


def query_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval using dense embeddings + BM25, merged with RRF.
    RRF score = sum(1 / (60 + rank)) across both ranked lists.
    """
    if not _chunks:
        return []

    n = len(_chunks)
    k = 60

    # dense retrieval via cosine similarity
    q_vec = np.array(_embed([question])[0], dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)
    store = np.array(_embeddings, dtype=np.float32)
    store /= np.linalg.norm(store, axis=1, keepdims=True)
    dense_scores = store @ q_vec
    dense_rank = np.argsort(dense_scores)[::-1]

    # sparse retrieval via BM25
    bm25_scores = _bm25.get_scores(question.lower().split())
    sparse_rank = np.argsort(bm25_scores)[::-1]

    # reciprocal rank fusion
    rrf = np.zeros(n)
    for rank, idx in enumerate(dense_rank):
        rrf[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_rank):
        rrf[idx] += 1.0 / (k + rank + 1)

    top = np.argsort(rrf)[::-1][:top_k]
    results = []
    for idx in top:
        chunk = _chunks[idx].copy()
        chunk["score"] = float(rrf[idx])
        results.append(chunk)

    return results