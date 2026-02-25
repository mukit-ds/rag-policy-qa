"""
In-memory vector store using numpy cosine similarity.
Handles chunking, embedding via OpenAI, and deduplication.
"""

import os
import hashlib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# In-memory store
_chunks: list[dict] = []
_embeddings: list[list[float]] = []
_seen_hashes: set[str] = set()


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


def ingest_documents(data_dir: str) -> dict:
    """Load, chunk, embed and store all documents. Idempotent."""
    from pathlib import Path

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

    return {
        "status": "ok",
        "documents_processed": docs_processed,
        "chunks_indexed": len(_chunks)
    }


def query_chunks(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve top_k most relevant chunks for a question."""
    if not _chunks:
        return []

    q_emb = _embed([question])[0]
    q_vec = np.array(q_emb, dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)

    store_matrix = np.array(_embeddings, dtype=np.float32)
    norms = np.linalg.norm(store_matrix, axis=1, keepdims=True)
    store_matrix = store_matrix / norms

    scores = store_matrix @ q_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = _chunks[idx].copy()
        chunk["score"] = float(scores[idx])
        results.append(chunk)

    return results