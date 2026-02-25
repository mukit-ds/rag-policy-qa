"""
Query engine: retrieves relevant chunks and generates a grounded answer via OpenAI.
Supports both standard and streaming (SSE) responses.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.vector_store import query_chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Build system and user prompts from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['doc_id']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a precise information security policy assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Always cite which source document your answer comes from. "
        "If the answer is not in the context, say so clearly."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide a clear, concise answer and mention the source document."
    )

    return system_prompt, user_prompt


def _build_sources(chunks: list[dict]) -> list[dict]:
    """Build sources list from retrieved chunks."""
    return [
        {
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "score": round(chunk["score"], 4),
            "excerpt": chunk["text"][:300]
        }
        for chunk in chunks
    ]


def answer_question(question: str, top_k: int = 5) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""
    chunks = query_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Please call POST /ingest first.",
            "sources": [],
            "model_used": COMPLETION_MODEL,
            "embedding_model": EMBEDDING_MODEL
        }

    system_prompt, user_prompt = _build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": _build_sources(chunks),
        "model_used": COMPLETION_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }


async def answer_question_stream(question: str, top_k: int = 5):
    """
    Stream answer tokens via SSE, then emit a final sources event.
    Yields SSE-formatted strings.
    """
    chunks = query_chunks(question, top_k=top_k)

    if not chunks:
        yield f"data: {json.dumps({'token': 'No documents ingested yet.'})}\n\n"
        yield f"event: sources\ndata: {json.dumps({'sources': []})}\n\n"
        return

    system_prompt, user_prompt = _build_prompt(question, chunks)

    # Stream tokens from OpenAI
    stream = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield f"data: {json.dumps({'token': delta.content})}\n\n"

    # Final sources event
    sources = _build_sources(chunks)
    yield f"event: sources\ndata: {json.dumps({'sources': sources, 'model_used': COMPLETION_MODEL, 'embedding_model': EMBEDDING_MODEL})}\n\n"