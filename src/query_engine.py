"""
Query engine: retrieves relevant chunks and generates a grounded answer via OpenAI.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from src.vector_store import query_chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


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

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['doc_id']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    # Build prompt
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

    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    answer_text = response.choices[0].message.content.strip()

    # Build sources list
    sources = [
        {
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "score": round(chunk["score"], 4),
            "excerpt": chunk["text"][:300]
        }
        for chunk in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "model_used": COMPLETION_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }