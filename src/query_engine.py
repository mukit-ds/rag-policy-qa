import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.vector_store import query_chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{i+1}. {c['doc_id']}, chunk {c['chunk_index']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an information security policy assistant. "
                "Answer only using the context provided below. "
                "Always mention which source document the answer comes from. "
                "If the answer isn't in the context, say so."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]


def _format_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "score": round(c["score"], 4),
            "excerpt": c["text"][:300]
        }
        for c in chunks
    ]


def answer_question(question: str, top_k: int = 5) -> dict:
    chunks = query_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No documents ingested yet. Call POST /ingest first.",
            "sources": [],
            "model_used": COMPLETION_MODEL,
            "embedding_model": EMBEDDING_MODEL
        }

    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=_build_messages(question, chunks),
        temperature=0.0
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": _format_sources(chunks),
        "model_used": COMPLETION_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }


async def answer_question_stream(question: str, top_k: int = 5):
    chunks = query_chunks(question, top_k=top_k)

    if not chunks:
        yield f"data: {json.dumps({'token': 'No documents ingested yet.'})}\n\n"
        yield f"event: sources\ndata: {json.dumps({'sources': []})}\n\n"
        return

    stream = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=_build_messages(question, chunks),
        temperature=0.0,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield f"data: {json.dumps({'token': delta.content})}\n\n"

    yield f"event: sources\ndata: {json.dumps({'sources': _format_sources(chunks), 'model_used': COMPLETION_MODEL, 'embedding_model': EMBEDDING_MODEL})}\n\n"