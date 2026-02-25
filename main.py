"""
FastAPI RAG API
Endpoints:
  POST /ingest          — chunk, embed and index all documents in /data
  POST /query           — answer a natural language question with cited sources
  POST /query (stream)  — stream answer tokens via SSE when stream=true
  GET  /ingest/status   — audit log of all ingestion runs
  GET  /health          — health check
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.vector_store import ingest_documents, get_ingest_status
from src.query_engine import answer_question, answer_question_stream

DATA_DIR = "data"


# ─── Lifespan: auto-ingest on startup ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest_documents(DATA_DIR)
    yield


app = FastAPI(
    title="RAG Policy Q&A API",
    description="Retrieval-Augmented Generation over ISO 27001 policy documents.",
    version="1.0.0",
    lifespan=lifespan
)


# ─── Request / Response Models ────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=10)
    stream: bool = False


class SourceItem(BaseModel):
    doc_id: str
    chunk_index: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    answer: str
    sources: list[SourceItem]
    model_used: str
    embedding_model: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Ingest all documents from /data, chunk them, embed via OpenAI,
    and store vectors. Idempotent — safe to call multiple times.
    """
    try:
        result = ingest_documents(DATA_DIR)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: QueryRequest):
    """
    Accept a natural language question and return a grounded answer
    with cited source documents.
    If stream=true, returns a Server-Sent Events (SSE) stream of tokens,
    with a final event: sources block appended at the end.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if request.stream:
            return StreamingResponse(
                answer_question_stream(request.question, top_k=request.top_k),
                media_type="text/event-stream"
            )
        result = answer_question(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status")
def ingest_status():
    """Return audit log of all ingestion runs."""
    return {"runs": get_ingest_status()}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}