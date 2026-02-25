from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.vector_store import ingest_documents, get_ingest_status
from src.query_engine import answer_question, answer_question_stream

DATA_DIR = "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # kick off ingestion when the server starts so /query works immediately
    ingest_documents(DATA_DIR)
    yield


app = FastAPI(
    title="RAG Policy Q&A API",
    description="Q&A over ISO 27001 policy documents using OpenAI embeddings and GPT.",
    version="1.0.0",
    lifespan=lifespan
)


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


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Load all .txt files from /data, chunk and embed them, then store in memory.
    Safe to call multiple times — duplicate chunks are skipped.
    """
    try:
        return ingest_documents(DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: QueryRequest):
    """
    Takes a question, finds the most relevant policy chunks, and returns
    a grounded answer with source citations. Pass stream=true for SSE output.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        if request.stream:
            return StreamingResponse(
                answer_question_stream(request.question, top_k=request.top_k),
                media_type="text/event-stream"
            )
        return answer_question(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status")
def ingest_status():
    """Shows a history of every ingest run with token cost estimates."""
    return {"runs": get_ingest_status()}


@app.get("/health")
def health():
    return {"status": "ok"}