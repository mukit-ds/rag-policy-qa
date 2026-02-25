# RAG Policy Q&A API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI and OpenAI.
Ingests 20 ISO 27001 security policy documents, indexes them using OpenAI embeddings,
and answers natural language questions with cited, grounded responses.

## Architecture
```
┌─────────────┐     POST /ingest        ┌──────────────────┐     OpenAI API
│   Client    │ ──────────────────────▶ │                  │ ──────────────▶ text-embedding-3-small
│  (curl /    │                         │   FastAPI App    │
│   Swagger)  │     POST /query         │                  │ ──────────────▶ gpt-4o-mini
│             │ ──────────────────────▶ │                  │
│             │     POST /query(stream) └────────┬─────────┘
│             │ ──────────────────────▶          │
└─────────────┘                                  ▼
                                      ┌─────────────────────────┐
                                      │  Hybrid Vector Store    │
                                      │  Dense: OpenAI Embeddings│
                                      │  Sparse: BM25 (Okapi)   │
                                      │  Fusion: RRF            │
                                      └─────────────────────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────────┐
                                      │  SQLite Audit Log       │
                                      │  (ingest_audit.db)      │
                                      └─────────────────────────┘
```

## Setup & Run

### Requirements
- Docker Desktop installed and running
- OpenAI API key

### Steps

1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/rag-policy-qa.git
   cd rag-policy-qa
```

2. Create your `.env` file:
```bash
   cp .env.example .env
```
   Open `.env` and replace `your-openai-api-key-here` with your actual OpenAI API key.

3. Start the full stack:
```bash
   docker compose up --build
```

4. The API is now running at `http://localhost:8000`
5. Open Swagger UI: `http://localhost:8000/docs`

> **Note:** Documents are automatically ingested on startup. No manual `/ingest` call needed.

## API Endpoints

### POST /ingest
Ingests all documents from `/data`, chunks them, embeds via OpenAI, and stores vectors.
Idempotent — safe to call multiple times.

**Response:**
```json
{
  "status": "ok",
  "documents_processed": 20,
  "chunks_indexed": 39
}
```

### POST /query
Accepts a natural language question and returns a grounded answer with cited sources.
Supports optional streaming via SSE when `stream: true`.

**Standard request:**
```json
{
  "question": "How long must audit logs be retained?",
  "top_k": 5
}
```

**Streaming request:**
```json
{
  "question": "How long must audit logs be retained?",
  "top_k": 5,
  "stream": true
}
```

**Standard response:**
```json
{
  "answer": "Audit logs must be retained for a total of 36 months...",
  "sources": [
    {
      "doc_id": "policy_03_audit_logging.txt",
      "chunk_index": 2,
      "score": 0.91,
      "excerpt": "Audit logs must be archived for a total retention period of 36 months..."
    }
  ],
  "model_used": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small"
}
```

**Streaming response (SSE):**
```
data: {"token": "Audit"}
data: {"token": " logs"}
data: {"token": " must"}
...
event: sources
data: {"sources": [...], "model_used": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"}
```

### GET /ingest/status
Returns an audit log of all ingestion runs stored in SQLite.

**Response:**
```json
{
  "runs": [
    {
      "timestamp": "2026-02-25T16:52:08.380813",
      "documents_processed": 20,
      "chunks_indexed": 39,
      "embedding_model": "text-embedding-3-small",
      "estimated_token_cost_usd": 0.000376
    }
  ]
}
```

### GET /health
Health check endpoint.

## Hybrid Search

Retrieval combines **dense** and **sparse** search using **Reciprocal Rank Fusion (RRF)**:

- **Dense retrieval** — OpenAI `text-embedding-3-small` embeddings with cosine similarity
- **Sparse retrieval** — BM25 (Okapi) keyword matching via `rank-bm25`
- **Fusion formula:**
```
RRF_score(d) = Σ 1 / (k + rank(d))
```

Where:
- `k = 60` (standard RRF constant to dampen high rankings)
- `rank(d)` is the 1-based position of document `d` in each ranked list
- Scores from both dense and sparse lists are summed per document
- Final ranking is by descending RRF score

This approach improves recall for both semantic and keyword-heavy queries.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `COMPLETION_MODEL` | `gpt-4o-mini` | OpenAI completion model |
| `TOP_K_DEFAULT` | `5` | Default number of chunks to retrieve |

## Design Decisions

- **Auto-ingest on startup** — `lifespan` event handler runs ingestion automatically so `docker compose up` is truly the only command needed.
- **Hybrid search** — Combines dense (semantic) and sparse (keyword) retrieval via RRF for more robust results across different query types.
- **In-memory vector store** — No external vector DB required. Keeps the stack simple with a single Docker service. Trade-off: vectors are lost on container restart, requiring re-ingestion.
- **Word-based chunking** — 400-word chunks with 80-word overlap balances context preservation and retrieval precision.
- **Idempotent ingestion** — Each chunk is hashed (MD5) to prevent duplicate vectors when `/ingest` is called multiple times.
- **SQLite audit log** — Lightweight persistent log of every ingestion run including token cost estimates.
- **SSE streaming** — Answer tokens are streamed in real time with a final `event: sources` block appended at the end.

## Known Limitations

- Vectors are stored in memory — restarting the container requires re-ingestion (happens automatically on startup).
- No persistent vector storage — a production system should use a vector DB (e.g., Pinecone, Qdrant, pgvector).
- Single container — no horizontal scaling support in current setup.

## Estimated OpenAI Token Usage

| Operation | Model | Estimated Tokens | Estimated Cost |
|---|---|---|---|
| Ingestion (20 docs) | text-embedding-3-small | ~40,000 tokens | ~$0.001 |
| Per query | text-embedding-3-small | ~500 tokens | <$0.0001 |
| Per query | gpt-4o-mini | ~2,000 tokens | ~$0.001 |

**Total for ingestion + 5 evaluation queries: ~$0.01**

## Problem Understanding

The task requires building a production-ready RAG system over 20 ISO 27001 information security policy documents. The core challenge is:
- Accurately retrieving the most relevant policy chunks for a given natural language question
- Generating grounded, cited answers that are traceable back to the source document
- Ensuring the system is fully reproducible with a single `docker compose up` command

## Approach

1. **Document ingestion** — All 20 policy `.txt` files are loaded, split into overlapping 400-word chunks (80-word overlap), and embedded using OpenAI `text-embedding-3-small`
2. **Hybrid retrieval** — Queries are matched using both dense (cosine similarity over OpenAI embeddings) and sparse (BM25 keyword) retrieval, merged via Reciprocal Rank Fusion (RRF)
3. **Answer generation** — Top-k retrieved chunks are passed as context to `gpt-4o-mini` with a strict system prompt to answer only from the provided context and cite sources
4. **Idempotency** — Each chunk is MD5-hashed to prevent duplicate indexing on repeated `/ingest` calls
5. **Auto-ingest** — FastAPI lifespan handler triggers ingestion on startup so no manual steps are needed after `docker compose up`

## Experiments & Results

All 5 evaluation questions from `questions.json` were tested against the API:

| Q# | Question Summary | Expected Source | Answer Correct | Source Correct |
|---|---|---|---|---|
| Q1 | Access review period for standard users | policy_01_access_control.txt | ✅ 180 days | ✅ |
| Q2 | Who declares a security incident? | policy_02_incident_response.txt | ✅ CISO / Incident Commander | ✅ |
| Q3 | Total audit log retention period | policy_03_audit_logging.txt | ✅ 36 months | ✅ |
| Q4 | Encryption standard for data at rest | policy_07_data_encryption_at_rest.txt | ✅ AES-256 via TDE | ✅ |
| Q5 | SLA for Critical vulnerability on Tier 1 | policy_12_patch_sla.txt | ✅ 24 hours | ✅ |

**Result: 5/5 correct answers, 5/5 correct source citations**

## Next Steps

- **Persistent vector store** — Replace in-memory store with a dedicated vector DB (e.g., Qdrant, pgvector) so vectors survive container restarts without re-ingestion
- **Re-ranking** — Add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM`) between retrieval and generation for improved precision
- **Evaluation framework** — Integrate RAGAS or TruLens for automated faithfulness, answer relevance, and context precision scoring
- **Multi-turn conversations** — Add conversation history support for follow-up questions
- **Larger embedding model** — Experiment with `text-embedding-3-large` for improved retrieval accuracy on complex queries


## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `openai` | Embeddings + completions |
| `rank-bm25` | Sparse BM25 retrieval |
| `sse-starlette` | Server-Sent Events streaming |
| `numpy` | Vector operations |
| `python-dotenv` | Environment variable loading |
| `pydantic` | Request/response validation |

