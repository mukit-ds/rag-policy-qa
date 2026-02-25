# RAG Policy Q&A API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI and OpenAI.
Ingests 20 ISO 27001 security policy documents, indexes them using OpenAI embeddings,
and answers natural language questions with cited, grounded responses.

## Architecture
```
┌─────────────┐     POST /ingest      ┌──────────────────┐     OpenAI API
│   Client    │ ───────────────────▶  │                  │ ──────────────▶ text-embedding-3-small
│  (curl /    │                       │   FastAPI App    │
│   Swagger)  │     POST /query       │                  │ ──────────────▶ gpt-4o-mini
│             │ ───────────────────▶  │                  │
└─────────────┘                       └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  In-Memory Vector    │
                                    │  Store (numpy cosine │
                                    │  similarity)         │
                                    └──────────────────────┘
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

**Request:**
```json
{
  "question": "How long must audit logs be retained?",
  "top_k": 5
}
```

**Response:**
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

### GET /health
Health check endpoint.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `COMPLETION_MODEL` | `gpt-4o-mini` | OpenAI completion model |
| `TOP_K_DEFAULT` | `5` | Default number of chunks to retrieve |

## Design Decisions

- **In-memory vector store** — No external vector DB required. Keeps the stack simple with a single Docker service. Trade-off: vectors are lost on container restart, requiring a fresh `/ingest` call.
- **Word-based chunking** — 400-word chunks with 80-word overlap balances context preservation and retrieval precision.
- **Idempotent ingestion** — Each chunk is hashed (MD5) to prevent duplicate vectors when `/ingest` is called multiple times.
- **Cosine similarity** — Vectors are L2-normalized and compared via dot product for efficient similarity scoring.

## Known Limitations

- Vectors are stored in memory — restarting the container requires calling `/ingest` again.
- No persistent storage — a production system should use a vector DB (e.g., Pinecone, Qdrant, pgvector).
- Single container — no horizontal scaling support in current setup.

## Estimated OpenAI Token Usage

| Operation | Model | Estimated Tokens | Estimated Cost |
|---|---|---|---|
| Ingestion (20 docs) | text-embedding-3-small | ~40,000 tokens | ~$0.001 |
| Per query | text-embedding-3-small | ~500 tokens | <$0.0001 |
| Per query | gpt-4o-mini | ~2,000 tokens | ~$0.001 |

**Total for ingestion + 5 evaluation queries: ~$0.01**