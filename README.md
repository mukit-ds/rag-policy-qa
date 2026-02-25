# RAG Policy Q&A API

FastAPI service that answers natural language questions over a corpus of ISO 27001 security policy documents. Uses OpenAI embeddings for semantic search and GPT for answer generation, with hybrid BM25 retrieval on top.

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

## Getting Started

### Requirements
- Docker Desktop
- OpenAI API key

### Steps

1. Clone the repo:
```bash
git clone https://github.com/mukit-ds/rag-policy-qa.git
cd rag-policy-qa
```

2. Set up your env file:
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key.

3. Start the server:
```bash
docker compose up --build
```

The API runs on `http://localhost:8000`. Documents are ingested automatically on startup — no extra steps needed.

Swagger UI is available at `http://localhost:8000/docs`.

## Endpoints

### POST /ingest
Loads all 20 policy files from `/data`, splits them into chunks, embeds via OpenAI and stores in memory. Calling this multiple times is safe — already-indexed chunks are skipped.

```json
{
  "status": "ok",
  "documents_processed": 20,
  "chunks_indexed": 39
}
```

### POST /query
Retrieves the most relevant chunks for a question and returns a grounded answer with source citations. `top_k` controls how many chunks to retrieve (1–10, default 5).

```json
{
  "question": "How long must audit logs be retained?",
  "top_k": 5
}
```

Response:
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

Pass `"stream": true` to get a token-by-token SSE stream instead:
```
data: {"token": "Audit"}
data: {"token": " logs"}
...
event: sources
data: {"sources": [...], "model_used": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"}
```

### GET /ingest/status
Returns a log of every ingest run — useful for tracking token usage over time.

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
Basic health check.

## Hybrid Search

Retrieval combines dense and sparse search via Reciprocal Rank Fusion (RRF):

- **Dense** — cosine similarity over OpenAI `text-embedding-3-small` embeddings
- **Sparse** — BM25 (Okapi) keyword scoring via `rank-bm25`
- **Fusion:**

```
RRF_score(d) = Σ 1 / (k + rank(d))
```

`k = 60` is the standard dampening constant. Each document gets scores from both lists summed together, then results are sorted by the combined RRF score. This helps with queries that are either keyword-heavy or more semantic in nature.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `COMPLETION_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `TOP_K_DEFAULT` | `5` | Default chunks to retrieve |

## Design Notes

I went with an in-memory vector store to keep the stack simple — no external dependencies means `docker compose up` really is the only command needed. The trade-off is that vectors are rebuilt on each container start, but since ingestion runs automatically at startup this is transparent to the caller.

Chunks are 400 words with 80-word overlap. I tried smaller chunk sizes but found the context window ended up too narrow for some of the denser policy sections. MD5 hashing on chunk content ensures the `/ingest` endpoint is idempotent.

For retrieval, pure semantic search worked well on most queries but struggled with exact term lookups (policy IDs, specific role names). Adding BM25 alongside the embeddings and merging with RRF improved results on those cases noticeably.

## Known Limitations

- Vectors live in memory, so a container restart triggers re-ingestion. In production I'd swap this out for something like Qdrant or pgvector.
- No conversation history — each query is independent. Follow-up questions don't have context from previous turns.
- Single container setup, no load balancing.

## Problem Understanding

The goal was to build a Q&A system that can accurately retrieve answers from 20 security policy documents and return grounded, cited responses. The main challenges were getting retrieval right across both semantic and keyword-heavy queries, and making sure the system starts cleanly with no manual setup.

## Approach

Documents are split into overlapping word-based chunks and embedded using `text-embedding-3-small`. At query time, both dense (cosine similarity) and sparse (BM25) retrieval are run and merged using RRF. The top-k chunks are passed to `gpt-4o-mini` as context with a prompt that instructs it to answer only from the provided text and cite its source.

Ingestion is triggered automatically via FastAPI's lifespan handler so the API is ready to answer queries immediately after `docker compose up`.

## Results

Tested against all 5 questions in `questions.json`:

| # | Question | Source | Answer |
|---|---|---|---|
| Q1 | Access review period for standard users | policy_01_access_control.txt | 180 days |
| Q2 | Who declares a security incident? | policy_02_incident_response.txt | CISO / Incident Commander |
| Q3 | Total audit log retention | policy_03_audit_logging.txt | 36 months |
| Q4 | Encryption standard for data at rest | policy_07_data_encryption_at_rest.txt | AES-256 via TDE |
| Q5 | SLA for Critical vuln on Tier 1 | policy_12_patch_sla.txt | 24 hours |

All 5 returned the correct answer citing the correct source document.

## Next Steps

- Swap in-memory store for a persistent vector DB (Qdrant or pgvector) so restarts don't require re-ingestion
- Add a cross-encoder re-ranker after retrieval to improve precision on ambiguous queries
- Add conversation history support for multi-turn Q&A
- Experiment with `text-embedding-3-large` on harder queries
- Plug in RAGAS for automated evaluation of retrieval and answer quality

## Estimated OpenAI Token Usage

| Operation | Model | Tokens | Cost |
|---|---|---|---|
| Ingestion (20 docs) | text-embedding-3-small | ~40,000 | ~$0.001 |
| Per query | text-embedding-3-small | ~500 | <$0.0001 |
| Per query | gpt-4o-mini | ~2,000 | ~$0.001 |

Total for ingestion + 5 evaluation queries: roughly $0.01.

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `openai` | Embeddings and completions |
| `rank-bm25` | BM25 sparse retrieval |
| `sse-starlette` | SSE streaming |
| `numpy` | Vector math |
| `python-dotenv` | Env var loading |
| `pydantic` | Request/response validation |
