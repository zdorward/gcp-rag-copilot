# GCP RAG Copilot - Design Document

**Date:** 2026-02-17
**Status:** Approved
**Author:** Claude (Senior GCP Engineer)

## Overview

A document Q&A copilot deployed on Cloud Run that ingests PDFs/TXT/MD from GCS, creates embeddings via Vertex AI, stores chunks in BigQuery, and answers questions using Gemini with citations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GCP RAG Copilot Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │   Google     │         │   Cloud Run  │         │   Cloud Run  │       │
│   │   Cloud      │────────▶│   Job        │────────▶│   API        │       │
│   │   Storage    │ ingest  │  (Ingestion) │ query   │  (FastAPI)   │       │
│   │   (PDFs/TXT) │         │              │         │              │       │
│   └──────────────┘         └──────┬───────┘         └──────┬───────┘       │
│                                   │                        │               │
│                                   │ embed                  │ embed query   │
│                                   ▼                        ▼               │
│                            ┌──────────────┐         ┌──────────────┐       │
│                            │  Vertex AI   │         │  Vertex AI   │       │
│                            │  Embeddings  │         │  Gemini LLM  │       │
│                            └──────┬───────┘         └──────────────┘       │
│                                   │                        ▲               │
│                                   │ store                  │ retrieve      │
│                                   ▼                        │               │
│                            ┌──────────────────────────────┐│               │
│                            │         BigQuery             ││               │
│                            │  ┌────────────────────────┐  ││               │
│                            │  │ chunks table           │  │┘               │
│                            │  │ - doc_id, chunk_id     │  │                │
│                            │  │ - source_uri, page     │  │                │
│                            │  │ - chunk_text           │  │                │
│                            │  │ - embedding ARRAY<F64> │  │                │
│                            │  │ - created_at           │  │                │
│                            │  └────────────────────────┘  │                │
│                            └──────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **GCS Bucket** - Source documents (PDF/TXT/MD)
2. **Cloud Run Job (Ingestion)** - Reads GCS, chunks, embeds via Vertex AI, stores in BigQuery
3. **BigQuery** - Single `chunks` table with embeddings as `ARRAY<FLOAT64>`
4. **Cloud Run Service (API)** - FastAPI with `/ask`, `/sources`, `/health`
5. **Vertex AI** - Embeddings (`text-embedding-004`) + Gemini (`gemini-1.5-flash`)

### Data Flow

- **Ingestion:** GCS → extract text → chunk (800-1200 chars, 200 overlap) → embed → BigQuery
- **Query:** question → embed → fetch all chunks → Python cosine similarity → top-k → Gemini prompt → answer + citations

## Data Model

### BigQuery Table: `{dataset}.chunks`

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | STRING | Hash of source_uri (deterministic, for idempotency) |
| `chunk_id` | STRING | `{doc_id}_{chunk_index}` |
| `source_uri` | STRING | `gs://bucket/path/file.pdf` |
| `page` | INT64 | Page number (1-indexed for PDFs, NULL for txt/md) |
| `chunk_index` | INT64 | Position within document (0-indexed) |
| `chunk_text` | STRING | The actual text content |
| `embedding` | ARRAY<FLOAT64> | 768-dimensional vector from text-embedding-004 |
| `created_at` | TIMESTAMP | Ingestion timestamp |

### Idempotency Strategy

- Composite key: `doc_id` + `chunk_id`
- On re-ingestion: DELETE all rows WHERE `doc_id = X`, then INSERT new chunks

### Chunking Parameters

```python
CHUNK_SIZE = 1000      # target chars
CHUNK_OVERLAP = 200    # overlap between chunks
MIN_CHUNK_SIZE = 100   # discard tiny trailing chunks
```

## API Design

### POST /ask

```json
// Request
{
  "question": "What is the refund policy?",
  "top_k": 5
}

// Response
{
  "answer": "According to the documentation, refunds are available within 30 days...",
  "citations": [
    {"source_uri": "gs://bucket/policies.pdf", "page": 3, "chunk_id": "abc123_2"},
    {"source_uri": "gs://bucket/faq.md", "page": null, "chunk_id": "def456_0"}
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "abc123_2",
      "source_uri": "gs://bucket/policies.pdf",
      "page": 3,
      "chunk_text": "Refunds are available within 30 days of purchase...",
      "score": 0.89
    }
  ],
  "request_id": "req_abc123",
  "latency_ms": 1250
}
```

### GET /sources

```json
{
  "sources": [
    {"doc_id": "abc123", "source_uri": "gs://bucket/policies.pdf", "chunk_count": 12},
    {"doc_id": "def456", "source_uri": "gs://bucket/faq.md", "chunk_count": 5}
  ],
  "total_chunks": 17
}
```

### GET /health

```json
{
  "status": "ok",
  "version": "1.0.0",
  "bq_connected": true,
  "vertex_available": true
}
```

### Error Handling

- 400: Invalid request (missing question, top_k out of range)
- 500: Internal error (BQ/Vertex failures)
- All errors return `{"error": "message", "request_id": "..."}`

### RAG Prompt Template

```
You are a helpful assistant answering questions based on the provided documents.
Use ONLY the information from the chunks below. If the answer is not in the chunks, say so.
Cite sources using [source_uri, page X] format.

CHUNKS:
{chunks_formatted}

QUESTION: {question}

ANSWER:
```

## Security & IAM

### Service Accounts

| Service Account | Purpose | Roles |
|-----------------|---------|-------|
| `rag-api@{project}.iam` | Cloud Run API service | `roles/bigquery.dataViewer`, `roles/bigquery.jobUser`, `roles/aiplatform.user` |
| `rag-ingest@{project}.iam` | Ingestion job | `roles/bigquery.dataEditor`, `roles/bigquery.jobUser`, `roles/storage.objectViewer`, `roles/aiplatform.user` |

### Principle of Least Privilege

- API SA cannot write to BigQuery (only read)
- API SA cannot access GCS (no need)
- Ingestion SA cannot delete buckets, only read objects
- Both use Vertex AI via ADC - no API keys needed

### Secret Management

- No API keys required - all GCP services use ADC with service account identity
- If future integrations need secrets: use Secret Manager with `roles/secretmanager.secretAccessor`
- Environment variables for non-sensitive config (project ID, dataset name, bucket)

### Data Handling

- Logs never contain full chunk text (truncate to 100 chars max)
- Request IDs for tracing without exposing PII
- No document content in error messages

## Observability

### Structured Logging Format

```json
{
  "severity": "INFO",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123",
  "message": "RAG query completed",
  "component": "rag",
  "metrics": {
    "embedding_latency_ms": 120,
    "retrieval_latency_ms": 85,
    "llm_latency_ms": 950,
    "total_latency_ms": 1180,
    "chunks_retrieved": 5,
    "chunks_total": 150
  }
}
```

### Recommended Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| High error rate | >5% errors over 5min | Critical |
| Slow responses | P95 latency >5s for 10min | Warning |
| Ingestion failure | Job exit code != 0 | Critical |
| No chunks found | Log pattern "0 chunks retrieved" >10/hour | Warning |

## CI/CD

### GitHub Actions Workflow

1. Checkout code
2. Authenticate via Workload Identity Federation
3. Configure Docker for Artifact Registry
4. Build and push image
5. Deploy to Cloud Run with env vars
6. Run smoke test (curl /health)

### Required GitHub Secrets/Variables

| Name | Type | Example |
|------|------|---------|
| `GCP_PROJECT_ID` | Variable | `my-project-123` |
| `GCP_REGION` | Variable | `us-central1` |
| `GCP_WIF_PROVIDER` | Secret | `projects/123/locations/global/...` |
| `GCP_SERVICE_ACCOUNT` | Secret | `github-deploy@project.iam.gserviceaccount.com` |

## Repository Structure

```
gcp-rag-copilot/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker/
│   └── Dockerfile
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── rag.py
│   ├── bq.py
│   ├── vertex.py
│   ├── config.py
│   └── logging.py
├── ingestion/
│   ├── __init__.py
│   ├── ingest.py
│   └── chunking.py
├── infra/
│   ├── create_resources.sh
│   ├── deploy_cloudrun.sh
│   └── run_ingest.sh
├── .github/
│   └── workflows/
│       └── deploy.yml
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   └── test_rag_smoke.py
├── sample_docs/
│   ├── sample1.txt
│   └── sample2.md
├── runbook.md
└── SECURITY.md
```

## Key Design Decisions

1. **Python cosine similarity** for retrieval (simple, no special BQ features required)
2. **Cloud Run Job** for ingestion (same container tech, easy local testing)
3. **ARRAY<FLOAT64>** for embeddings in BigQuery
4. **Delete-then-insert** for idempotent re-ingestion
5. **ADC** for all GCP auth (no API keys)
6. **Workload Identity Federation** for CI/CD (keyless)
7. **Structured JSON logging** with request_id tracing

## Cost Estimates (Small Scale, ~100 docs)

- Cloud Run: ~$0-5/month (free tier covers light usage)
- BigQuery: ~$0-1/month (10GB free, queries cheap)
- Vertex AI Embeddings: ~$0.0001 per 1K chars
- Vertex AI Gemini: ~$0.00025 per 1K input tokens
- GCS: ~$0.02/GB/month
