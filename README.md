# GCP RAG Copilot

A production-ready document Q&A copilot deployed on Google Cloud Platform. Ingests PDFs/TXT/MD from Cloud Storage, creates embeddings via Vertex AI, stores chunks in BigQuery, and answers questions using Gemini with citations.

## Architecture

```
+--------------+     +--------------+     +--------------+
|    GCS       |---->|  Cloud Run   |---->|  Cloud Run   |
|   Bucket     |     |    Job       |     |    API       |
|  (docs)      |     |  (ingest)    |     |  (FastAPI)   |
+--------------+     +------+-------+     +------+-------+
                            |                    |
                     embed  |                    | embed + generate
                            v                    v
                     +--------------+     +--------------+
                     |  Vertex AI   |     |  Vertex AI   |
                     |  Embeddings  |     |   Gemini     |
                     +------+-------+     +--------------+
                            |                    ^
                     store  |                    | retrieve
                            v                    |
                     +---------------------------+
                     |        BigQuery           |
                     |    (chunks table)         |
                     +---------------------------+
```

## Quick Start

### Prerequisites

- Google Cloud SDK (`gcloud`)
- Docker
- Python 3.11+
- A GCP project with billing enabled

### Setup (< 30 minutes)

1. **Clone and configure**

```bash
git clone <repo-url>
cd gcp-rag-copilot

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your GCP_PROJECT_ID and GCS_BUCKET
```

2. **Create GCP resources**

```bash
chmod +x infra/*.sh
./infra/create_resources.sh
```

3. **Upload sample documents**

```bash
gsutil cp sample_docs/* gs://YOUR_BUCKET/documents/
```

4. **Run ingestion**

```bash
./infra/run_ingest.sh local
```

5. **Deploy API**

```bash
./infra/deploy_cloudrun.sh
```

### API Usage

```bash
# Health check
curl https://YOUR_SERVICE_URL/health

# Ask a question
curl -X POST https://YOUR_SERVICE_URL/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?", "top_k": 5}'

# List indexed sources
curl https://YOUR_SERVICE_URL/sources
```

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Set up authentication
gcloud auth application-default login

# Run locally
uvicorn app.main:app --reload --port 8080

# Run tests
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/ask` | POST | RAG query with question and top_k |
| `/sources` | GET | List indexed document sources |

### POST /ask

Request:
```json
{
  "question": "What is the return policy?",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "The company offers a 30-day return policy...",
  "citations": [
    {"source_uri": "gs://bucket/policy.pdf", "page": 1, "chunk_id": "abc_0"}
  ],
  "retrieved_chunks": [...],
  "request_id": "req_abc123",
  "latency_ms": 1250
}
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_PROJECT_ID` | Yes | - | GCP project ID |
| `GCS_BUCKET` | Yes | - | Source documents bucket |
| `GCP_REGION` | No | us-central1 | GCP region |
| `BQ_DATASET` | No | rag_copilot | BigQuery dataset |
| `BQ_TABLE` | No | chunks | BigQuery table |
| `GCS_PREFIX` | No | documents/ | GCS path prefix |
| `EMBEDDING_MODEL` | No | text-embedding-004 | Vertex AI embedding model |
| `LLM_MODEL` | No | gemini-2.0-flash | Vertex AI LLM |
| `LOG_LEVEL` | No | INFO | Logging level |

## Cost Estimates

For small-scale usage (~100 documents, ~1000 queries/month):

| Service | Estimated Cost |
|---------|---------------|
| Cloud Run | $0-5/month (free tier) |
| BigQuery | $0-1/month (10GB free) |
| Vertex AI Embeddings | ~$0.10/month |
| Vertex AI Gemini | ~$1-5/month |
| GCS | ~$0.02/GB/month |

**Total: ~$5-15/month** for light usage.

## CI/CD

GitHub Actions workflow deploys on push to `main`:

1. Runs tests
2. Builds Docker image
3. Pushes to Artifact Registry
4. Deploys to Cloud Run
5. Runs smoke test

### Setup Workload Identity Federation

See [GitHub Actions docs](https://github.com/google-github-actions/auth#setup) for WIF setup.

Required GitHub secrets:
- `GCP_WIF_PROVIDER`: Workload Identity Provider
- `GCP_SERVICE_ACCOUNT`: Deployment service account

Required GitHub variables:
- `GCP_PROJECT_ID`: GCP project ID
- `BQ_DATASET`: BigQuery dataset name
- `GCS_BUCKET`: GCS bucket name

## Security

See [SECURITY.md](SECURITY.md) for:
- IAM roles and service accounts
- Secret management
- Data handling practices

## Operations

See [runbook.md](runbook.md) for:
- Monitoring and alerting
- Common issues and solutions
- Debugging procedures

## License

MIT License - see [LICENSE](LICENSE)
