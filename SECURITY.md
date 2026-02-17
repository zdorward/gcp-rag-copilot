# Security Guide

## Service Accounts

### API Service Account (`rag-api@PROJECT.iam`)

Used by the Cloud Run API service.

**Roles (least privilege):**
- `roles/bigquery.dataViewer` - Read chunks table
- `roles/bigquery.jobUser` - Execute queries
- `roles/aiplatform.user` - Call Vertex AI

**Cannot:**
- Write to BigQuery
- Access GCS
- Create/delete resources

### Ingestion Service Account (`rag-ingest@PROJECT.iam`)

Used by the ingestion Cloud Run Job.

**Roles (least privilege):**
- `roles/bigquery.dataEditor` - Read/write chunks table
- `roles/bigquery.jobUser` - Execute queries
- `roles/storage.objectViewer` - Read documents from GCS
- `roles/aiplatform.user` - Call Vertex AI for embeddings

**Cannot:**
- Delete buckets
- Access other projects
- Modify IAM

## Authentication

### Application Default Credentials (ADC)

All GCP services use ADC via service account identity. No API keys are stored or required.

```python
# This works automatically with the service account
from google.cloud import bigquery
client = bigquery.Client()  # Uses ADC
```

### Secret Manager (Optional)

If you need to store secrets (e.g., third-party API keys):

1. Store secret in Secret Manager
2. Grant `roles/secretmanager.secretAccessor` to the service account
3. Access in code:

```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/{project}/secrets/{secret_id}/versions/latest"
response = client.access_secret_version(request={"name": name})
secret_value = response.payload.data.decode("UTF-8")
```

## Data Handling

### Logging

- Full document text is **never** logged
- Chunk text is truncated to 100 chars in logs
- Request IDs enable tracing without exposing content
- Error messages don't include document content

```python
# Good
logger.info("Processed chunk", extra={"chunk_id": chunk.chunk_id})

# Bad - DON'T DO THIS
logger.info(f"Chunk content: {chunk.chunk_text}")
```

### BigQuery

- Data at rest is encrypted by default
- Column-level security can be added if needed
- Audit logs track all access

### Network Security

- Cloud Run: Configure `--ingress=internal` for private APIs
- VPC Service Controls can restrict data exfiltration
- BigQuery accessed via Google's private network

## API Security

### Current: Unauthenticated (Demo Mode)

Default deployment allows unauthenticated access. Suitable for:
- Internal testing
- Development
- Demo environments

### Production: Add Authentication

**Option 1: Cloud Run IAM**

```bash
# Deploy without --allow-unauthenticated
gcloud run deploy rag-copilot-api \
  --no-allow-unauthenticated \
  ...

# Grant access to specific users/service accounts
gcloud run services add-iam-policy-binding rag-copilot-api \
  --member="user:alice@example.com" \
  --role="roles/run.invoker"
```

**Option 2: API Gateway with API Keys**

1. Create API Gateway
2. Configure API key requirement
3. Route to Cloud Run backend

**Option 3: Identity-Aware Proxy (IAP)**

1. Enable IAP on Cloud Run
2. Configure OAuth consent screen
3. Add authorized users

## Compliance Notes

- **Data Residency**: Configure BigQuery and GCS location appropriately
- **Data Retention**: Implement deletion policies if required
- **Audit Logging**: Enable Cloud Audit Logs for compliance
- **Encryption**: All data encrypted at rest and in transit by default

## Vulnerability Reporting

Report security vulnerabilities to: security@example.com

Do not create public GitHub issues for security vulnerabilities.
