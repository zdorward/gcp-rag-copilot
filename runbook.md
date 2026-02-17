# Operations Runbook

## Monitoring

### Cloud Monitoring Dashboard

Create a dashboard with these widgets:

1. **Request Count** (Cloud Run metric: `request_count`)
2. **Latency P50/P95** (Cloud Run metric: `request_latencies`)
3. **Error Rate** (Log-based: `severity=ERROR`)
4. **Memory Usage** (Cloud Run metric: `container/memory/utilization`)
5. **Ingestion Job Status** (Cloud Run Jobs metric: `job/completed_task_attempt_count`)

### Recommended Alerts

| Alert | Condition | Channel |
|-------|-----------|---------|
| High Error Rate | Error rate > 5% for 5 min | PagerDuty |
| High Latency | P95 > 5s for 10 min | Slack |
| Ingestion Failed | Job exit code != 0 | Email |
| Low Similarity Scores | "0 chunks retrieved" > 10/hour | Slack |

### Log Queries

**View recent errors:**
```
resource.type="cloud_run_revision"
severity=ERROR
```

**Trace a specific request:**
```
resource.type="cloud_run_revision"
jsonPayload.request_id="req_abc123"
```

**View slow queries:**
```
resource.type="cloud_run_revision"
jsonPayload.metrics.total_latency_ms > 3000
```

## Common Issues

### 1. "No documents have been indexed"

**Symptom:** API returns "No documents have been indexed yet"

**Diagnosis:**
```bash
# Check if chunks exist
bq query "SELECT COUNT(*) FROM \`PROJECT.rag_copilot.chunks\`"

# Check ingestion logs
gcloud logging read "resource.type=cloud_run_job" --limit=50
```

**Resolution:**
1. Verify documents exist in GCS: `gsutil ls gs://BUCKET/documents/`
2. Run ingestion: `./infra/run_ingest.sh local`
3. Check for ingestion errors in logs

### 2. Vertex AI Rate Limits

**Symptom:** 429 errors in logs, slow responses

**Diagnosis:**
```bash
gcloud logging read "jsonPayload.message:\"rate limit\"" --limit=20
```

**Resolution:**
1. Reduce batch size in `vertex.py` (default 250)
2. Add exponential backoff
3. Request quota increase from GCP

### 3. BigQuery Quota Exceeded

**Symptom:** 403 errors mentioning quota

**Diagnosis:**
```bash
gcloud logging read "jsonPayload.message:\"quota\"" --limit=20
```

**Resolution:**
1. Check quota usage in Cloud Console
2. For large datasets, implement pagination
3. Request quota increase

### 4. PDF Extraction Failures

**Symptom:** Some PDFs return no text

**Diagnosis:**
Check ingestion logs for specific file errors.

**Resolution:**
1. PDFs may be scanned images (need OCR)
2. Try alternative library (pdfplumber)
3. Skip problematic files

### 5. Low Similarity Scores

**Symptom:** Retrieved chunks have low relevance

**Diagnosis:**
```bash
# Check typical scores in responses
gcloud logging read "jsonPayload.retrieved_chunks" --limit=10
```

**Resolution:**
1. Verify embedding model matches between ingest and query
2. Check chunk quality (too long/short?)
3. Consider query reformulation

## Procedures

### Re-running Ingestion

Ingestion is idempotent. Safe to re-run:

```bash
# Delete and re-ingest all documents
./infra/run_ingest.sh local
```

### Scaling Up

For higher load:

```bash
# Increase Cloud Run instances
gcloud run services update rag-copilot-api \
  --max-instances=50 \
  --memory=2Gi \
  --cpu=2
```

### Viewing Metrics

```bash
# Get recent latency stats
gcloud logging read "resource.type=cloud_run_revision" \
  --format="table(jsonPayload.metrics.total_latency_ms)" \
  --limit=100
```

### Emergency: Disable API

```bash
# Set instances to 0
gcloud run services update rag-copilot-api --max-instances=0
```

### Rollback Deployment

```bash
# List revisions
gcloud run revisions list --service=rag-copilot-api

# Route traffic to previous revision
gcloud run services update-traffic rag-copilot-api \
  --to-revisions=rag-copilot-api-XXXX=100
```

## Contacts

- **On-call**: oncall@example.com
- **GCP Support**: https://console.cloud.google.com/support
- **Repository**: https://github.com/example/gcp-rag-copilot
