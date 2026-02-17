# GCP RAG Copilot - Demo Guide

## Quick Setup

The API is already deployed at:
```
https://rag-api-708256638053.us-central1.run.app
```

No setup needed to run the demo - just use the curl commands below.

---

## Demo Script

### 1. Health Check (30 seconds)

**What to say:** "First, let's verify the system is running and connected to all services."

```bash
curl -s https://rag-api-708256638053.us-central1.run.app/health | python3 -m json.tool
```

**Expected output:**
```json
{
    "status": "ok",
    "version": "1.0.0",
    "bq_connected": true,
    "vertex_available": true
}
```

**Talking point:** "The system is connected to BigQuery for storage and Vertex AI for embeddings and generation."

---

### 2. Show Indexed Documents (30 seconds)

**What to say:** "Let's see what documents are in our knowledge base."

```bash
curl -s https://rag-api-708256638053.us-central1.run.app/sources | python3 -m json.tool
```

**Expected output:**
```json
{
    "sources": [
        {
            "doc_id": "cf400e31c4f6b898",
            "source_uri": "gs://gcp-rag-copilot-docs-708256/documents/sample1.txt",
            "chunk_count": 1
        },
        {
            "doc_id": "2aeb40d3886bea83",
            "source_uri": "gs://gcp-rag-copilot-docs-708256/documents/sample2.md",
            "chunk_count": 1
        }
    ],
    "total_chunks": 2
}
```

**Talking point:** "We have two documents indexed - a return policy and product documentation."

---

### 3. Ask Questions (2 minutes)

**What to say:** "Now let's ask questions in natural language."

#### Question 1: Return Policy

```bash
curl -s -X POST 'https://rag-api-708256638053.us-central1.run.app/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the return policy?"}' | python3 -c "
import sys,json
r = json.load(sys.stdin)
print('Answer:', r['answer'])
print(f\"Latency: {r['latency_ms']}ms\")
"
```

**Talking point:** "Notice the answer includes citations - we can trace every fact back to its source document."

#### Question 2: Product Features

```bash
curl -s -X POST 'https://rag-api-708256638053.us-central1.run.app/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What features does the product have?"}' | python3 -c "
import sys,json
r = json.load(sys.stdin)
print('Answer:', r['answer'])
print(f\"Latency: {r['latency_ms']}ms\")
"
```

#### Question 3: Contact Support

```bash
curl -s -X POST 'https://rag-api-708256638053.us-central1.run.app/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I contact support?"}' | python3 -c "
import sys,json
r = json.load(sys.stdin)
print('Answer:', r['answer'])
print(f\"Latency: {r['latency_ms']}ms\")
"
```

**Talking point:** "The system found the support email and phone number from the product documentation."

#### (Optional) Show Full JSON Response

To show the complete RAG pipeline with retrieved chunks and similarity scores:

```bash
curl -s -X POST 'https://rag-api-708256638053.us-central1.run.app/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the return policy?"}' | python3 -m json.tool
```

**Talking point:** "Under the hood, you can see the retrieved chunks with similarity scores - this proves the answer is grounded in your actual documents, not hallucinated."

---

### 4. Live Document Ingestion (Optional, 3 minutes)

**What to say:** "Let me show you how easy it is to add new documents to the knowledge base."

#### Step 1: Create a new document

```bash
echo "ACME Corp was founded in 2020 in San Francisco. We have 500 employees and are valued at 2 billion dollars. Our CEO is Jane Smith." > /tmp/company-info.txt
```

#### Step 2: Upload to Cloud Storage

```bash
gsutil cp /tmp/company-info.txt gs://gcp-rag-copilot-docs-708256/documents/
```

#### Step 3: Run ingestion

```bash
cd /Users/zackdorward/dev/atco && ./infra/run_ingest.sh local
```

#### Step 4: Query the new document

```bash
curl -s -X POST 'https://rag-api-708256638053.us-central1.run.app/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question": "Who is the CEO and when was the company founded?"}' | python3 -c "
import sys,json
r = json.load(sys.stdin)
print('Answer:', r['answer'])
print(f\"Latency: {r['latency_ms']}ms\")
"
```

**Talking point:** "Within seconds, the new document is searchable and the AI can answer questions about it."

---

### 5. Show CI/CD (Optional, 1 minute)

**What to say:** "The project also has automated deployment."

1. Open https://github.com/zdorward/gcp-rag-copilot/actions
2. Show the green checkmarks on recent commits
3. Explain: "Every push to main automatically runs tests and deploys to production."

---

## Key Talking Points

1. **RAG Architecture**: "RAG stands for Retrieval-Augmented Generation. Instead of relying on the AI's training data, we retrieve relevant documents first, then generate an answer based on those specific sources."

2. **Why This Matters**: "This prevents hallucinations because the AI only answers based on your actual documents, not made-up information."

3. **Citations**: "Every answer includes citations so you can verify the source and build trust."

4. **Scalability**: "This runs on Google Cloud and can scale from 0 to thousands of requests automatically."

5. **Cost Effective**: "For light usage, this costs about $5-15/month."

---

## Troubleshooting

**If the API is slow:** First request after idle may take 5-10 seconds (cold start). Subsequent requests are fast.

**If you get errors:** Check that you're connected to the internet and the service is running with the health check.
