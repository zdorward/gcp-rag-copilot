# GCP RAG Copilot - Project Summary

## What It Does

This is a **document Q&A system**. You upload documents (PDFs, text files, markdown), and then you can ask questions about them in plain English. The system finds the relevant information and answers your question with citations.

## How It Works (Simple Version)

```
1. Upload documents to cloud storage
2. System reads and "understands" the documents
3. You ask a question
4. System finds relevant parts and generates an answer
```

## How It Works (Technical Version)

1. **Ingestion**: Documents in Google Cloud Storage are chunked into smaller pieces
2. **Embedding**: Each chunk is converted to a numerical vector using Vertex AI (this captures the "meaning")
3. **Storage**: Chunks and their embeddings are stored in BigQuery
4. **Query**: When you ask a question, it's also converted to a vector
5. **Retrieval**: The system finds chunks with similar vectors (similar meaning)
6. **Generation**: Gemini AI reads the relevant chunks and writes an answer with citations

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Storage | Google Cloud Storage | Store source documents |
| Database | BigQuery | Store document chunks and embeddings |
| Embeddings | Vertex AI (text-embedding-004) | Convert text to vectors |
| LLM | Gemini 2.0 Flash | Generate answers |
| API | FastAPI on Cloud Run | Serve the application |
| CI/CD | GitHub Actions | Automatic testing and deployment |

## API Endpoints

| Endpoint | What It Does |
|----------|--------------|
| `GET /health` | Check if the system is running |
| `GET /sources` | List all indexed documents |
| `POST /ask` | Ask a question, get an answer |

## Key Features

- **Citations**: Every answer includes references to source documents
- **Scalable**: Runs on Google Cloud, scales automatically
- **Secure**: Uses service accounts with minimal permissions
- **Automated**: Push code to GitHub → automatically deployed

## Cost

For light usage (~100 documents, ~1000 queries/month): **~$5-15/month**

## Architecture Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Google    │────▶│  Ingestion  │────▶│  BigQuery   │
│   Cloud     │     │    Job      │     │  (chunks)   │
│   Storage   │     └─────────────┘     └──────┬──────┘
│  (documents)│                                │
└─────────────┘                                │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │────▶│  Cloud Run  │────▶│  Vertex AI  │
│  (question) │◀────│    API      │◀────│   Gemini    │
└─────────────┘     └─────────────┘     └─────────────┘
     answer              │
                         ▼
                  retrieves relevant
                  chunks from BigQuery
```
