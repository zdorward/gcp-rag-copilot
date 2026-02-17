"""Document ingestion script for GCP RAG Copilot."""

import sys
from datetime import datetime, timezone

from google.cloud import storage

from app.config import get_settings
from app.logging import setup_logging, get_logger
from app.bq import BigQueryClient, Chunk, generate_doc_id
from app.vertex import VertexClient
from ingestion.chunking import extract_text, chunk_document_with_pages

logger = get_logger(__name__)


def list_documents(bucket_name: str, prefix: str) -> list[storage.Blob]:
    """List all documents in GCS bucket with given prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))

    # Filter to supported file types
    supported_extensions = (".pdf", ".txt", ".md", ".markdown")
    documents = [
        b for b in blobs
        if b.name.lower().endswith(supported_extensions) and b.size > 0
    ]

    logger.info(f"Found {len(documents)} documents in gs://{bucket_name}/{prefix}")
    return documents


def process_document(
    blob: storage.Blob,
    vertex_client: VertexClient,
    bq_client: BigQueryClient,
) -> int:
    """
    Process a single document: extract, chunk, embed, store.

    Returns number of chunks created.
    """
    source_uri = f"gs://{blob.bucket.name}/{blob.name}"
    doc_id = generate_doc_id(source_uri)

    logger.info(f"Processing document: {source_uri}")

    # Download document content
    content = blob.download_as_bytes()

    # Extract text
    full_text, pages = extract_text(content, blob.name)

    if not full_text.strip():
        logger.warning(f"No text extracted from {source_uri}")
        return 0

    # Chunk the document
    chunks_data = chunk_document_with_pages(
        pages,
        chunk_size=1000,
        overlap=200,
        min_chunk_size=100,
    )

    if not chunks_data:
        logger.warning(f"No chunks created from {source_uri}")
        return 0

    # Generate embeddings
    chunk_texts = [c["text"] for c in chunks_data]
    embeddings = vertex_client.embed_texts(chunk_texts)

    # Create Chunk objects
    now = datetime.now(timezone.utc)
    chunks = []

    for chunk_data, embedding in zip(chunks_data, embeddings):
        chunk = Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}_{chunk_data['chunk_index']}",
            source_uri=source_uri,
            page=chunk_data["page"],
            chunk_index=chunk_data["chunk_index"],
            chunk_text=chunk_data["text"],
            embedding=embedding,
            created_at=now,
        )
        chunks.append(chunk)

    # Delete existing chunks for this document (idempotency)
    bq_client.delete_document_chunks(doc_id)

    # Insert new chunks
    bq_client.insert_chunks(chunks)

    logger.info(f"Ingested {len(chunks)} chunks from {source_uri}")
    return len(chunks)


def run_ingestion() -> dict:
    """
    Run the full ingestion process.

    Returns summary statistics.
    """
    settings = get_settings()
    setup_logging(level=settings.log_level)

    logger.info("Starting document ingestion", extra={
        "context": {
            "bucket": settings.gcs_bucket,
            "prefix": settings.gcs_prefix,
            "project": settings.gcp_project_id,
        }
    })

    # Initialize clients
    bq_client = BigQueryClient(
        project_id=settings.gcp_project_id,
        dataset=settings.bq_dataset,
        table=settings.bq_table,
    )

    vertex_client = VertexClient(
        project_id=settings.gcp_project_id,
        region=settings.gcp_region,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )

    # Ensure BQ resources exist
    bq_client.ensure_dataset_exists()
    bq_client.ensure_table_exists()

    # List documents
    documents = list_documents(settings.gcs_bucket, settings.gcs_prefix)

    if not documents:
        logger.warning("No documents found to ingest")
        return {"documents_processed": 0, "total_chunks": 0, "errors": 0}

    # Process each document
    total_chunks = 0
    errors = 0

    for blob in documents:
        try:
            chunks_created = process_document(blob, vertex_client, bq_client)
            total_chunks += chunks_created
        except Exception as e:
            logger.error(f"Error processing {blob.name}: {e}", exc_info=True)
            errors += 1

    summary = {
        "documents_processed": len(documents) - errors,
        "total_chunks": total_chunks,
        "errors": errors,
    }

    logger.info("Ingestion complete", extra={"metrics": summary})
    return summary


if __name__ == "__main__":
    try:
        summary = run_ingestion()
        print(f"Ingestion complete: {summary}")
        sys.exit(0 if summary["errors"] == 0 else 1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)
