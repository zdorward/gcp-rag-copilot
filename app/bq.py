"""BigQuery helper functions for chunk storage and retrieval."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

from app.logging import get_logger

logger = get_logger(__name__)


# BigQuery schema for chunks table
CHUNKS_SCHEMA = [
    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("source_uri", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("page", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("chunk_index", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("chunk_text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]


@dataclass
class Chunk:
    """Represents a document chunk with embedding."""

    doc_id: str
    chunk_id: str
    source_uri: str
    page: int | None
    chunk_index: int
    chunk_text: str
    embedding: list[float]
    created_at: datetime

    def to_bq_row(self) -> dict[str, Any]:
        """Convert to BigQuery row format."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_uri": self.source_uri,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text,
            "embedding": self.embedding,
            "created_at": self.created_at,
        }


def generate_doc_id(source_uri: str) -> str:
    """Generate deterministic document ID from source URI."""
    return hashlib.sha256(source_uri.encode()).hexdigest()[:16]


class BigQueryClient:
    """Client for BigQuery operations on chunks table."""

    def __init__(self, project_id: str, dataset: str, table: str):
        self.project_id = project_id
        self.dataset = dataset
        self.table = table
        self.client = bigquery.Client(project=project_id)
        self.full_table_id = f"{project_id}.{dataset}.{table}"

    def ensure_dataset_exists(self) -> None:
        """Create dataset if it doesn't exist."""
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{self.dataset}")
        dataset_ref.location = "US"

        try:
            self.client.create_dataset(dataset_ref, exists_ok=True)
            logger.info(f"Dataset {self.dataset} ready")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    def ensure_table_exists(self) -> None:
        """Create chunks table if it doesn't exist."""
        table_ref = bigquery.Table(self.full_table_id, schema=CHUNKS_SCHEMA)

        try:
            self.client.create_table(table_ref, exists_ok=True)
            logger.info(f"Table {self.table} ready")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise

    def delete_document_chunks(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns rows deleted."""
        query = f"""
            DELETE FROM `{self.full_table_id}`
            WHERE doc_id = @doc_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id)
            ]
        )

        result = self.client.query(query, job_config=job_config).result()
        deleted = result.num_dml_affected_rows or 0
        logger.info(f"Deleted {deleted} chunks for doc_id={doc_id}")
        return deleted

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert chunks into BigQuery."""
        if not chunks:
            return

        rows = [chunk.to_bq_row() for chunk in chunks]
        errors = self.client.insert_rows_json(self.full_table_id, rows)

        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise RuntimeError(f"Failed to insert chunks: {errors}")

        logger.info(f"Inserted {len(chunks)} chunks")

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Fetch all chunks from BigQuery."""
        query = f"""
            SELECT doc_id, chunk_id, source_uri, page, chunk_index,
                   chunk_text, embedding, created_at
            FROM `{self.full_table_id}`
        """

        results = []
        for row in self.client.query(query).result():
            results.append({
                "doc_id": row.doc_id,
                "chunk_id": row.chunk_id,
                "source_uri": row.source_uri,
                "page": row.page,
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "embedding": list(row.embedding),
                "created_at": row.created_at,
            })

        logger.info(f"Fetched {len(results)} chunks from BigQuery")
        return results

    def get_sources(self) -> list[dict[str, Any]]:
        """Get distinct sources with chunk counts."""
        query = f"""
            SELECT doc_id, source_uri, COUNT(*) as chunk_count
            FROM `{self.full_table_id}`
            GROUP BY doc_id, source_uri
            ORDER BY source_uri
        """

        results = []
        for row in self.client.query(query).result():
            results.append({
                "doc_id": row.doc_id,
                "source_uri": row.source_uri,
                "chunk_count": row.chunk_count,
            })

        return results

    def get_total_chunk_count(self) -> int:
        """Get total number of chunks."""
        query = f"SELECT COUNT(*) as count FROM `{self.full_table_id}`"
        result = list(self.client.query(query).result())
        return result[0].count if result else 0

    def health_check(self) -> bool:
        """Check if BigQuery is accessible."""
        try:
            query = "SELECT 1"
            list(self.client.query(query).result())
            return True
        except Exception as e:
            logger.error(f"BigQuery health check failed: {e}")
            return False
