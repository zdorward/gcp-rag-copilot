"""Tests for BigQuery helper module."""

import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest


def test_chunk_model_to_bq_row():
    """Chunk model should convert to BigQuery row format."""
    from app.bq import Chunk

    chunk = Chunk(
        doc_id="abc123",
        chunk_id="abc123_0",
        source_uri="gs://bucket/file.pdf",
        page=1,
        chunk_index=0,
        chunk_text="Sample text content",
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )

    row = chunk.to_bq_row()

    assert row["doc_id"] == "abc123"
    assert row["chunk_id"] == "abc123_0"
    assert row["source_uri"] == "gs://bucket/file.pdf"
    assert row["page"] == 1
    assert row["chunk_index"] == 0
    assert row["chunk_text"] == "Sample text content"
    assert row["embedding"] == [0.1, 0.2, 0.3]
    assert row["created_at"] == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


def test_generate_doc_id():
    """doc_id should be deterministic hash of source_uri."""
    from app.bq import generate_doc_id

    uri = "gs://bucket/path/file.pdf"
    doc_id1 = generate_doc_id(uri)
    doc_id2 = generate_doc_id(uri)

    assert doc_id1 == doc_id2
    assert len(doc_id1) == 16  # First 16 chars of sha256


def test_bq_schema_definition():
    """BQ schema should match expected structure."""
    from app.bq import CHUNKS_SCHEMA

    field_names = [f.name for f in CHUNKS_SCHEMA]

    assert "doc_id" in field_names
    assert "chunk_id" in field_names
    assert "source_uri" in field_names
    assert "page" in field_names
    assert "chunk_index" in field_names
    assert "chunk_text" in field_names
    assert "embedding" in field_names
    assert "created_at" in field_names
