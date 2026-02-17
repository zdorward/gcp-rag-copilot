# GCP RAG Copilot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready document Q&A copilot on GCP that ingests documents from GCS, creates embeddings, stores in BigQuery, and answers questions via Gemini with citations.

**Architecture:** FastAPI service on Cloud Run with a separate Cloud Run Job for ingestion. Documents are chunked and embedded via Vertex AI, stored in BigQuery. Retrieval uses Python cosine similarity over fetched embeddings, then Gemini generates answers with citations.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, google-cloud-bigquery, google-cloud-storage, google-cloud-aiplatform, pydantic-settings, pypdf, pytest

---

## Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

**Step 1: Create .gitignore**

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
.venv/
env/

# Environment variables
.env
.env.local

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build
dist/
build/
*.egg-info/

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

**Step 2: Create LICENSE (MIT)**

```text
MIT License

Copyright (c) 2026 GCP RAG Copilot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 3: Create .env.example**

```bash
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# BigQuery Configuration
BQ_DATASET=rag_copilot
BQ_TABLE=chunks

# GCS Configuration
GCS_BUCKET=your-rag-copilot-docs
GCS_PREFIX=documents/

# Vertex AI Configuration
EMBEDDING_MODEL=text-embedding-004
LLM_MODEL=gemini-1.5-flash

# Application Configuration
LOG_LEVEL=INFO
APP_VERSION=1.0.0
```

**Step 4: Create requirements.txt**

```text
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
pydantic>=2.5.0,<3.0.0
pydantic-settings>=2.1.0,<3.0.0
google-cloud-bigquery>=3.14.0,<4.0.0
google-cloud-storage>=2.14.0,<3.0.0
google-cloud-aiplatform>=1.38.0,<2.0.0
pypdf>=3.17.0,<5.0.0
numpy>=1.26.0,<2.0.0
```

**Step 5: Create requirements-dev.txt**

```text
-r requirements.txt
pytest>=7.4.0,<9.0.0
pytest-asyncio>=0.23.0,<1.0.0
httpx>=0.26.0,<1.0.0
```

**Step 6: Commit**

```bash
git add .gitignore LICENSE .env.example requirements.txt requirements-dev.txt
git commit -m "chore: add project scaffolding files"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Step 1: Create app/__init__.py**

```python
"""GCP RAG Copilot application package."""
```

**Step 2: Create tests/__init__.py**

```python
"""Test package for GCP RAG Copilot."""
```

**Step 3: Write the failing test for config**

Create `tests/test_config.py`:

```python
"""Tests for configuration module."""

import os
from unittest.mock import patch


def test_settings_loads_from_env():
    """Settings should load values from environment variables."""
    env_vars = {
        "GCP_PROJECT_ID": "test-project",
        "GCP_REGION": "us-west1",
        "BQ_DATASET": "test_dataset",
        "BQ_TABLE": "test_table",
        "GCS_BUCKET": "test-bucket",
        "GCS_PREFIX": "docs/",
        "EMBEDDING_MODEL": "text-embedding-004",
        "LLM_MODEL": "gemini-1.5-flash",
        "LOG_LEVEL": "DEBUG",
        "APP_VERSION": "2.0.0",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        # Import inside patch to pick up env vars
        from app.config import Settings
        settings = Settings()

        assert settings.gcp_project_id == "test-project"
        assert settings.gcp_region == "us-west1"
        assert settings.bq_dataset == "test_dataset"
        assert settings.bq_table == "test_table"
        assert settings.gcs_bucket == "test-bucket"
        assert settings.gcs_prefix == "docs/"
        assert settings.embedding_model == "text-embedding-004"
        assert settings.llm_model == "gemini-1.5-flash"
        assert settings.log_level == "DEBUG"
        assert settings.app_version == "2.0.0"


def test_settings_has_defaults():
    """Settings should have sensible defaults."""
    env_vars = {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        from app.config import Settings
        settings = Settings()

        assert settings.gcp_region == "us-central1"
        assert settings.bq_dataset == "rag_copilot"
        assert settings.bq_table == "chunks"
        assert settings.gcs_prefix == "documents/"
        assert settings.embedding_model == "text-embedding-004"
        assert settings.llm_model == "gemini-1.5-flash"
        assert settings.log_level == "INFO"
        assert settings.app_version == "1.0.0"
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.config'"

**Step 5: Write the implementation**

Create `app/config.py`:

```python
"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GCP Configuration
    gcp_project_id: str
    gcp_region: str = "us-central1"

    # BigQuery Configuration
    bq_dataset: str = "rag_copilot"
    bq_table: str = "chunks"

    # GCS Configuration
    gcs_bucket: str
    gcs_prefix: str = "documents/"

    # Vertex AI Configuration
    embedding_model: str = "text-embedding-004"
    llm_model: str = "gemini-1.5-flash"

    # Application Configuration
    log_level: str = "INFO"
    app_version: str = "1.0.0"

    @property
    def bq_full_table_id(self) -> str:
        """Return fully qualified BigQuery table ID."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{self.bq_table}"


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add app/__init__.py app/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add configuration module with pydantic-settings"
```

---

## Task 3: Structured Logging Module

**Files:**
- Create: `app/logging.py`
- Create: `tests/test_logging.py`

**Step 1: Write the failing test**

Create `tests/test_logging.py`:

```python
"""Tests for structured logging module."""

import json
import logging
from io import StringIO

from app.logging import setup_logging, get_logger, RequestContext


def test_setup_logging_configures_json_format():
    """Logging should output JSON format."""
    # Capture log output
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    setup_logging(level="INFO")
    logger = get_logger("test")

    # Replace handler to capture output
    logger.handlers = [handler]
    logger.propagate = False

    # Get the formatter from setup_logging
    from app.logging import JSONFormatter
    handler.setFormatter(JSONFormatter())

    logger.info("test message")

    output = stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["message"] == "test message"
    assert log_entry["severity"] == "INFO"
    assert "timestamp" in log_entry
    assert log_entry["component"] == "test"


def test_request_context_adds_request_id():
    """RequestContext should add request_id to log entries."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    from app.logging import JSONFormatter
    handler.setFormatter(JSONFormatter())

    logger = get_logger("test")
    logger.handlers = [handler]
    logger.propagate = False

    with RequestContext(request_id="req_123"):
        logger.info("contextual message")

    output = stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["request_id"] == "req_123"


def test_logger_with_metrics():
    """Logger should support metrics in extra field."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    from app.logging import JSONFormatter
    handler.setFormatter(JSONFormatter())

    logger = get_logger("test")
    logger.handlers = [handler]
    logger.propagate = False

    logger.info("query completed", extra={"metrics": {"latency_ms": 150}})

    output = stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["metrics"]["latency_ms"] == 150
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.logging'"

**Step 3: Write the implementation**

Create `app/logging.py`:

```python
"""Structured JSON logging for Cloud Run compatibility."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable for request-scoped data
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestContext:
    """Context manager for request-scoped logging context."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.token = None

    def __enter__(self):
        self.token = _request_id.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _request_id.reset(self.token)
        return False


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for Cloud Logging."""

    LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self.LEVEL_MAP.get(record.levelno, "INFO"),
            "message": record.getMessage(),
            "component": record.name,
        }

        # Add request_id if available
        request_id = _request_id.get()
        if request_id:
            log_entry["request_id"] = request_id

        # Add extra fields (metrics, context, etc.)
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add JSON handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/logging.py tests/test_logging.py
git commit -m "feat: add structured JSON logging module"
```

---

## Task 4: BigQuery Helper Module

**Files:**
- Create: `app/bq.py`
- Create: `tests/test_bq.py`

**Step 1: Write the failing test**

Create `tests/test_bq.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bq.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.bq'"

**Step 3: Write the implementation**

Create `app/bq.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bq.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/bq.py tests/test_bq.py
git commit -m "feat: add BigQuery helper module for chunk storage"
```

---

## Task 5: Vertex AI Client Module

**Files:**
- Create: `app/vertex.py`
- Create: `tests/test_vertex.py`

**Step 1: Write the failing test**

Create `tests/test_vertex.py`:

```python
"""Tests for Vertex AI client module."""

from unittest.mock import MagicMock, patch


def test_vertex_client_initialization():
    """VertexClient should initialize with project and region."""
    with patch("app.vertex.aiplatform") as mock_aiplatform:
        from app.vertex import VertexClient

        client = VertexClient(
            project_id="test-project",
            region="us-central1",
            embedding_model="text-embedding-004",
            llm_model="gemini-1.5-flash",
        )

        mock_aiplatform.init.assert_called_once_with(
            project="test-project",
            location="us-central1",
        )


def test_cosine_similarity():
    """cosine_similarity should compute correct similarity."""
    from app.vertex import cosine_similarity

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    # Identical vectors should have similarity 1.0
    assert abs(cosine_similarity(vec1, vec2) - 1.0) < 0.001

    # Orthogonal vectors should have similarity 0.0
    vec3 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec1, vec3)) < 0.001

    # Opposite vectors should have similarity -1.0
    vec4 = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec1, vec4) - (-1.0)) < 0.001


def test_rank_chunks_by_similarity():
    """rank_chunks_by_similarity should return top_k most similar."""
    from app.vertex import rank_chunks_by_similarity

    query_embedding = [1.0, 0.0, 0.0]
    chunks = [
        {"chunk_id": "a", "embedding": [1.0, 0.0, 0.0], "chunk_text": "exact match"},
        {"chunk_id": "b", "embedding": [0.0, 1.0, 0.0], "chunk_text": "orthogonal"},
        {"chunk_id": "c", "embedding": [0.7, 0.7, 0.0], "chunk_text": "partial"},
    ]

    ranked = rank_chunks_by_similarity(query_embedding, chunks, top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["chunk_id"] == "a"  # Highest similarity
    assert ranked[0]["score"] > 0.99
    assert ranked[1]["chunk_id"] == "c"  # Second highest
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vertex.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.vertex'"

**Step 3: Write the implementation**

Create `app/vertex.py`:

```python
"""Vertex AI client for embeddings and LLM generation."""

import math
from typing import Any

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

from app.logging import get_logger

logger = get_logger(__name__)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def rank_chunks_by_similarity(
    query_embedding: list[float],
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank chunks by cosine similarity to query embedding."""
    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunk = {**chunk, "score": score}
        scored_chunks.append(scored_chunk)

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]


class VertexClient:
    """Client for Vertex AI embeddings and LLM generation."""

    def __init__(
        self,
        project_id: str,
        region: str,
        embedding_model: str = "text-embedding-004",
        llm_model: str = "gemini-1.5-flash",
    ):
        self.project_id = project_id
        self.region = region
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

        # Initialize models (lazy loaded on first use)
        self._embedding_model = None
        self._llm_model = None

    @property
    def embedding_model(self) -> TextEmbeddingModel:
        """Get or create embedding model instance."""
        if self._embedding_model is None:
            self._embedding_model = TextEmbeddingModel.from_pretrained(
                self.embedding_model_name
            )
        return self._embedding_model

    @property
    def llm_model(self) -> GenerativeModel:
        """Get or create LLM model instance."""
        if self._llm_model is None:
            self._llm_model = GenerativeModel(self.llm_model_name)
        return self._llm_model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Vertex AI has a limit on batch size
        batch_size = 250
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in embeddings])

        logger.info(
            f"Generated {len(all_embeddings)} embeddings",
            extra={"metrics": {"texts_embedded": len(texts)}},
        )

        return all_embeddings

    def generate(self, prompt: str) -> str:
        """Generate text using Gemini."""
        response = self.llm_model.generate_content(prompt)

        logger.info(
            "LLM generation completed",
            extra={
                "metrics": {
                    "prompt_chars": len(prompt),
                    "response_chars": len(response.text) if response.text else 0,
                }
            },
        )

        return response.text or ""

    def health_check(self) -> bool:
        """Check if Vertex AI is accessible."""
        try:
            # Simple test embedding
            self.embed_text("health check")
            return True
        except Exception as e:
            logger.error(f"Vertex AI health check failed: {e}")
            return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vertex.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/vertex.py tests/test_vertex.py
git commit -m "feat: add Vertex AI client for embeddings and LLM"
```

---

## Task 6: RAG Module

**Files:**
- Create: `app/rag.py`
- Create: `tests/test_rag.py`

**Step 1: Write the failing test**

Create `tests/test_rag.py`:

```python
"""Tests for RAG module."""

from unittest.mock import MagicMock


def test_build_prompt():
    """build_prompt should format chunks and question correctly."""
    from app.rag import build_prompt

    chunks = [
        {
            "source_uri": "gs://bucket/doc.pdf",
            "page": 1,
            "chunk_text": "First chunk content.",
        },
        {
            "source_uri": "gs://bucket/doc2.md",
            "page": None,
            "chunk_text": "Second chunk content.",
        },
    ]
    question = "What is the answer?"

    prompt = build_prompt(chunks, question)

    assert "First chunk content." in prompt
    assert "Second chunk content." in prompt
    assert "gs://bucket/doc.pdf" in prompt
    assert "page 1" in prompt
    assert "What is the answer?" in prompt


def test_extract_citations():
    """extract_citations should parse source references from answer."""
    from app.rag import extract_citations

    chunks = [
        {"chunk_id": "a_0", "source_uri": "gs://bucket/doc.pdf", "page": 1},
        {"chunk_id": "b_0", "source_uri": "gs://bucket/doc2.md", "page": None},
    ]

    # Answer that references the chunks
    answer = "Based on doc.pdf page 1, the answer is X. Also see doc2.md."

    citations = extract_citations(answer, chunks)

    # Should find citations for chunks that are referenced
    assert len(citations) >= 1


def test_truncate_chunk_text():
    """truncate_chunk_text should limit length for logging."""
    from app.rag import truncate_chunk_text

    long_text = "A" * 200
    truncated = truncate_chunk_text(long_text, max_length=100)

    assert len(truncated) <= 103  # 100 + "..."
    assert truncated.endswith("...")

    short_text = "Short text"
    assert truncate_chunk_text(short_text, max_length=100) == short_text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.rag'"

**Step 3: Write the implementation**

Create `app/rag.py`:

```python
"""RAG (Retrieval-Augmented Generation) logic."""

import time
from dataclasses import dataclass
from typing import Any

from app.bq import BigQueryClient
from app.vertex import VertexClient, rank_chunks_by_similarity
from app.logging import get_logger

logger = get_logger(__name__)


RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided documents.
Use ONLY the information from the chunks below. If the answer is not in the chunks, say "I cannot find this information in the provided documents."
Cite sources using [source_uri, page X] format when referencing specific information.

CHUNKS:
{chunks_formatted}

QUESTION: {question}

ANSWER:"""


def truncate_chunk_text(text: str, max_length: int = 100) -> str:
    """Truncate text for logging (avoid logging full documents)."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def build_prompt(chunks: list[dict[str, Any]], question: str) -> str:
    """Build the RAG prompt with chunks and question."""
    chunks_formatted = []

    for i, chunk in enumerate(chunks, 1):
        source = chunk["source_uri"]
        page = chunk.get("page")
        page_str = f", page {page}" if page else ""
        text = chunk["chunk_text"]

        chunks_formatted.append(f"[{i}] Source: {source}{page_str}\n{text}")

    return RAG_PROMPT_TEMPLATE.format(
        chunks_formatted="\n\n".join(chunks_formatted),
        question=question,
    )


def extract_citations(
    answer: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract citations from answer based on which chunks were likely used."""
    citations = []
    answer_lower = answer.lower()

    for chunk in chunks:
        source_uri = chunk["source_uri"]
        # Check if source filename is mentioned in answer
        filename = source_uri.split("/")[-1].lower()

        if filename in answer_lower or source_uri.lower() in answer_lower:
            citations.append({
                "source_uri": source_uri,
                "page": chunk.get("page"),
                "chunk_id": chunk["chunk_id"],
            })

    # If no explicit citations found, include top chunks as likely sources
    if not citations and chunks:
        for chunk in chunks[:3]:  # Top 3 as fallback
            citations.append({
                "source_uri": chunk["source_uri"],
                "page": chunk.get("page"),
                "chunk_id": chunk["chunk_id"],
            })

    # Deduplicate by chunk_id
    seen = set()
    unique_citations = []
    for c in citations:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique_citations.append(c)

    return unique_citations


@dataclass
class RAGResponse:
    """Response from RAG query."""

    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    request_id: str
    latency_ms: int


class RAGEngine:
    """RAG engine for document Q&A."""

    def __init__(self, bq_client: BigQueryClient, vertex_client: VertexClient):
        self.bq_client = bq_client
        self.vertex_client = vertex_client

    def query(
        self,
        question: str,
        top_k: int = 5,
        request_id: str = "",
    ) -> RAGResponse:
        """Execute RAG query and return answer with citations."""
        start_time = time.time()

        # Step 1: Embed the question
        embed_start = time.time()
        query_embedding = self.vertex_client.embed_text(question)
        embed_latency = int((time.time() - embed_start) * 1000)

        # Step 2: Fetch all chunks from BigQuery
        retrieval_start = time.time()
        all_chunks = self.bq_client.get_all_chunks()
        retrieval_latency = int((time.time() - retrieval_start) * 1000)

        if not all_chunks:
            logger.warning(
                "No chunks found in database",
                extra={"request_id": request_id},
            )
            return RAGResponse(
                answer="No documents have been indexed yet. Please run the ingestion process first.",
                citations=[],
                retrieved_chunks=[],
                request_id=request_id,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        # Step 3: Rank chunks by similarity
        ranked_chunks = rank_chunks_by_similarity(
            query_embedding, all_chunks, top_k=top_k
        )

        # Step 4: Build prompt and generate answer
        llm_start = time.time()
        prompt = build_prompt(ranked_chunks, question)
        answer = self.vertex_client.generate(prompt)
        llm_latency = int((time.time() - llm_start) * 1000)

        # Step 5: Extract citations
        citations = extract_citations(answer, ranked_chunks)

        total_latency = int((time.time() - start_time) * 1000)

        # Prepare retrieved chunks for response (remove embedding for size)
        response_chunks = []
        for chunk in ranked_chunks:
            response_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "source_uri": chunk["source_uri"],
                "page": chunk.get("page"),
                "chunk_text": chunk["chunk_text"],
                "score": round(chunk["score"], 4),
            })

        logger.info(
            "RAG query completed",
            extra={
                "request_id": request_id,
                "metrics": {
                    "embedding_latency_ms": embed_latency,
                    "retrieval_latency_ms": retrieval_latency,
                    "llm_latency_ms": llm_latency,
                    "total_latency_ms": total_latency,
                    "chunks_retrieved": len(ranked_chunks),
                    "chunks_total": len(all_chunks),
                },
                "context": {
                    "top_k": top_k,
                    "question_truncated": truncate_chunk_text(question),
                },
            },
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=response_chunks,
            request_id=request_id,
            latency_ms=total_latency,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "feat: add RAG engine for retrieval and generation"
```

---

## Task 7: FastAPI Application

**Files:**
- Create: `app/main.py`
- Create: `tests/test_health.py`
- Create: `tests/test_rag_smoke.py`

**Step 1: Write the failing test for health endpoint**

Create `tests/test_health.py`:

```python
"""Tests for health endpoint."""

import os
from unittest.mock import MagicMock, patch


def test_health_endpoint_returns_ok():
    """Health endpoint should return status ok."""
    # Mock the GCP clients
    with patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }):
        with patch("app.main.BigQueryClient") as mock_bq:
            with patch("app.main.VertexClient") as mock_vertex:
                mock_bq_instance = MagicMock()
                mock_bq_instance.health_check.return_value = True
                mock_bq.return_value = mock_bq_instance

                mock_vertex_instance = MagicMock()
                mock_vertex_instance.health_check.return_value = True
                mock_vertex.return_value = mock_vertex_instance

                from fastapi.testclient import TestClient
                from app.main import create_app

                app = create_app()
                client = TestClient(app)

                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
                assert "version" in data
```

**Step 2: Write the failing test for RAG smoke test**

Create `tests/test_rag_smoke.py`:

```python
"""Smoke tests for RAG endpoint."""

import os
from unittest.mock import MagicMock, patch


def test_ask_endpoint_returns_answer():
    """Ask endpoint should return answer with citations."""
    with patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }):
        with patch("app.main.BigQueryClient") as mock_bq:
            with patch("app.main.VertexClient") as mock_vertex:
                # Setup mocks
                mock_bq_instance = MagicMock()
                mock_bq_instance.get_all_chunks.return_value = [
                    {
                        "doc_id": "abc",
                        "chunk_id": "abc_0",
                        "source_uri": "gs://bucket/doc.pdf",
                        "page": 1,
                        "chunk_index": 0,
                        "chunk_text": "Test content about refunds.",
                        "embedding": [0.1] * 768,
                    }
                ]
                mock_bq.return_value = mock_bq_instance

                mock_vertex_instance = MagicMock()
                mock_vertex_instance.embed_text.return_value = [0.1] * 768
                mock_vertex_instance.generate.return_value = "The refund policy states..."
                mock_vertex.return_value = mock_vertex_instance

                from fastapi.testclient import TestClient
                from app.main import create_app

                app = create_app()
                client = TestClient(app)

                response = client.post("/ask", json={
                    "question": "What is the refund policy?",
                    "top_k": 5,
                })

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "citations" in data
                assert "retrieved_chunks" in data
                assert "request_id" in data


def test_sources_endpoint():
    """Sources endpoint should return document list."""
    with patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }):
        with patch("app.main.BigQueryClient") as mock_bq:
            with patch("app.main.VertexClient"):
                mock_bq_instance = MagicMock()
                mock_bq_instance.get_sources.return_value = [
                    {"doc_id": "abc", "source_uri": "gs://bucket/doc.pdf", "chunk_count": 5}
                ]
                mock_bq_instance.get_total_chunk_count.return_value = 5
                mock_bq.return_value = mock_bq_instance

                from fastapi.testclient import TestClient
                from app.main import create_app

                app = create_app()
                client = TestClient(app)

                response = client.get("/sources")

                assert response.status_code == 200
                data = response.json()
                assert "sources" in data
                assert "total_chunks" in data
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_health.py tests/test_rag_smoke.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.main'"

**Step 4: Write the implementation**

Create `app/main.py`:

```python
"""FastAPI application for GCP RAG Copilot."""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logging import setup_logging, get_logger, RequestContext
from app.bq import BigQueryClient
from app.vertex import VertexClient
from app.rag import RAGEngine

logger = get_logger(__name__)

# Global instances (initialized on startup)
_bq_client: BigQueryClient | None = None
_vertex_client: VertexClient | None = None
_rag_engine: RAGEngine | None = None


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    """Citation reference in response."""

    source_uri: str
    page: int | None
    chunk_id: str


class RetrievedChunk(BaseModel):
    """Retrieved chunk in response."""

    chunk_id: str
    source_uri: str
    page: int | None
    chunk_text: str
    score: float


class AskResponse(BaseModel):
    """Response body for /ask endpoint."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    request_id: str
    latency_ms: int


class SourceInfo(BaseModel):
    """Source document info."""

    doc_id: str
    source_uri: str
    chunk_count: int


class SourcesResponse(BaseModel):
    """Response body for /sources endpoint."""

    sources: list[SourceInfo]
    total_chunks: int


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    version: str
    bq_connected: bool
    vertex_available: bool


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    request_id: str


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup and shutdown events."""
        global _bq_client, _vertex_client, _rag_engine

        # Setup logging
        setup_logging(level=settings.log_level)
        logger.info("Starting GCP RAG Copilot", extra={
            "context": {
                "project_id": settings.gcp_project_id,
                "region": settings.gcp_region,
                "version": settings.app_version,
            }
        })

        # Initialize clients
        _bq_client = BigQueryClient(
            project_id=settings.gcp_project_id,
            dataset=settings.bq_dataset,
            table=settings.bq_table,
        )

        _vertex_client = VertexClient(
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
        )

        _rag_engine = RAGEngine(
            bq_client=_bq_client,
            vertex_client=_vertex_client,
        )

        logger.info("Application startup complete")
        yield
        logger.info("Application shutdown")

    app = FastAPI(
        title="GCP RAG Copilot",
        description="Document Q&A with RAG on Google Cloud",
        version=settings.app_version,
        lifespan=lifespan,
    )

    @app.post(
        "/ask",
        response_model=AskResponse,
        responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def ask(request: AskRequest) -> AskResponse:
        """Answer a question using RAG over indexed documents."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        with RequestContext(request_id=request_id):
            try:
                if _rag_engine is None:
                    raise HTTPException(status_code=500, detail="RAG engine not initialized")

                result = _rag_engine.query(
                    question=request.question,
                    top_k=request.top_k,
                    request_id=request_id,
                )

                return AskResponse(
                    answer=result.answer,
                    citations=[Citation(**c) for c in result.citations],
                    retrieved_chunks=[RetrievedChunk(**c) for c in result.retrieved_chunks],
                    request_id=result.request_id,
                    latency_ms=result.latency_ms,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=str(e),
                )

    @app.get("/sources", response_model=SourcesResponse)
    async def get_sources() -> SourcesResponse:
        """Get list of indexed document sources."""
        if _bq_client is None:
            raise HTTPException(status_code=500, detail="BigQuery client not initialized")

        sources = _bq_client.get_sources()
        total_chunks = _bq_client.get_total_chunk_count()

        return SourcesResponse(
            sources=[SourceInfo(**s) for s in sources],
            total_chunks=total_chunks,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        settings = get_settings()

        bq_ok = _bq_client.health_check() if _bq_client else False
        vertex_ok = _vertex_client.health_check() if _vertex_client else False

        return HealthResponse(
            status="ok" if (bq_ok and vertex_ok) else "degraded",
            version=settings.app_version,
            bq_connected=bq_ok,
            vertex_available=vertex_ok,
        )

    return app


# Create app instance for uvicorn
app = create_app()
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_health.py tests/test_rag_smoke.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/main.py tests/test_health.py tests/test_rag_smoke.py
git commit -m "feat: add FastAPI application with /ask, /sources, /health endpoints"
```

---

## Task 8: Chunking Module

**Files:**
- Create: `ingestion/__init__.py`
- Create: `ingestion/chunking.py`
- Create: `tests/test_chunking.py`

**Step 1: Create ingestion/__init__.py**

```python
"""Ingestion package for GCP RAG Copilot."""
```

**Step 2: Write the failing test**

Create `tests/test_chunking.py`:

```python
"""Tests for chunking module."""


def test_chunk_text_creates_overlapping_chunks():
    """chunk_text should create overlapping chunks."""
    from ingestion.chunking import chunk_text

    text = "A" * 2500  # Long enough for multiple chunks
    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    assert len(chunks) >= 2
    # Check overlap exists
    assert chunks[0]["text"][-200:] == chunks[1]["text"][:200]


def test_chunk_text_preserves_short_text():
    """Short text should produce single chunk."""
    from ingestion.chunking import chunk_text

    text = "Short text"
    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_chunk_text_includes_indices():
    """Chunks should include chunk_index."""
    from ingestion.chunking import chunk_text

    text = "A" * 3000
    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_extract_text_from_txt():
    """extract_text should handle txt files."""
    from ingestion.chunking import extract_text

    content = b"Hello, world!"
    text, pages = extract_text(content, "file.txt")

    assert text == "Hello, world!"
    assert pages == [{"page": None, "text": "Hello, world!"}]


def test_extract_text_from_markdown():
    """extract_text should handle markdown files."""
    from ingestion.chunking import extract_text

    content = b"# Header\n\nParagraph text."
    text, pages = extract_text(content, "file.md")

    assert "Header" in text
    assert "Paragraph" in text
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ingestion.chunking'"

**Step 4: Write the implementation**

Create `ingestion/chunking.py`:

```python
"""Text extraction and chunking for document ingestion."""

import io
from typing import Any

from pypdf import PdfReader

from app.logging import get_logger

logger = get_logger(__name__)


def extract_text(content: bytes, filename: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract text from document content.

    Returns:
        Tuple of (full_text, pages) where pages is a list of
        {"page": int|None, "text": str}
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return _extract_pdf(content)
    elif filename_lower.endswith((".txt", ".md", ".markdown")):
        return _extract_plain_text(content)
    else:
        logger.warning(f"Unknown file type: {filename}, treating as plain text")
        return _extract_plain_text(content)


def _extract_pdf(content: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract text from PDF content."""
    reader = PdfReader(io.BytesIO(content))
    pages = []
    full_text_parts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if page_text:
            pages.append({"page": i + 1, "text": page_text})
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts)
    logger.info(f"Extracted {len(pages)} pages from PDF")

    return full_text, pages


def _extract_plain_text(content: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract text from plain text content."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    text = text.strip()
    pages = [{"page": None, "text": text}]

    return text, pages


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk (smaller chunks are discarded)

    Returns:
        List of {"chunk_index": int, "text": str}
    """
    if len(text) <= chunk_size:
        return [{"chunk_index": 0, "text": text}]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        # If this isn't the last chunk, try to break at a sentence/word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ?) within last 100 chars
            boundary_search_start = max(start + chunk_size - 100, start)
            best_boundary = end

            for i in range(end - 1, boundary_search_start - 1, -1):
                if text[i] in ".!?\n":
                    best_boundary = i + 1
                    break

            # If no sentence boundary, try word boundary
            if best_boundary == end:
                for i in range(end - 1, boundary_search_start - 1, -1):
                    if text[i] == " ":
                        best_boundary = i + 1
                        break

            end = best_boundary

        chunk_text = text[start:end].strip()

        # Only add chunk if it meets minimum size
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else end

        # Prevent infinite loop
        if end >= len(text):
            break

    logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
    return chunks


def chunk_document_with_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Chunk document while preserving page information.

    For PDFs, chunks include page number.
    For text files, page is None.

    Returns:
        List of {"chunk_index": int, "text": str, "page": int|None}
    """
    all_chunks = []
    chunk_index = 0

    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

        for chunk in page_chunks:
            all_chunks.append({
                "chunk_index": chunk_index,
                "text": chunk["text"],
                "page": page_num,
            })
            chunk_index += 1

    return all_chunks
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add ingestion/__init__.py ingestion/chunking.py tests/test_chunking.py
git commit -m "feat: add text extraction and chunking module"
```

---

## Task 9: Ingestion Script

**Files:**
- Create: `ingestion/ingest.py`

**Step 1: Write the ingestion script**

Create `ingestion/ingest.py`:

```python
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
```

**Step 2: Commit**

```bash
git add ingestion/ingest.py
git commit -m "feat: add document ingestion script"
```

---

## Task 10: Dockerfile

**Files:**
- Create: `docker/Dockerfile`

**Step 1: Write the Dockerfile**

Create `docker/Dockerfile`:

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (for pypdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ ./app/
COPY ingestion/ ./ingestion/

# Create non-root user for security
RUN useradd --create-home appuser
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Step 2: Commit**

```bash
git add docker/Dockerfile
git commit -m "feat: add Dockerfile for Cloud Run deployment"
```

---

## Task 11: Infrastructure Scripts

**Files:**
- Create: `infra/create_resources.sh`
- Create: `infra/deploy_cloudrun.sh`
- Create: `infra/run_ingest.sh`

**Step 1: Create create_resources.sh**

Create `infra/create_resources.sh`:

```bash
#!/bin/bash
# Create GCP resources for RAG Copilot
# Usage: ./create_resources.sh

set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${GCP_REGION:=us-central1}"
: "${BQ_DATASET:=rag_copilot}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"

echo "=== Creating GCP Resources for RAG Copilot ==="
echo "Project: $GCP_PROJECT_ID"
echo "Region: $GCP_REGION"
echo "Dataset: $BQ_DATASET"
echo "Bucket: $GCS_BUCKET"
echo ""

# Set project
gcloud config set project "$GCP_PROJECT_ID"

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

# Create GCS bucket
echo "Creating GCS bucket..."
gcloud storage buckets create "gs://$GCS_BUCKET" \
    --location="$GCP_REGION" \
    --uniform-bucket-level-access \
    2>/dev/null || echo "Bucket already exists or error occurred"

# Create BigQuery dataset
echo "Creating BigQuery dataset..."
bq --location=US mk --dataset \
    --description "RAG Copilot document chunks" \
    "$GCP_PROJECT_ID:$BQ_DATASET" \
    2>/dev/null || echo "Dataset already exists or error occurred"

# Create service accounts
echo "Creating service accounts..."

# API service account
gcloud iam service-accounts create rag-api \
    --display-name="RAG Copilot API" \
    2>/dev/null || echo "Service account rag-api already exists"

# Ingestion service account
gcloud iam service-accounts create rag-ingest \
    --display-name="RAG Copilot Ingestion" \
    2>/dev/null || echo "Service account rag-ingest already exists"

# Assign roles to API service account
echo "Assigning roles to rag-api..."
API_SA="rag-api@$GCP_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$API_SA" \
    --role="roles/bigquery.dataViewer" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$API_SA" \
    --role="roles/bigquery.jobUser" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$API_SA" \
    --role="roles/aiplatform.user" \
    --quiet

# Assign roles to Ingestion service account
echo "Assigning roles to rag-ingest..."
INGEST_SA="rag-ingest@$GCP_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/bigquery.dataEditor" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/bigquery.jobUser" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/storage.objectViewer" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/aiplatform.user" \
    --quiet

# Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create rag-copilot \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="RAG Copilot container images" \
    2>/dev/null || echo "Repository already exists"

echo ""
echo "=== Resource Creation Complete ==="
echo ""
echo "Service Accounts:"
echo "  API: $API_SA"
echo "  Ingestion: $INGEST_SA"
echo ""
echo "Next steps:"
echo "  1. Upload documents to gs://$GCS_BUCKET/documents/"
echo "  2. Run ./infra/run_ingest.sh to ingest documents"
echo "  3. Run ./infra/deploy_cloudrun.sh to deploy the API"
```

**Step 2: Create deploy_cloudrun.sh**

Create `infra/deploy_cloudrun.sh`:

```bash
#!/bin/bash
# Build and deploy to Cloud Run
# Usage: ./deploy_cloudrun.sh

set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${GCP_REGION:=us-central1}"
: "${BQ_DATASET:=rag_copilot}"
: "${BQ_TABLE:=chunks}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"
: "${EMBEDDING_MODEL:=text-embedding-004}"
: "${LLM_MODEL:=gemini-1.5-flash}"

SERVICE_NAME="rag-copilot-api"
IMAGE_NAME="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/rag-copilot/api"
API_SA="rag-api@$GCP_PROJECT_ID.iam.gserviceaccount.com"

echo "=== Deploying RAG Copilot to Cloud Run ==="
echo "Project: $GCP_PROJECT_ID"
echo "Region: $GCP_REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Set project
gcloud config set project "$GCP_PROJECT_ID"

# Configure Docker for Artifact Registry
echo "Configuring Docker..."
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev" --quiet

# Build the image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:latest" -f docker/Dockerfile .

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push "$IMAGE_NAME:latest"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE_NAME:latest" \
    --platform=managed \
    --region="$GCP_REGION" \
    --service-account="$API_SA" \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --timeout=300 \
    --concurrency=80 \
    --min-instances=0 \
    --max-instances=10 \
    --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_REGION=$GCP_REGION,BQ_DATASET=$BQ_DATASET,BQ_TABLE=$BQ_TABLE,GCS_BUCKET=$GCS_BUCKET,EMBEDDING_MODEL=$EMBEDDING_MODEL,LLM_MODEL=$LLM_MODEL,LOG_LEVEL=INFO"

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --platform=managed \
    --region="$GCP_REGION" \
    --format="value(status.url)")

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the API:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "  curl -X POST $SERVICE_URL/ask \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"What is in the documents?\", \"top_k\": 5}'"
```

**Step 3: Create run_ingest.sh**

Create `infra/run_ingest.sh`:

```bash
#!/bin/bash
# Run document ingestion
# Usage: ./run_ingest.sh [local|cloud]

set -euo pipefail

MODE="${1:-local}"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${GCP_REGION:=us-central1}"
: "${BQ_DATASET:=rag_copilot}"
: "${BQ_TABLE:=chunks}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"
: "${GCS_PREFIX:=documents/}"
: "${EMBEDDING_MODEL:=text-embedding-004}"
: "${LLM_MODEL:=gemini-1.5-flash}"

echo "=== Running Document Ingestion ==="
echo "Mode: $MODE"
echo "Bucket: gs://$GCS_BUCKET/$GCS_PREFIX"
echo ""

if [ "$MODE" == "local" ]; then
    # Run locally (requires gcloud auth)
    echo "Running ingestion locally..."

    # Ensure dependencies are installed
    pip install -r requirements.txt --quiet

    # Run the ingestion script
    python -m ingestion.ingest

elif [ "$MODE" == "cloud" ]; then
    # Run as Cloud Run Job
    echo "Running ingestion as Cloud Run Job..."

    JOB_NAME="rag-copilot-ingest"
    IMAGE_NAME="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/rag-copilot/api"
    INGEST_SA="rag-ingest@$GCP_PROJECT_ID.iam.gserviceaccount.com"

    # Set project
    gcloud config set project "$GCP_PROJECT_ID"

    # Create or update Cloud Run Job
    gcloud run jobs deploy "$JOB_NAME" \
        --image="$IMAGE_NAME:latest" \
        --region="$GCP_REGION" \
        --service-account="$INGEST_SA" \
        --memory=2Gi \
        --cpu=2 \
        --task-timeout=3600 \
        --max-retries=1 \
        --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_REGION=$GCP_REGION,BQ_DATASET=$BQ_DATASET,BQ_TABLE=$BQ_TABLE,GCS_BUCKET=$GCS_BUCKET,GCS_PREFIX=$GCS_PREFIX,EMBEDDING_MODEL=$EMBEDDING_MODEL,LLM_MODEL=$LLM_MODEL,LOG_LEVEL=INFO" \
        --command="python,-m,ingestion.ingest"

    # Execute the job
    echo "Executing Cloud Run Job..."
    gcloud run jobs execute "$JOB_NAME" \
        --region="$GCP_REGION" \
        --wait

else
    echo "Usage: ./run_ingest.sh [local|cloud]"
    exit 1
fi

echo ""
echo "=== Ingestion Complete ==="
```

**Step 4: Make scripts executable and commit**

```bash
chmod +x infra/create_resources.sh infra/deploy_cloudrun.sh infra/run_ingest.sh
git add infra/
git commit -m "feat: add infrastructure scripts for GCP resource management"
```

---

## Task 12: GitHub Actions CI/CD

**Files:**
- Create: `.github/workflows/deploy.yml`

**Step 1: Create deploy.yml**

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  GCP_REGION: us-central1
  SERVICE_NAME: rag-copilot-api
  REPOSITORY: rag-copilot

jobs:
  deploy:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      id-token: write  # Required for Workload Identity Federation

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ -v --tb=short
        env:
          GCP_PROJECT_ID: test-project
          GCS_BUCKET: test-bucket

      # Authenticate using Workload Identity Federation
      # See: https://github.com/google-github-actions/auth#setup
      - name: Authenticate to Google Cloud
        id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker
        run: |
          gcloud auth configure-docker ${{ env.GCP_REGION }}-docker.pkg.dev --quiet

      - name: Build Docker image
        run: |
          docker build \
            -t ${{ env.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT_ID }}/${{ env.REPOSITORY }}/api:${{ github.sha }} \
            -t ${{ env.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT_ID }}/${{ env.REPOSITORY }}/api:latest \
            -f docker/Dockerfile .

      - name: Push Docker image
        run: |
          docker push ${{ env.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT_ID }}/${{ env.REPOSITORY }}/api:${{ github.sha }}
          docker push ${{ env.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT_ID }}/${{ env.REPOSITORY }}/api:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image=${{ env.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT_ID }}/${{ env.REPOSITORY }}/api:${{ github.sha }} \
            --platform=managed \
            --region=${{ env.GCP_REGION }} \
            --service-account=rag-api@${{ vars.GCP_PROJECT_ID }}.iam.gserviceaccount.com \
            --allow-unauthenticated \
            --memory=1Gi \
            --cpu=1 \
            --timeout=300 \
            --set-env-vars="GCP_PROJECT_ID=${{ vars.GCP_PROJECT_ID }},GCP_REGION=${{ env.GCP_REGION }},BQ_DATASET=${{ vars.BQ_DATASET }},BQ_TABLE=chunks,GCS_BUCKET=${{ vars.GCS_BUCKET }},EMBEDDING_MODEL=text-embedding-004,LLM_MODEL=gemini-1.5-flash,LOG_LEVEL=INFO"

      - name: Get service URL
        run: |
          SERVICE_URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} \
            --platform=managed \
            --region=${{ env.GCP_REGION }} \
            --format="value(status.url)")
          echo "Service deployed at: $SERVICE_URL"

      - name: Smoke test
        run: |
          SERVICE_URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} \
            --platform=managed \
            --region=${{ env.GCP_REGION }} \
            --format="value(status.url)")

          # Wait for service to be ready
          sleep 10

          # Check health endpoint
          curl -f "$SERVICE_URL/health" || exit 1
          echo "Health check passed!"
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/deploy.yml
git commit -m "feat: add GitHub Actions CI/CD workflow"
```

---

## Task 13: Sample Documents

**Files:**
- Create: `sample_docs/sample1.txt`
- Create: `sample_docs/sample2.md`

**Step 1: Create sample documents**

Create `sample_docs/sample1.txt`:

```text
Company Return Policy

Our company offers a 30-day return policy for all products. To be eligible for a return, items must be unused and in their original packaging.

To initiate a return:
1. Contact our customer service team within 30 days of purchase
2. Provide your order number and reason for return
3. Ship the item back using the provided return label

Refunds will be processed within 5-7 business days after we receive the returned item. The refund will be credited to the original payment method.

Exceptions:
- Sale items are final sale and cannot be returned
- Personalized items cannot be returned
- Digital products are non-refundable

For questions, contact support@example.com
```

Create `sample_docs/sample2.md`:

```markdown
# Product Documentation

## Getting Started

Welcome to our product! This guide will help you get started quickly.

### Installation

1. Download the installer from our website
2. Run the installer and follow the prompts
3. Launch the application

### Configuration

The application can be configured via the settings menu:

- **Theme**: Choose between light and dark mode
- **Language**: Select your preferred language
- **Notifications**: Enable or disable notifications

### Features

Our product includes the following key features:

1. **Dashboard**: View your data at a glance
2. **Reports**: Generate detailed reports
3. **Export**: Export data to CSV or PDF
4. **Integration**: Connect with third-party services

## Troubleshooting

If you encounter issues:

1. Check our FAQ at help.example.com
2. Restart the application
3. Contact support if the issue persists

## Support

For technical support, email techsupport@example.com or call 1-800-EXAMPLE.
```

**Step 2: Commit**

```bash
mkdir -p sample_docs
git add sample_docs/
git commit -m "feat: add sample documents for testing"
```

---

## Task 14: Documentation Files

**Files:**
- Create: `README.md`
- Create: `SECURITY.md`
- Create: `runbook.md`

**Step 1: Create README.md**

Create `README.md`:

```markdown
# GCP RAG Copilot

A production-ready document Q&A copilot deployed on Google Cloud Platform. Ingests PDFs/TXT/MD from Cloud Storage, creates embeddings via Vertex AI, stores chunks in BigQuery, and answers questions using Gemini with citations.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    GCS      │────▶│  Cloud Run  │────▶│  Cloud Run  │
│   Bucket    │     │    Job      │     │    API      │
│  (docs)     │     │  (ingest)   │     │  (FastAPI)  │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                   │
                    embed  │                   │ embed + generate
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Vertex AI  │     │  Vertex AI  │
                    │  Embeddings │     │   Gemini    │
                    └──────┬──────┘     └─────────────┘
                           │                   ▲
                    store  │                   │ retrieve
                           ▼                   │
                    ┌──────────────────────────┘
                    │        BigQuery          │
                    │    (chunks table)        │
                    └──────────────────────────┘
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
| `LLM_MODEL` | No | gemini-1.5-flash | Vertex AI LLM |
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
```

**Step 2: Create SECURITY.md**

Create `SECURITY.md`:

```markdown
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
```

**Step 3: Create runbook.md**

Create `runbook.md`:

```markdown
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
```

**Step 4: Commit**

```bash
git add README.md SECURITY.md runbook.md
git commit -m "docs: add README, SECURITY guide, and operations runbook"
```

---

## Task 15: Final Verification

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Verify file structure**

```bash
find . -type f -name "*.py" -o -name "*.sh" -o -name "*.yml" -o -name "*.md" | head -30
```

**Step 3: Final commit with all files**

```bash
git status
git log --oneline
```

Verify all files are committed.

---

## Summary

This plan creates a complete GCP RAG Copilot with:

- **15 tasks** covering all components
- **TDD approach** for core modules
- **Complete code** in each step
- **Exact file paths** throughout
- **Commit after each task**

Total files created: ~25 files across app/, ingestion/, infra/, tests/, .github/, sample_docs/, and root documentation.
