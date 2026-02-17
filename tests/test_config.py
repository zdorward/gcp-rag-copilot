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
        "LLM_MODEL": "gemini-2.0-flash",
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
        assert settings.llm_model == "gemini-2.0-flash"
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
        assert settings.llm_model == "gemini-2.0-flash"
        assert settings.log_level == "INFO"
        assert settings.app_version == "1.0.0"
