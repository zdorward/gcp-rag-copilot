"""Application configuration using pydantic-settings."""

from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
