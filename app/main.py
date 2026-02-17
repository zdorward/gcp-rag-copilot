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
