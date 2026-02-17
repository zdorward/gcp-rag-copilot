"""Smoke tests for RAG endpoint."""

import os
import sys
from unittest.mock import MagicMock, patch


def test_ask_endpoint_returns_answer():
    """Ask endpoint should return answer with citations."""
    # Clear cached imports to ensure fresh import
    modules_to_clear = [k for k in list(sys.modules.keys())
                        if k.startswith('app.main')]
    for mod in modules_to_clear:
        del sys.modules[mod]

    with patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }):
        with patch("app.main.BigQueryClient") as mock_bq_class:
            with patch("app.main.VertexClient") as mock_vertex_class:
                with patch("app.main.RAGEngine") as mock_rag_class:
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
                    mock_bq_class.return_value = mock_bq_instance

                    mock_vertex_instance = MagicMock()
                    mock_vertex_instance.embed_text.return_value = [0.1] * 768
                    mock_vertex_instance.generate.return_value = "The refund policy states..."
                    mock_vertex_class.return_value = mock_vertex_instance

                    # Setup RAG mock to return proper response
                    mock_rag_instance = MagicMock()
                    mock_rag_response = MagicMock()
                    mock_rag_response.answer = "The refund policy states..."
                    mock_rag_response.citations = [
                        {"source_uri": "gs://bucket/doc.pdf", "page": 1, "chunk_id": "abc_0"}
                    ]
                    mock_rag_response.retrieved_chunks = [
                        {
                            "chunk_id": "abc_0",
                            "source_uri": "gs://bucket/doc.pdf",
                            "page": 1,
                            "chunk_text": "Test content about refunds.",
                            "score": 0.95,
                        }
                    ]
                    mock_rag_response.request_id = "req_test123"
                    mock_rag_response.latency_ms = 100
                    mock_rag_instance.query.return_value = mock_rag_response
                    mock_rag_class.return_value = mock_rag_instance

                    from fastapi.testclient import TestClient
                    from app.main import create_app

                    app = create_app()

                    # Use context manager to trigger lifespan events
                    with TestClient(app) as client:
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
    # Clear cached imports to ensure fresh import
    modules_to_clear = [k for k in list(sys.modules.keys())
                        if k.startswith('app.main')]
    for mod in modules_to_clear:
        del sys.modules[mod]

    with patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }):
        with patch("app.main.BigQueryClient") as mock_bq_class:
            with patch("app.main.VertexClient") as mock_vertex_class:
                with patch("app.main.RAGEngine") as mock_rag_class:
                    mock_bq_instance = MagicMock()
                    mock_bq_instance.get_sources.return_value = [
                        {"doc_id": "abc", "source_uri": "gs://bucket/doc.pdf", "chunk_count": 5}
                    ]
                    mock_bq_instance.get_total_chunk_count.return_value = 5
                    mock_bq_class.return_value = mock_bq_instance

                    mock_vertex_class.return_value = MagicMock()
                    mock_rag_class.return_value = MagicMock()

                    from fastapi.testclient import TestClient
                    from app.main import create_app

                    app = create_app()

                    # Use context manager to trigger lifespan events
                    with TestClient(app) as client:
                        response = client.get("/sources")

                        assert response.status_code == 200
                        data = response.json()
                        assert "sources" in data
                        assert "total_chunks" in data
