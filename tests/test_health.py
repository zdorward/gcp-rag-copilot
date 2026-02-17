"""Tests for health endpoint."""

import os
import sys
from unittest.mock import MagicMock, patch


def test_health_endpoint_returns_ok():
    """Health endpoint should return status ok."""
    # Clear cached imports to ensure fresh import
    modules_to_clear = [k for k in list(sys.modules.keys())
                        if k.startswith('app.main')]
    for mod in modules_to_clear:
        del sys.modules[mod]

    # Mock the GCP clients
    with patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GCS_BUCKET": "test-bucket",
    }):
        # Patch the classes where they are USED (in app.main namespace)
        with patch("app.main.BigQueryClient") as mock_bq_class:
            with patch("app.main.VertexClient") as mock_vertex_class:
                with patch("app.main.RAGEngine") as mock_rag_class:
                    # Create mock instances
                    mock_bq_instance = MagicMock()
                    mock_bq_instance.health_check.return_value = True
                    mock_bq_class.return_value = mock_bq_instance

                    mock_vertex_instance = MagicMock()
                    mock_vertex_instance.health_check.return_value = True
                    mock_vertex_class.return_value = mock_vertex_instance

                    mock_rag_instance = MagicMock()
                    mock_rag_class.return_value = mock_rag_instance

                    # Now import and create app
                    from fastapi.testclient import TestClient
                    from app.main import create_app

                    app = create_app()

                    # Use context manager to trigger lifespan events
                    with TestClient(app) as client:
                        response = client.get("/health")

                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "ok"
                        assert "version" in data
