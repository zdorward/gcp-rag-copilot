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
            llm_model="gemini-2.0-flash",
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
