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
