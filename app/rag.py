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
