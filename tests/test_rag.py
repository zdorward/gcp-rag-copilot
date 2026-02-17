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
