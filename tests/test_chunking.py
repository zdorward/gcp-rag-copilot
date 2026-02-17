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
