"""Text extraction and chunking for document ingestion."""

import io
from typing import Any

from pypdf import PdfReader

from app.logging import get_logger

logger = get_logger(__name__)


def extract_text(content: bytes, filename: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract text from document content.

    Returns:
        Tuple of (full_text, pages) where pages is a list of
        {"page": int|None, "text": str}
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return _extract_pdf(content)
    elif filename_lower.endswith((".txt", ".md", ".markdown")):
        return _extract_plain_text(content)
    else:
        logger.warning(f"Unknown file type: {filename}, treating as plain text")
        return _extract_plain_text(content)


def _extract_pdf(content: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract text from PDF content."""
    reader = PdfReader(io.BytesIO(content))
    pages = []
    full_text_parts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if page_text:
            pages.append({"page": i + 1, "text": page_text})
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts)
    logger.info(f"Extracted {len(pages)} pages from PDF")

    return full_text, pages


def _extract_plain_text(content: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract text from plain text content."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    text = text.strip()
    pages = [{"page": None, "text": text}]

    return text, pages


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk (smaller chunks are discarded)

    Returns:
        List of {"chunk_index": int, "text": str}
    """
    if len(text) <= chunk_size:
        return [{"chunk_index": 0, "text": text}]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        # If this isn't the last chunk, try to break at a sentence/word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ?) within last 100 chars
            boundary_search_start = max(start + chunk_size - 100, start)
            best_boundary = end

            for i in range(end - 1, boundary_search_start - 1, -1):
                if text[i] in ".!?\n":
                    best_boundary = i + 1
                    break

            # If no sentence boundary, try word boundary
            if best_boundary == end:
                for i in range(end - 1, boundary_search_start - 1, -1):
                    if text[i] == " ":
                        best_boundary = i + 1
                        break

            end = best_boundary

        chunk_text = text[start:end].strip()

        # Only add chunk if it meets minimum size
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else end

        # Prevent infinite loop
        if end >= len(text):
            break

    logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
    return chunks


def chunk_document_with_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Chunk document while preserving page information.

    For PDFs, chunks include page number.
    For text files, page is None.

    Returns:
        List of {"chunk_index": int, "text": str, "page": int|None}
    """
    all_chunks = []
    chunk_index = 0

    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

        for chunk in page_chunks:
            all_chunks.append({
                "chunk_index": chunk_index,
                "text": chunk["text"],
                "page": page_num,
            })
            chunk_index += 1

    return all_chunks
