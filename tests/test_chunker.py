"""Tests for section-aware chunking logic."""

from src.ingestion.chunker import ChunkerConfig, chunk_pages
from src.ingestion.loader import DocumentPage


def _make_pages(text: str, source: str = "test.pdf") -> list[DocumentPage]:
    """Create a single-page document from raw text."""
    return [DocumentPage(text=text, page_number=1, source_file=source, total_pages=1)]


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_pages([]) == []


def test_short_content_below_min_tokens_is_discarded() -> None:
    pages = _make_pages("Too short.")
    config = ChunkerConfig(min_chunk_tokens=100)
    assert chunk_pages(pages, config) == []


def test_single_section_produces_one_chunk() -> None:
    text = "RECOMMENDATIONS\n" + "Follow evidence-based practice. " * 40
    pages = _make_pages(text)
    chunks = chunk_pages(pages)
    assert len(chunks) >= 1
    assert chunks[0].heading == "RECOMMENDATIONS"
    assert chunks[0].source_file == "test.pdf"


def test_multiple_headings_produce_multiple_chunks() -> None:
    text = (
        "BACKGROUND\n" + "Context for the guideline. " * 30 + "\n"
        "METHODS\n" + "How the evidence was reviewed. " * 30 + "\n"
        "RECOMMENDATIONS\n" + "Specific clinical recommendations. " * 30
    )
    pages = _make_pages(text)
    chunks = chunk_pages(pages)
    headings = [c.heading for c in chunks]
    assert "BACKGROUND" in headings
    assert "METHODS" in headings
    assert "RECOMMENDATIONS" in headings


def test_large_section_splits_on_sentence_boundaries() -> None:
    # Create a section larger than max_chunk_tokens
    sentence = "This is a complete sentence about clinical evidence. "
    long_content = "EVIDENCE REVIEW\n" + sentence * 200
    pages = _make_pages(long_content)
    config = ChunkerConfig(max_chunk_tokens=100)
    chunks = chunk_pages(pages, config)
    assert len(chunks) > 1
    # All chunks should reference the same heading
    assert all(c.heading == "EVIDENCE REVIEW" for c in chunks)


def test_chunk_indices_are_sequential() -> None:
    text = (
        "SECTION ONE\n" + "Content here. " * 30 + "\n"
        "SECTION TWO\n" + "More content. " * 30
    )
    pages = _make_pages(text)
    chunks = chunk_pages(pages)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
