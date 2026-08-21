"""Section-aware document chunking for clinical guidelines.

Rather than splitting on arbitrary token counts, this module splits on
structural boundaries (headings, section breaks) so that each chunk
represents a coherent unit of clinical guidance.
"""

import re
from dataclasses import dataclass, field

from src.ingestion.loader import DocumentPage

# Patterns that indicate a new section in clinical guidelines
_CLINICAL_KEYWORDS = r"^(?:Recommendation|Summary|Background|Methods|Evidence|Discussion)\b"
_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z ]{4,}$", re.MULTILINE),
    re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE),
    re.compile(_CLINICAL_KEYWORDS, re.MULTILINE | re.IGNORECASE),
]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A self-contained chunk of guideline text with provenance."""

    text: str
    heading: str
    source_file: str
    page_numbers: tuple[int, ...]
    chunk_index: int


@dataclass
class ChunkerConfig:
    """Configuration for the section-aware chunker."""

    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 50
    overlap_sentences: int = 1
    heading_patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: list(_HEADING_PATTERNS)
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def _find_headings(text: str, patterns: list[re.Pattern[str]]) -> list[tuple[int, str]]:
    """Find all heading positions and their text."""
    headings: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            headings.append((match.start(), match.group().strip()))
    headings.sort(key=lambda h: h[0])
    return headings


def chunk_pages(pages: list[DocumentPage], config: ChunkerConfig | None = None) -> list[Chunk]:
    """Split document pages into section-aware chunks.

    Strategy:
    1. Concatenate all page text (preserving page boundary markers)
    2. Split on detected headings
    3. If a section exceeds max_chunk_tokens, split on sentence boundaries
    4. Discard chunks below min_chunk_tokens (likely noise)

    Args:
        pages: List of DocumentPage objects from a single document.
        config: Chunking configuration. Uses defaults if None.

    Returns:
        List of Chunk objects with provenance metadata.
    """
    if config is None:
        config = ChunkerConfig()

    if not pages:
        return []

    source_file = pages[0].source_file
    full_text = "\n\n".join(p.text for p in pages)
    headings = _find_headings(full_text, config.heading_patterns)

    # Split text into sections at heading boundaries
    sections: list[tuple[str, str]] = []  # (heading, content)
    if not headings:
        sections.append(("Document", full_text))
    else:
        # Content before first heading
        if headings[0][0] > 0:
            sections.append(("Preamble", full_text[: headings[0][0]].strip()))
        for i, (pos, heading_text) in enumerate(headings):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
            content = full_text[pos + len(heading_text) : end].strip()
            sections.append((heading_text, content))

    # Build chunks, splitting large sections on sentence boundaries
    chunks: list[Chunk] = []
    chunk_idx = 0

    for heading, content in sections:
        if _estimate_tokens(content) < config.min_chunk_tokens:
            continue

        if _estimate_tokens(content) <= config.max_chunk_tokens:
            chunks.append(
                Chunk(
                    text=content,
                    heading=heading,
                    source_file=source_file,
                    page_numbers=tuple(p.page_number for p in pages),
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1
        else:
            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", content)
            current_sentences: list[str] = []
            current_tokens = 0

            for sentence in sentences:
                sent_tokens = _estimate_tokens(sentence)
                if current_tokens + sent_tokens > config.max_chunk_tokens and current_sentences:
                    chunks.append(
                        Chunk(
                            text=" ".join(current_sentences),
                            heading=heading,
                            source_file=source_file,
                            page_numbers=tuple(p.page_number for p in pages),
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                    # Keep overlap
                    current_sentences = current_sentences[-config.overlap_sentences :]
                    current_tokens = sum(_estimate_tokens(s) for s in current_sentences)

                current_sentences.append(sentence)
                current_tokens += sent_tokens

            remaining_text = " ".join(current_sentences)
            if current_sentences and _estimate_tokens(remaining_text) >= config.min_chunk_tokens:
                chunks.append(
                    Chunk(
                        text=remaining_text,
                        heading=heading,
                        source_file=source_file,
                        page_numbers=tuple(p.page_number for p in pages),
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1

    return chunks
