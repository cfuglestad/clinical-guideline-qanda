"""Load clinical guideline PDFs and extract raw text with page metadata."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of extracted text with source metadata."""

    text: str
    page_number: int
    source_file: str
    total_pages: int


def load_pdf(path: Path) -> list[DocumentPage]:
    """Extract text from all pages of a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        List of DocumentPage objects, one per page.
    """
    reader = PdfReader(str(path))
    total = len(reader.pages)
    pages: list[DocumentPage] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            DocumentPage(
                text=text.strip(),
                page_number=i + 1,
                source_file=path.name,
                total_pages=total,
            )
        )

    return pages


def load_directory(directory: Path) -> list[DocumentPage]:
    """Load all PDFs from a directory.

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        All pages from all PDFs, sorted by filename then page number.
    """
    all_pages: list[DocumentPage] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        all_pages.extend(load_pdf(pdf_path))
    return all_pages
