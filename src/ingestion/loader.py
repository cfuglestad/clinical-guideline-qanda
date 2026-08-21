"""Load clinical guidelines (PDF or TXT) and extract text with metadata."""

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


def load_txt(path: Path) -> list[DocumentPage]:
    """Load a plain text file as a single page.

    Args:
        path: Path to the text file.

    Returns:
        List containing one DocumentPage with the full text.
    """
    text = path.read_text(encoding="utf-8")
    return [
        DocumentPage(
            text=text.strip(),
            page_number=1,
            source_file=path.name,
            total_pages=1,
        )
    ]


def load_file(path: Path) -> list[DocumentPage]:
    """Load a single file (PDF or TXT) based on extension."""
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    if path.suffix.lower() == ".txt":
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_directory(directory: Path) -> list[DocumentPage]:
    """Load all supported files (PDF, TXT) from a directory.

    Args:
        directory: Path to directory containing guideline files.

    Returns:
        All pages from all files, sorted by filename then page number.
    """
    all_pages: list[DocumentPage] = []
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in (".pdf", ".txt"):
            all_pages.extend(load_file(file_path))
    return all_pages
