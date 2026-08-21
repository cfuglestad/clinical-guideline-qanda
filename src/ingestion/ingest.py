"""CLI entry point to ingest PDF guidelines into the vector store.

Usage:
    python -m src.ingestion.ingest [--data-dir data/raw] [--persist-dir chroma_db]
"""

import argparse
import sys
from pathlib import Path

from src.ingestion.chunker import ChunkerConfig, chunk_pages
from src.ingestion.loader import load_pdf
from src.retrieval.store import GuidelineStore


def main(data_dir: Path, persist_dir: str) -> None:
    """Load PDFs, chunk them, and store embeddings."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {data_dir}")

    store = GuidelineStore(persist_dir=persist_dir)
    config = ChunkerConfig()
    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, config)
        store.add_chunks(chunks)
        total_chunks += len(chunks)
        print(f"    {len(pages)} pages → {len(chunks)} chunks")

    print(f"\nDone. {total_chunks} total chunks stored in {persist_dir}/")
    print(f"Store now contains {store.count} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clinical guideline PDFs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing PDF files (default: data/raw)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="ChromaDB persistence directory (default: chroma_db)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.persist_dir)
