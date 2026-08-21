"""ChromaDB vector store for clinical guideline chunks."""

from typing import Protocol

import chromadb
from chromadb.config import Settings

from src.ingestion.chunker import Chunk

DEFAULT_COLLECTION = "clinical_guidelines"
DEFAULT_PERSIST_DIR = "chroma_db"


class Embedder(Protocol):
    """Protocol for embedding text into vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformer embedding model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into normalized embeddings."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [row.tolist() for row in embeddings]


class GuidelineStore:
    """Manages chunk storage and retrieval via ChromaDB."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
    ) -> None:
        self._embedder = embedder or SentenceTransformerEmbedder()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of chunks in the store."""
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks with metadata."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts)

        ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "heading": c.heading,
                "source_file": c.source_file,
                "page_numbers": ",".join(str(p) for p in c.page_numbers),
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def query(self, question: str, n_results: int = 5) -> list[dict[str, object]]:
        """Retrieve the most relevant chunks for a question.

        Args:
            question: The user's clinical question.
            n_results: Number of chunks to retrieve.

        Returns:
            List of dicts with keys: text, heading, source_file, score, page_numbers.
        """
        query_embedding = self._embedder.encode([question])

        results = self._collection.query(
            query_embeddings=query_embedding,  # type: ignore[arg-type]
            n_results=n_results,
            include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
        )

        retrieved: list[dict[str, object]] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            retrieved.append(
                {
                    "text": doc,
                    "heading": meta["heading"],
                    "source_file": meta["source_file"],
                    "page_numbers": meta["page_numbers"],
                    "score": 1.0 - dist,  # cosine distance -> similarity
                }
            )

        return retrieved

    def reset(self) -> None:
        """Delete all chunks from the collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=DEFAULT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
