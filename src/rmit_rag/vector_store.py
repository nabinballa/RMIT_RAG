from __future__ import annotations
from typing import Sequence
from pathlib import Path
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str | Path = "chroma") -> None:
        """Persistent Chroma collection wrapper.

        Creates/loads a named collection stored under `persist_directory`.
        """
        # Ensure the directory exists before initializing the client
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        # Disable telemetry to avoid network timeouts/errors during operations
        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(collection_name)

    @property
    def collection(self):
        """Access the underlying Chroma collection."""
        return self._collection

    def clear(self) -> None:
        """Delete and recreate the collection, removing all entries."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(self._collection.name)

    def add(self, *, documents: Sequence[str], embeddings: Sequence[Sequence[float]], ids: Sequence[str], metadatas: Sequence[dict] | None = None) -> None:
        """Add documents with precomputed embeddings and optional metadatas."""
        self._collection.add(documents=list(documents), embeddings=list(embeddings), ids=list(ids), metadatas=list(metadatas) if metadatas is not None else None)

    def query(self, *, query_embeddings: Sequence[Sequence[float]], n_results: int = 5):
        """Retrieve top matches for the given query embeddings."""
        return self._collection.query(query_embeddings=list(query_embeddings), n_results=n_results)
