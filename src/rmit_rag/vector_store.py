from __future__ import annotations
from typing import Sequence
from pathlib import Path
import chromadb


class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str | Path = "chroma") -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(collection_name)

    @property
    def collection(self):
        return self._collection

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(self._collection.name)

    def add(self, *, documents: Sequence[str], embeddings: Sequence[Sequence[float]], ids: Sequence[str], metadatas: Sequence[dict] | None = None) -> None:
        self._collection.add(documents=list(documents), embeddings=list(embeddings), ids=list(ids), metadatas=list(metadatas) if metadatas is not None else None)

    def query(self, *, query_embeddings: Sequence[Sequence[float]], n_results: int = 5):
        return self._collection.query(query_embeddings=list(query_embeddings), n_results=n_results)
