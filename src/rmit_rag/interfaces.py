from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    def clear(self) -> None:
        ...

    def add(
        self,
        *,
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        ...

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
    ) -> dict:
        ...


