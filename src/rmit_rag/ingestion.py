from __future__ import annotations
from typing import Sequence


def generate_sequential_ids(num_items: int) -> list[str]:
    """Generate simple string IDs ("0", "1", ...).

    Keeps a stable order mapping between documents and embeddings.
    """
    return [str(i) for i in range(num_items)]


def ingest_documents(
    *,
    embedder,
    store,
    documents: Sequence[str],
    metadatas: Sequence[dict] | None = None,
) -> None:
    """Embed `documents` and write them with optional `metadatas` to the vector store.

    Assumes any preprocessing has already been applied to `documents`.
    """
    texts = list(documents)
    embeddings = embedder.encode(texts)
    ids = generate_sequential_ids(len(texts))
    store.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)


