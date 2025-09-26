from __future__ import annotations
from typing import Sequence
import ollama
from .embedder import Embedder
from .vector_store import VectorStore
from .config import settings
from .interfaces import EmbedderProtocol, VectorStoreProtocol


class RAGPipeline:
    def __init__(
        self,
        collection_name: str,
        embed_model: str = "all-MiniLM-L6-v2",
        *,
        embedder: EmbedderProtocol | None = None,
        store: VectorStoreProtocol | None = None,
    ) -> None:
        """Construct a RAG pipeline with injectable components.

        If `embedder`/`store` are omitted, sensible defaults are created
        using the configured embed model and Chroma persist directory.
        """
        self.embedder: EmbedderProtocol = embedder or Embedder(embed_model)
        self.store: VectorStoreProtocol = store or VectorStore(
            collection_name, persist_directory=settings.chroma_dir
        )

    def index(self, documents: Sequence[str], metadatas: Sequence[dict] | None = None) -> None:
        """Embed `documents` and write them to the vector store.

        Prefer `rmit_rag.ingestion.ingest_documents` for single-shot ingestion;
        this method is a thin wrapper for convenience/testing.
        """
        embeddings = self.embedder.encode(list(documents))
        ids = [str(i) for i in range(len(documents))]
        self.store.add(
            documents=list(documents),
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def query(self, question: str, n_results: int = 5) -> str:
        """Retrieve top-matching documents and ask the chat model to answer.

        Builds a strict prompt to constrain answers to retrieved context.
        """
        query_embedding = self.embedder.encode([question])
        results = self.store.query(query_embeddings=query_embedding, n_results=n_results)
        context = " ".join(results["documents"][0]) if results and results.get("documents") else ""
        prompt = (
            "Answer the question using ONLY the provided context. "
            "Return a single concise paragraph (no bullet points or numbered lists). "
            "If the context lacks sufficient information, answer with exactly: "
            "'I don't have enough information to answer this question based on the provided context.'\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
        )
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": "You respond with a single concise paragraph. No lists or headings."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )
        return response["message"]["content"]
