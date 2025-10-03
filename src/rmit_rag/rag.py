from __future__ import annotations
from typing import Sequence
import ollama
from .embedder import Embedder
from .vector_store import VectorStore
from .config import settings
from .interfaces import EmbedderProtocol, VectorStoreProtocol
from .personality import get_personality_config


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
        # Get personality configuration
        system_prompt, user_template, temperature = get_personality_config(settings.personality_level)
        
        # Use custom temperature if provided, otherwise use personality default
        final_temperature = settings.temperature if settings.temperature != 0.4 else temperature
        
        prompt = user_template.format(context=context, question=question)
        
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": final_temperature,
                "top_p": 0.9,
                "num_predict": settings.max_response_length,  # Limit response length for faster generation
                "num_ctx": settings.context_window,           # Limit context window for faster processing
                "stop": ["\n\n", "Question:", "Context:"],    # Stop tokens for faster generation
            },
        )
        return response["message"]["content"]
