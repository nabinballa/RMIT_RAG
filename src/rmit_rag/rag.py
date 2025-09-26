from __future__ import annotations
from typing import Sequence
import ollama
from .embedder import Embedder
from .vector_store import VectorStore
from .config import settings


class RAGPipeline:
    def __init__(self, collection_name: str, embed_model: str = "all-MiniLM-L6-v2") -> None:
        self.embedder = Embedder(embed_model)
        self.store = VectorStore(collection_name, persist_directory=settings.chroma_dir)

    def index(self, documents: Sequence[str], metadatas: Sequence[dict] | None = None) -> None:
        embeddings = self.embedder.encode(list(documents))
        ids = [str(i) for i in range(len(documents))]
        self.store.add(documents=list(documents), embeddings=embeddings, ids=ids, metadatas=metadatas)

    def query(self, question: str, n_results: int = 5) -> str:
        query_embedding = self.embedder.encode([question])
        results = self.store.query(query_embeddings=query_embedding, n_results=n_results)
        context = " ".join(results["documents"][0]) if results and results.get("documents") else ""
        prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Follow these guidelines:\n"
            "1. Answer based ONLY on the provided context\n"
            "2. If the answer isn't in the context, say \"I don't have enough information to answer this question based on the provided context.\"\n"
            "3. Be concise and accurate\n"
            "4. When possible, reference specific parts of the context\n"
            "5. If you're uncertain, express that uncertainty\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
        )
        response = ollama.chat(model=settings.ollama_model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
