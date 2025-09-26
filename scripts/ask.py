#!/usr/bin/env python
from __future__ import annotations
import os
from rmit_rag.rag import RAGPipeline
from rmit_rag.embedder import Embedder
from rmit_rag.vector_store import VectorStore
from rmit_rag.config import settings


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def main() -> None:
    collection = _get_env("COLLECTION", "combined_docs") or "combined_docs"
    k_raw = _get_env("K", "5") or "5"
    try:
        k = int(k_raw)
    except Exception:
        k = 5

    embedder = Embedder("all-MiniLM-L6-v2")
    store = VectorStore(collection, persist_directory=settings.chroma_dir)
    pipeline = RAGPipeline(collection, embedder=embedder, store=store)

    question = _get_env("QUESTION", None)
    if question:
        print(pipeline.query(question, n_results=k))
        return

    # REPL loop
    try:
        while True:
            try:
                user_input = input("Enter question (or 'exit' to quit): ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", ":q", "q"}:
                break
            answer = pipeline.query(user_input, n_results=k)
            print(answer)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
