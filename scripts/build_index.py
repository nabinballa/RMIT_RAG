#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import click
from rmit_rag.data_loader import load_sources, dataframe_to_documents
from rmit_rag.rag import RAGPipeline
from rmit_rag.config import settings


@click.command()
@click.option("--data-dir", type=click.Path(path_type=Path), default=Path("./data"))
@click.option("--collection", type=str, default="combined_docs")
@click.option("--clear", is_flag=True, help="Clear existing collection before indexing")
def main(data_dir: Path, collection: str, clear: bool) -> None:
    df = load_sources(data_dir)
    docs = dataframe_to_documents(df)
    metadatas = [{"source": src} for src in df["source"].astype(str).tolist()]

    pipeline = RAGPipeline(collection)
    if clear:
        pipeline.store.clear()

    pipeline.index(docs, metadatas=metadatas)
    print(json.dumps({"status": "ok", "collection": collection, "count": len(docs), "persist": settings.chroma_dir}))


if __name__ == "__main__":
    main()
