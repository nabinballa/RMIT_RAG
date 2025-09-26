#!/usr/bin/env python
from __future__ import annotations
import click
from rmit_rag.rag import RAGPipeline


@click.command()
@click.argument("question", type=str)
@click.option("--collection", type=str, default="combined_docs")
@click.option("--k", type=int, default=5, help="Number of results to retrieve")
def main(question: str, collection: str, k: int) -> None:
    pipeline = RAGPipeline(collection)
    answer = pipeline.query(question, n_results=k)
    print(answer)


if __name__ == "__main__":
    main()
