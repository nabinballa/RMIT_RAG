#!/usr/bin/env python
from __future__ import annotations
import json
import os
from pathlib import Path
from rmit_rag.data_loader import (
    load_qa_csv,
    qa_dataframe_to_documents,
)
from rmit_rag.rag import RAGPipeline
from rmit_rag.ingestion import ingest_documents
from rmit_rag.config import settings
from rmit_rag.preprocess import clean_documents_and_metadatas


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def parse_qa_specs(raw: str | None) -> list[tuple[Path, str]]:
    if not raw:
        return []
    specs: list[tuple[Path, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            path_str, label = part.split(":", 1)
        else:
            path_str, label = part, "qa"
        specs.append((Path(path_str), label))
    return specs


def main() -> None:
    collection = _get_env("COLLECTION", "combined_docs") or "combined_docs"
    qa_raw = _get_env("QA", None)  # comma-separated: path[:label],path2[:label2]
    qa_mode = _get_env("QA_MODE", "concat") or "concat"
    clear_flag = _get_env("CLEAR", "0") or "0"
    clear = str(clear_flag).lower() in {"1", "true", "yes", "on"}
    data_dir_raw = _get_env("DATA_DIR", "./data") or "./data"
    data_dir = Path(data_dir_raw)

    qa_specs = parse_qa_specs(qa_raw)
    if not qa_specs:
        # Auto-discover all CSVs in DATA_DIR; label by filename stem
        if not data_dir.exists() or not data_dir.is_dir():
            raise SystemExit(f"ERROR: DATA_DIR not found: {data_dir}")
        discovered = sorted(data_dir.glob("*.csv"))
        if not discovered:
            raise SystemExit("ERROR: No QA provided and no CSVs found in DATA_DIR.")
        qa_specs = [(p, p.stem) for p in discovered]

    combined_docs: list[str] = []
    combined_metas: list[dict] = []

    for path, label in qa_specs:
        qa_df = load_qa_csv(path, source_label=label)
        qa_docs, qa_metas = qa_dataframe_to_documents(qa_df, mode=qa_mode)
        combined_docs.extend(qa_docs)
        combined_metas.extend(qa_metas)

    pipeline = RAGPipeline(collection)
    if clear:
        pipeline.store.clear()

    # Optional preprocessing
    enable_pre = _get_env("PREPROCESS", "0") or "0"
    if str(enable_pre).lower() in {"1", "true", "yes", "on"}:
        to_lower = ( _get_env("PRE_TO_LOWER", "1") or "1" ).lower() in {"1","true","yes","on"}
        strip_controls = ( _get_env("PRE_STRIP_CONTROLS", "1") or "1" ).lower() in {"1","true","yes","on"}
        normalize_spaces = ( _get_env("PRE_NORMALIZE_SPACES", "1") or "1" ).lower() in {"1","true","yes","on"}
        min_length_raw = _get_env("PRE_MIN_LENGTH", "0") or "0"
        try:
            min_length = int(min_length_raw)
        except Exception:
            min_length = 0
        combined_docs, combined_metas = clean_documents_and_metadatas(
            combined_docs,
            combined_metas,
            to_lower=to_lower,
            strip_controls=strip_controls,
            normalize_spaces=normalize_spaces,
            min_length=min_length,
        )

    # Use the ingestion function for clearer separation of concerns
    ingest_documents(embedder=pipeline.embedder, store=pipeline.store, documents=combined_docs, metadatas=combined_metas)
    print(json.dumps({"status": "ok", "collection": collection, "count": len(combined_docs), "persist": settings.chroma_dir}))


if __name__ == "__main__":
    main()
