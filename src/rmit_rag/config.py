from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")
    sheet_name: str = os.getenv("SHEET_NAME", "Sheet1")
    # Chroma backend implementation: 'duckdb' (default) or 'sqlite'.
    # This is read by Chroma itself; we expose it here for visibility.
    chroma_db_impl: str = os.getenv("CHROMA_DB_IMPL", os.getenv("CHROMA_DB", "duckdb"))


settings = Settings()
