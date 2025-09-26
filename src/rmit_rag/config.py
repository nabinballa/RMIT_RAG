from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    sheet_name: str = os.getenv("SHEET_NAME", "Sheet1")


settings = Settings()
