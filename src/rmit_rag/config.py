from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral:7b")
    sheet_name: str = os.getenv("SHEET_NAME", "Sheet1")
    # Personality settings
    personality_level: str = os.getenv("PERSONALITY_LEVEL", "friendly")  # friendly, professional, casual
    temperature: float = float(os.getenv("TEMPERATURE", "0.4"))  # 0.1-1.0, higher = more creative
    
    # Performance settings
    max_response_length: int = int(os.getenv("MAX_RESPONSE_LENGTH", "150"))  # Limit response length for speed
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "256"))  # Limit context window for speed
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))  # Larger batch size for embedding efficiency
    
    # Chroma backend implementation: 'duckdb' (default) or 'sqlite'.
    # This is read by Chroma itself; we expose it here for visibility.
    chroma_db_impl: str = os.getenv("CHROMA_DB_IMPL", os.getenv("CHROMA_DB", "duckdb"))


settings = Settings()
