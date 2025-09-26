from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import logging

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> None:
        """
        Initialize the Embedder with a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model. Defaults to 'all-MiniLM-L6-v2'.
            batch_size (int): Batch size for encoding. Defaults to 32.
        Raises:
            ValueError: If model_name is invalid or unsupported.
        """
        try:
            # Prefer Apple Silicon (MPS) or CUDA when available for faster encoding
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            self.model = SentenceTransformer(model_name, device=device)
            self.batch_size = batch_size
            logging.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {str(e)}")
            raise ValueError(f"Invalid model name: {model_name}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts (List[str]): List of texts to encode.
        Returns:
            List[List[float]]: List of embeddings for each text.
        Raises:
            ValueError: If texts is empty or contains invalid entries.
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        # Filter out invalid entries (e.g., None, empty strings)
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided for encoding")
        
        try:
            # Encode with batching for efficiency
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False
            ).tolist()
            logging.info(f"Encoded {len(valid_texts)} texts successfully")
            return embeddings
        except Exception as e:
            logging.error(f"Encoding failed: {str(e)}")
            raise
