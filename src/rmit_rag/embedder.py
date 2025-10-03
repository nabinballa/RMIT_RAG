from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import logging
import functools
import threading

# Global model cache to avoid reloading
_model_cache = {}
_model_lock = threading.Lock()

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = None) -> None:
        """
        Initialize the Embedder with a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model. Defaults to 'all-MiniLM-L6-v2'.
            batch_size (int): Batch size for encoding. Defaults to 32.
        Raises:
            ValueError: If model_name is invalid or unsupported.
        """
        from .config import settings
        self.model_name = model_name
        self.batch_size = batch_size or settings.batch_size
        
        # Use cached model if available
        with _model_lock:
            if model_name not in _model_cache:
                try:
                    # Prefer Apple Silicon (MPS) or CUDA when available for faster encoding
                    device = "cpu"
                    if torch.backends.mps.is_available():
                        device = "mps"
                    elif torch.cuda.is_available():
                        device = "cuda"
                    
                    # Load with optimizations
                    _model_cache[model_name] = SentenceTransformer(
                        model_name, 
                        device=device,
                        cache_folder=None,  # Disable model caching to save disk I/O
                        trust_remote_code=False
                    )
                    
                    # Warm up the model with a dummy encoding
                    _model_cache[model_name].encode(["warmup"], convert_to_tensor=False)
                    
                    logging.info(f"Loaded and cached SentenceTransformer model: {model_name} on {device}")
                except Exception as e:
                    logging.error(f"Failed to load model {model_name}: {str(e)}")
                    raise ValueError(f"Invalid model name: {model_name}")
            
            self.model = _model_cache[model_name]

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
            # Encode with optimized settings for speed
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=False,  # Skip normalization for speed
                device=None  # Use model's device
            ).tolist()
            
            # Only log for larger batches to reduce logging overhead
            if len(valid_texts) > 10:
                logging.debug(f"Encoded {len(valid_texts)} texts successfully")
            return embeddings
        except Exception as e:
            logging.error(f"Encoding failed: {str(e)}")
            raise
