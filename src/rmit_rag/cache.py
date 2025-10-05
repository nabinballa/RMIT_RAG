"""Response caching for RAG queries to improve performance on repeated questions."""

from __future__ import annotations
import hashlib
import json
from typing import Dict, Optional
from functools import lru_cache
import logging

# Simple in-memory cache with LRU eviction
@lru_cache(maxsize=100)
def _cached_response(query_hash: str) -> Optional[str]:
    """Internal cache function - not used directly."""
    return None

# Cache storage
_response_cache: Dict[str, str] = {}
_cache_stats = {"hits": 0, "misses": 0}

def get_cached_response(question: str) -> Optional[str]:
    """Get cached response for a question if available.
    
    Args:
        question: The user's question
        
    Returns:
        Cached response if available, None otherwise
    """
    # Create a simple hash of the question for caching
    query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    if query_hash in _response_cache:
        _cache_stats["hits"] += 1
        logging.debug(f"Cache hit for query: {question[:50]}...")
        return _response_cache[query_hash]
    
    _cache_stats["misses"] += 1
    return None

def cache_response(question: str, response: str) -> None:
    """Cache a response for future queries.
    
    Args:
        question: The user's question
        response: The generated response
    """
    query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    # Limit cache size (simple LRU-like behavior)
    if len(_response_cache) >= 100:
        # Remove oldest entry (simple approach)
        oldest_key = next(iter(_response_cache))
        del _response_cache[oldest_key]
    
    _response_cache[query_hash] = response
    logging.debug(f"Cached response for query: {question[:50]}...")

def clear_cache() -> None:
    """Clear all cached responses."""
    global _response_cache, _cache_stats
    _response_cache.clear()
    _cache_stats = {"hits": 0, "misses": 0}
    logging.info("Response cache cleared")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics.
    
    Returns:
        Dictionary with cache hit/miss counts and size
    """
    return {
        **_cache_stats,
        "size": len(_response_cache),
        "hit_rate": _cache_stats["hits"] / max(1, _cache_stats["hits"] + _cache_stats["misses"])
    }
