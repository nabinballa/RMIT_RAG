from __future__ import annotations
import re
from typing import Sequence, Tuple, List


WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _strip_control_chars(text: str) -> str:
    # Remove non-printable control characters except common whitespace
    return "".join(ch for ch in text if ch == "\t" or ch == "\n" or ch == " " or (ord(ch) >= 32 and ord(ch) != 127))


def clean_documents_and_metadatas(
    documents: Sequence[str],
    metadatas: Sequence[dict] | None = None,
    *,
    to_lower: bool = True,
    strip_controls: bool = True,
    normalize_spaces: bool = True,
    min_length: int = 0,
) -> Tuple[List[str], List[dict] | None]:
    """Apply lightweight, safe text cleaning to documents while preserving alignment with metadatas.

    - to_lower: convert to lowercase
    - strip_controls: remove control characters
    - normalize_spaces: collapse multiple whitespace, trim ends
    - min_length: drop documents shorter than this many characters (after cleaning)
    """
    cleaned_docs: List[str] = []
    cleaned_metas: List[dict] | None = [] if metadatas is not None else None

    for idx, doc in enumerate(documents):
        text = doc
        if strip_controls:
            text = _strip_control_chars(text)
        if normalize_spaces:
            text = _normalize_whitespace(text)
        if to_lower:
            text = text.lower()

        if min_length and len(text) < min_length:
            # skip this doc and its metadata
            continue

        cleaned_docs.append(text)
        if cleaned_metas is not None and metadatas is not None:
            cleaned_metas.append(metadatas[idx])

    return cleaned_docs, cleaned_metas


