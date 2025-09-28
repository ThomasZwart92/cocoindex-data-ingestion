"""Document processors"""

# Avoid importing heavy/optional dependencies at package import time
try:
    from .parser import DocumentParser  # may require optional llama_parse
except Exception:  # pragma: no cover
    DocumentParser = None  # type: ignore

from .chunker import DocumentChunker
from .embedder import EmbeddingGenerator

__all__ = [
    "DocumentParser",
    "DocumentChunker",
    "EmbeddingGenerator"
]
