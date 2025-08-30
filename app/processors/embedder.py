"""Embedding generator placeholder"""
import logging
from typing import List
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for chunks (placeholder)"""
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[str]:
        """
        Generate embeddings for chunks
        Returns list of embedding IDs (placeholder implementation)
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks (placeholder)")
        
        # Return placeholder embedding IDs
        return [f"emb_{chunk.id}" for chunk in chunks]