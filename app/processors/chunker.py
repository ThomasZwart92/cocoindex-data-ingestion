"""Document chunking with multiple strategies"""
import logging
from typing import List, Optional
from app.models.chunk import ChunkingStrategy

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Chunk documents using various strategies"""
    
    def chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[dict]:
        """
        Chunk text using specified strategy with metadata
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy to use
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Get raw chunks using existing method
        raw_chunks = self.chunk(text, strategy, chunk_size, chunk_overlap)
        
        # Add metadata to each chunk
        chunks_with_metadata = []
        current_position = 0
        
        for i, chunk_text in enumerate(raw_chunks):
            # Find the actual position of this chunk in the original text
            chunk_start = text.find(chunk_text, current_position)
            if chunk_start == -1:
                chunk_start = current_position
            
            chunk_end = chunk_start + len(chunk_text)
            
            chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_index": i,
                    "chunk_size": len(chunk_text),
                    "start_index": chunk_start,
                    "end_index": chunk_end,
                    "strategy": strategy.value
                }
            })
            
            current_position = max(current_position, chunk_start + 1)
        
        return chunks_with_metadata
    
    def chunk(
        self,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        Chunk text using specified strategy
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy to use
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        logger.info(f"Chunking text with strategy: {strategy.value}")
        
        if strategy == ChunkingStrategy.FIXED:
            return self._fixed_chunking(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive_chunking(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, chunk_size)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _fixed_chunking(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Simple fixed-size chunking"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start = end - chunk_overlap if end < text_length else end
        
        return chunks
    
    def _sentence_chunking(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Chunk by sentences"""
        # Simple sentence splitting
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _recursive_chunking(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Recursively chunk text by different separators
        Priority: paragraphs -> sentences -> words
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_size = len(paragraph)
            
            if para_size > chunk_size:
                # Paragraph too large, split by sentences
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph
                sentence_chunks = self._sentence_chunking(
                    paragraph, 
                    chunk_size, 
                    chunk_overlap
                )
                chunks.extend(sentence_chunks)
                
            elif current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_chunk) > 0:
                    # Keep last paragraph for overlap
                    overlap_text = current_chunk[-1]
                    if len(overlap_text) > chunk_overlap:
                        # Trim overlap to specified size
                        overlap_text = overlap_text[-chunk_overlap:]
                    current_chunk = [overlap_text, paragraph]
                    current_size = len(overlap_text) + para_size
                else:
                    current_chunk = [paragraph]
                    current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size + 2  # +2 for \n\n
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _semantic_chunking(
        self,
        text: str,
        chunk_size: int
    ) -> List[str]:
        """
        Semantic chunking based on topic boundaries
        For now, fallback to recursive chunking
        TODO: Implement actual semantic chunking with embeddings
        """
        logger.info("Semantic chunking not fully implemented, using recursive strategy")
        return self._recursive_chunking(text, chunk_size, chunk_overlap=50)