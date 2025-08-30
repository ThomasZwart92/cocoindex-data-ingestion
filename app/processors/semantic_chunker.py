"""
True Semantic Chunking Implementation
Based on sentence similarity and topic boundaries
"""
import re
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Implements true semantic chunking using embedding similarity
    to find natural topic boundaries in text.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 50,  # Min tokens
        max_chunk_size: int = 200,  # Max tokens  
        min_sentences: int = 1,
        max_sentences: int = 10
    ):
        """
        Initialize semantic chunker with embedding model.
        
        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Minimum similarity to keep sentences together (0-1)
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            min_sentences: Minimum sentences per chunk
            max_sentences: Maximum sentences per chunk
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b([A-Z])\.\s*', r'\1<DOT> ', text)
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots and clean
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        return [s for s in sentences if s]
    
    def calculate_semantic_similarity(self, sentences: List[str]) -> np.ndarray:
        """
        Calculate pairwise semantic similarity between sentences.
        
        Returns:
            Similarity matrix where element [i,j] is similarity between sentence i and j
        """
        if not sentences:
            return np.array([])
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def find_semantic_boundaries(
        self, 
        sentences: List[str],
        similarity_matrix: np.ndarray
    ) -> List[int]:
        """
        Find semantic boundaries based on similarity drops.
        
        Returns:
            List of sentence indices where semantic breaks occur
        """
        if len(sentences) <= 1:
            return []
        
        boundaries = []
        
        # Calculate similarity between consecutive sentences
        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i, i + 1]
            
            # If similarity drops below threshold, mark as boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)
        
        return boundaries
    
    def merge_short_chunks(
        self,
        chunks: List[List[str]],
        similarity_matrix: np.ndarray,
        sentence_indices: List[Tuple[int, int]]
    ) -> List[List[str]]:
        """
        Merge chunks that are too short with their most similar neighbor.
        """
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_size = sum(len(s.split()) for s in current_chunk)
            
            # If chunk is too small and not the last chunk
            if current_size < self.min_chunk_size and i < len(chunks) - 1:
                # Check similarity with next chunk
                curr_start, curr_end = sentence_indices[i]
                next_start, next_end = sentence_indices[i + 1]
                
                # Calculate average similarity between chunks
                avg_similarity = np.mean(
                    similarity_matrix[curr_start:curr_end, next_start:next_end]
                )
                
                # If similar enough, merge
                if avg_similarity >= self.similarity_threshold * 0.8:  # Slightly lower threshold for merging
                    merged_chunk = current_chunk + chunks[i + 1]
                    merged.append(merged_chunk)
                    i += 2  # Skip next chunk since we merged it
                    continue
            
            merged.append(current_chunk)
            i += 1
        
        return merged
    
    def create_semantic_chunks(
        self,
        text: str,
        maintain_context: bool = True
    ) -> List[Tuple[str, dict]]:
        """
        Create truly semantic chunks based on topic coherence.
        
        Args:
            text: Text to chunk
            maintain_context: Whether to add context from surrounding chunks
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Calculate semantic similarity
        similarity_matrix = self.calculate_semantic_similarity(sentences)
        
        # Find semantic boundaries
        boundaries = self.find_semantic_boundaries(sentences, similarity_matrix)
        
        # Create initial chunks based on boundaries
        chunks = []
        sentence_indices = []
        start_idx = 0
        
        for boundary_idx in boundaries:
            chunk = sentences[start_idx:boundary_idx]
            chunks.append(chunk)
            sentence_indices.append((start_idx, boundary_idx))
            start_idx = boundary_idx
        
        # Don't forget the last chunk
        if start_idx < len(sentences):
            chunks.append(sentences[start_idx:])
            sentence_indices.append((start_idx, len(sentences)))
        
        # Merge chunks that are too short
        chunks = self.merge_short_chunks(chunks, similarity_matrix, sentence_indices)
        
        # Split chunks that are too long
        final_chunks = []
        for chunk in chunks:
            chunk_text = ' '.join(chunk)
            token_count = len(chunk_text.split())
            
            if token_count > self.max_chunk_size:
                # Split large chunk at natural boundaries
                sub_chunks = self._split_large_chunk(chunk, self.max_chunk_size)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # Create output with metadata
        result = []
        for i, chunk_sentences in enumerate(final_chunks):
            chunk_text = ' '.join(chunk_sentences)
            
            # Calculate semantic focus (main topic)
            if len(chunk_sentences) > 1:
                # Find the most representative sentence (highest avg similarity to others)
                chunk_embeddings = self.model.encode(chunk_sentences)
                chunk_similarities = cosine_similarity(chunk_embeddings)
                avg_similarities = np.mean(chunk_similarities, axis=1)
                most_representative_idx = np.argmax(avg_similarities)
                semantic_focus = chunk_sentences[most_representative_idx][:100]  # First 100 chars
            else:
                semantic_focus = chunk_sentences[0][:100] if chunk_sentences else ""
            
            metadata = {
                'sentence_count': len(chunk_sentences),
                'token_count': len(chunk_text.split()),
                'semantic_focus': semantic_focus,
                'chunk_index': i,
                'total_chunks': len(final_chunks)
            }
            
            # Add context if requested
            if maintain_context and len(final_chunks) > 1:
                if i > 0:
                    prev_chunk = ' '.join(final_chunks[i-1])
                    metadata['previous_context'] = prev_chunk[-200:]  # Last 200 chars
                if i < len(final_chunks) - 1:
                    next_chunk = ' '.join(final_chunks[i+1])
                    metadata['next_context'] = next_chunk[:200]  # First 200 chars
            
            result.append((chunk_text, metadata))
        
        return result
    
    def _split_large_chunk(
        self,
        sentences: List[str],
        max_size: int
    ) -> List[List[str]]:
        """Split a large chunk into smaller ones at natural boundaries."""
        sub_chunks = []
        current = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > max_size and current:
                sub_chunks.append(current)
                current = [sentence]
                current_size = sentence_size
            else:
                current.append(sentence)
                current_size += sentence_size
        
        if current:
            sub_chunks.append(current)
        
        return sub_chunks
    
    def identify_key_concepts(self, text: str, top_k: int = 5) -> List[str]:
        """
        Identify key concepts/topics in the text using embeddings.
        
        Args:
            text: Text to analyze
            top_k: Number of key concepts to return
            
        Returns:
            List of key concept phrases
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        # Find sentences that are most representative (high avg similarity)
        similarity_matrix = cosine_similarity(embeddings)
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # Get top-k most representative sentences
        top_indices = np.argsort(avg_similarities)[-top_k:][::-1]
        
        # Extract key phrases from these sentences
        key_concepts = []
        for idx in top_indices:
            sentence = sentences[idx]
            # Extract noun phrases or important segments
            # For now, just take the core part of the sentence
            concept = self._extract_key_phrase(sentence)
            if concept:
                key_concepts.append(concept)
        
        return key_concepts[:top_k]
    
    def _extract_key_phrase(self, sentence: str) -> str:
        """Extract the key phrase from a sentence."""
        # Simple approach: take the main clause
        # Remove common starting words
        sentence = re.sub(r'^(The|This|These|Those|It|They|We|You|I)\s+', '', sentence)
        
        # Take first 50 characters as key phrase
        key_phrase = sentence[:50]
        
        # Clean up
        if ',' in key_phrase:
            key_phrase = key_phrase.split(',')[0]
        
        return key_phrase.strip()


# Example usage
if __name__ == "__main__":
    text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    These systems improve their performance over time without being explicitly programmed.
    Neural networks are inspired by the human brain structure.
    They consist of interconnected nodes that process information.
    Deep learning uses multiple layers of neural networks.
    This approach has revolutionized computer vision and natural language processing.
    
    Climate change is affecting global weather patterns.
    Rising temperatures are causing ice caps to melt.
    Sea levels are increasing as a result.
    Many coastal cities face flooding risks.
    
    Python is a popular programming language.
    It has simple syntax and powerful libraries.
    Data scientists often use Python for analysis.
    The ecosystem includes tools like NumPy and Pandas.
    """
    
    chunker = SemanticChunker(similarity_threshold=0.5)
    chunks = chunker.create_semantic_chunks(text)
    
    print(f"Created {len(chunks)} semantic chunks:\n")
    for i, (chunk_text, metadata) in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Text: {chunk_text[:100]}...")
        print(f"  Sentences: {metadata['sentence_count']}")
        print(f"  Tokens: {metadata['token_count']}")
        print(f"  Focus: {metadata['semantic_focus']}")
        print()