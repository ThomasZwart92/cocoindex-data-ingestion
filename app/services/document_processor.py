"""
Document Processing Service
Handles document chunking and preprocessing
"""
from typing import List, Dict, Any, Optional
import re
import hashlib


class DocumentProcessor:
    """Process documents for ingestion"""
    
    async def chunk_document(
        self,
        content: str,
        method: str = "recursive",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        language: str = "markdown",
        min_chunk_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Chunk document into smaller pieces
        
        Args:
            content: Document content to chunk
            method: Chunking method (recursive, sentence, fixed)
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between chunks
            language: Document language/format
            min_chunk_size: Minimum chunk size
            
        Returns:
            List of chunks with text and metadata
        """
        if method == "recursive":
            return self._recursive_chunk(
                content, 
                chunk_size, 
                chunk_overlap,
                min_chunk_size
            )
        elif method == "sentence":
            return self._sentence_chunk(
                content,
                chunk_size,
                chunk_overlap
            )
        else:
            return self._fixed_chunk(
                content,
                chunk_size,
                chunk_overlap
            )
    
    def _recursive_chunk(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int
    ) -> List[Dict[str, Any]]:
        """Recursively split content by different separators"""
        
        # Hierarchy of separators for markdown
        separators = [
            "\n## ",  # H2 headers
            "\n### ",  # H3 headers
            "\n#### ",  # H4 headers
            "\n\n",  # Double newlines (paragraphs)
            "\n",  # Single newlines
            ". ",  # Sentences
            " ",  # Words
            ""  # Characters
        ]
        
        chunks = []
        current_position = 0
        
        while current_position < len(content):
            # Calculate chunk end position
            chunk_end = min(current_position + chunk_size, len(content))
            
            # Extract chunk text
            chunk_text = content[current_position:chunk_end]
            
            # Try to find a good break point
            if chunk_end < len(content):
                # Look for separators near the end
                best_break = chunk_end
                for separator in separators:
                    if separator in chunk_text[int(chunk_size * 0.8):]:
                        last_sep = chunk_text.rfind(separator)
                        if last_sep > 0:
                            best_break = current_position + last_sep + len(separator)
                            chunk_text = content[current_position:best_break]
                            break
            
            # Only add chunk if it meets minimum size
            if len(chunk_text.strip()) >= min_chunk_size:
                chunks.append({
                    "text": chunk_text.strip(),
                    "start": current_position,
                    "end": current_position + len(chunk_text),
                    "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                })
            
            # Move position with overlap
            if chunk_end < len(content):
                current_position = current_position + len(chunk_text) - chunk_overlap
            else:
                break
        
        return chunks
    
    def _sentence_chunk(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """Split by sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start": len(" ".join(chunks)) if chunks else 0,
                    "end": len(" ".join(chunks)) + len(chunk_text) if chunks else len(chunk_text),
                    "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                })
                
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        if overlap_size >= chunk_overlap:
                            break
                        overlap_sentences.insert(0, sent)
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start": len(" ".join(c["text"] for c in chunks[:-1])) if chunks else 0,
                "end": len(content),
                "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            })
        
        return chunks
    
    def _fixed_chunk(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """Simple fixed-size chunking"""
        chunks = []
        current_position = 0
        
        while current_position < len(content):
            chunk_end = min(current_position + chunk_size, len(content))
            chunk_text = content[current_position:chunk_end]
            
            chunks.append({
                "text": chunk_text,
                "start": current_position,
                "end": chunk_end,
                "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            })
            
            # Move with overlap
            current_position += chunk_size - chunk_overlap
            
            # Prevent infinite loop
            if current_position <= chunks[-1]["start"]:
                current_position = chunks[-1]["end"]
        
        return chunks
    
    async def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\"\'\/]', '', text)
        
        return text.strip()
    
    async def extract_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract sections from markdown content"""
        sections = []
        
        # Find all headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section:
                    sections.append({
                        "title": current_section["title"],
                        "level": current_section["level"],
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {"title": title, "level": level}
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_section:
            sections.append({
                "title": current_section["title"],
                "level": current_section["level"],
                "content": '\n'.join(current_content).strip()
            })
        
        return sections