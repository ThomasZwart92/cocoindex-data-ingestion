"""
Three-tier hierarchical chunking processor for maximum retrieval accuracy.
Implements Anthropic's Contextual Retrieval with additional semantic granularity.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import tiktoken
from datetime import datetime
import uuid
import hashlib

from app.services.llm_service import LLMService
from app.services.supabase_service import SupabaseService


@dataclass
class ChunkData:
    """Data structure for a chunk at any level."""
    document_id: str
    id: str  # Changed from chunk_id to id to match DB schema
    chunk_level: str  # 'page', 'paragraph', 'semantic' 
    chunk_index: int
    chunk_text: str  # Changed from content to chunk_text to match DB
    chunk_size: int  # Changed from token_count to chunk_size to match DB
    chunking_strategy: str = 'semantic'  # Required field in DB
    contextual_summary: str = ''
    contextualized_text: str = ''
    parent_chunk_id: Optional[str] = None
    bm25_tokens: List[str] = None
    sentence_count: Optional[int] = None
    semantic_focus: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ThreeTierChunker:
    """
    Implements three-tier hierarchical chunking:
    1. Page level (1200 tokens) - broad document context
    2. Section level (200-400 tokens) - balanced chunks
    3. Semantic level (20-100 tokens) - precise single-concept chunks
    """
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.llm_service = LLMService()
        self.supabase = SupabaseService()
        
        # Chunking parameters
        self.page_token_size = 1200
        self.page_overlap = 200
        self.section_token_size = 300
        self.section_overlap = 50
        self.semantic_max_sentences = 3
        self.semantic_max_tokens = 100
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 search."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stopwords (simplified list)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Ltd|Corp|Co)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(etc|vs|i\.e|e\.g)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b([A-Z])\.\s*', r'\1<DOT> ', text)  # Single letter abbreviations
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def create_semantic_chunks(self, text: str, parent_id: str, section_idx: int) -> List[Tuple[str, int]]:
        """
        Create TRUE semantic chunks based on topic coherence.
        Returns list of (chunk_text, sentence_count) tuples.
        """
        try:
            # Try to use semantic chunking if available
            from app.processors.semantic_chunker import SemanticChunker
            
            semantic_chunker = SemanticChunker(
                similarity_threshold=0.5,
                min_chunk_size=20,
                max_chunk_size=self.semantic_max_tokens,
                min_sentences=1,
                max_sentences=self.semantic_max_sentences
            )
            
            # Get semantic chunks with metadata
            semantic_results = semantic_chunker.create_semantic_chunks(text, maintain_context=True)
            
            # Convert to expected format
            chunks = []
            for chunk_text, metadata in semantic_results:
                chunks.append((chunk_text, metadata['sentence_count']))
            
            return chunks
            
        except ImportError:
            # Fallback to sentence-based chunking if semantic chunker not available
            print("Warning: SemanticChunker not available, falling back to sentence-based chunking")
            sentences = self.split_into_sentences(text)
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                # Check if adding this sentence exceeds limits
                if current_chunk and (
                    len(current_chunk) >= self.semantic_max_sentences or 
                    current_tokens + sentence_tokens > self.semantic_max_tokens
                ):
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append((chunk_text, len(current_chunk)))
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, len(current_chunk)))
            
            return chunks
    
    def create_section_chunks(self, text: str) -> List[str]:
        """Create section-level chunks respecting markdown structure."""
        # First, try to split by headers
        header_pattern = r'^(#{1,6}\s+.*)$'
        parts = re.split(f'({header_pattern})', text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_header = ""
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            # Check if this is a header
            if re.match(header_pattern, part, re.MULTILINE):
                # Save previous chunk if exists
                if current_chunk and current_tokens > 50:  # Min 50 tokens
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                current_header = part
                continue
            
            # Add header to chunk if we have one
            if current_header and not current_chunk:
                current_chunk = current_header + "\n\n"
                current_tokens = self.count_tokens(current_chunk)
                current_header = ""
            
            part_tokens = self.count_tokens(part)
            
            # If this part alone is too large, split it
            if part_tokens > self.section_token_size * 1.5:
                # Save current chunk first
                if current_chunk and current_tokens > 50:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large part by paragraphs
                paragraphs = part.split('\n\n')
                for para in paragraphs:
                    para_tokens = self.count_tokens(para)
                    if current_tokens + para_tokens > self.section_token_size:
                        if current_chunk and current_tokens > 50:
                            chunks.append(current_chunk)
                        current_chunk = para
                        current_tokens = para_tokens
                    else:
                        if current_chunk:
                            current_chunk += '\n\n' + para
                        else:
                            current_chunk = para
                        current_tokens += para_tokens
            
            # Normal part handling
            elif current_tokens + part_tokens > self.section_token_size:
                if current_chunk and current_tokens > 50:
                    chunks.append(current_chunk)
                current_chunk = part
                current_tokens = part_tokens
            else:
                if current_chunk:
                    current_chunk += '\n\n' + part
                else:
                    current_chunk = part
                current_tokens += part_tokens
        
        # Don't forget the last chunk
        if current_chunk and current_tokens > 30:  # Min 30 tokens for last chunk
            chunks.append(current_chunk)
        
        return chunks
    
    def create_page_chunks(self, text: str) -> List[str]:
        """Create page-level chunks for broad context."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_tokens + para_tokens > self.page_token_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                current_tokens = para_tokens
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def generate_contextual_summary(
        self, 
        chunk_text: str, 
        parent_context: str, 
        doc_title: str,
        chunk_level: str
    ) -> str:
        """Generate AI contextual summary for a chunk."""
        print(f"Generating contextual summary for {chunk_level} chunk...")
        
        # Adjust prompt based on chunk level
        if chunk_level == 'semantic':
            prompt = f"""Document: {doc_title}

Context: {parent_context[:200]}

Sentence(s): {chunk_text}

Write a single sentence that explains the specific fact or concept in this text. Be precise and factual."""
        
        elif chunk_level == 'paragraph':
            prompt = f"""Document: {doc_title}

Document Context: {parent_context[:300]}

Section Content: {chunk_text[:500]}

Write 1-2 sentences summarizing the main points covered in this section and how they relate to the document."""
        
        else:  # page level
            prompt = f"""Document: {doc_title}

Page Content Summary: {chunk_text[:600]}

Write 2-3 sentences summarizing the key topics and themes covered in this page of the document."""
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=150 if chunk_level == 'page' else 100,
            temperature=0.3
        )
        summary = response.content if hasattr(response, 'content') else str(response)
        print(f"Generated summary for {chunk_level}: {summary[:50]}...")
        
        return summary.strip()
    
    async def identify_semantic_focus(self, chunk_text: str) -> str:
        """Identify the key semantic focus of a chunk."""
        prompt = f"""Text: {chunk_text}

In 2-5 words, identify the main topic or concept discussed in this text. Examples:
- "carbon emission reduction"
- "customer satisfaction metrics"
- "regulatory compliance requirements"

Topic:"""
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=20,
            temperature=0.2
        )
        focus = response.content if hasattr(response, 'content') else str(response)
        
        return focus.strip().strip('"').strip("'")
    
    def generate_chunk_id(self, document_id: str, level: str, index: int, parent_id: Optional[str] = None) -> str:
        """Generate a unique chunk ID as UUID."""
        # Generate a deterministic UUID based on content
        if parent_id:
            base = f"{parent_id}_{level}_{index}"
        else:
            base = f"{document_id}_{level}_{index}"
        
        # Create a UUID from the hash
        hash_obj = hashlib.md5(base.encode())
        # Convert first 16 bytes to UUID format
        return str(uuid.UUID(bytes=hash_obj.digest()))
    
    async def process_document(
        self, 
        document_id: str,
        content: str, 
        title: str = "Document",
        metadata: Dict[str, Any] = None
    ) -> List[ChunkData]:
        """
        Process document through three-tier chunking hierarchy.
        Returns all chunks as a flat list with parent-child relationships.
        """
        all_chunks = []
        
        print(f"Processing document '{title}' with three-tier chunking...")
        
        # Level 1: Create page chunks
        page_chunks_text = self.create_page_chunks(content)
        print(f"Created {len(page_chunks_text)} page-level chunks")
        
        for page_idx, page_text in enumerate(page_chunks_text):
            page_id = self.generate_chunk_id(document_id, 'page', page_idx)
            
            # Generate contextual summary for page
            page_summary = await self.generate_contextual_summary(
                page_text,
                title,
                title,
                'page'
            )
            
            # Create contextualized text
            page_contextualized = f"{page_summary}\n\n{page_text}"
            
            # Create page chunk
            page_chunk = ChunkData(
                document_id=document_id,
                id=page_id,
                chunk_level='page',
                chunk_index=page_idx,
                chunk_text=page_text,
                chunk_size=self.count_tokens(page_text),
                contextual_summary=page_summary,
                contextualized_text=page_contextualized,
                bm25_tokens=self.tokenize_for_bm25(page_contextualized),
                metadata=metadata
            )
            
            all_chunks.append(page_chunk)
            
            # Level 2: Create section chunks within this page
            section_chunks_text = self.create_section_chunks(page_text)
            print(f"  Page {page_idx}: Created {len(section_chunks_text)} section chunks")
            
            for section_idx, section_text in enumerate(section_chunks_text):
                section_id = self.generate_chunk_id(document_id, 'section', section_idx, page_id)
                
                # Generate contextual summary for section
                section_summary = await self.generate_contextual_summary(
                    section_text,
                    page_summary,
                    title,
                    'section'
                )
                
                # Create contextualized text
                section_contextualized = f"{section_summary}\n\n{section_text}"
                
                # Create section chunk
                section_chunk = ChunkData(
                    document_id=document_id,
                    id=section_id,
                    chunk_level='paragraph',
                    chunk_index=section_idx,
                    chunk_text=section_text,
                    chunk_size=self.count_tokens(section_text),
                    contextual_summary=section_summary,
                    contextualized_text=section_contextualized,
                    parent_chunk_id=page_id,
                    bm25_tokens=self.tokenize_for_bm25(section_contextualized),
                    metadata=metadata
                )
                
                all_chunks.append(section_chunk)
                
                # Level 3: Create semantic chunks within this section
                semantic_chunks = self.create_semantic_chunks(section_text, section_id, section_idx)
                print(f"    Section {section_idx}: Created {len(semantic_chunks)} semantic chunks")
                
                for semantic_idx, (semantic_text, sentence_count) in enumerate(semantic_chunks):
                    semantic_id = self.generate_chunk_id(document_id, 'semantic', semantic_idx, section_id)
                    
                    # Generate contextual summary for semantic chunk
                    semantic_summary = await self.generate_contextual_summary(
                        semantic_text,
                        section_summary,
                        title,
                        'semantic'
                    )
                    
                    # Identify semantic focus
                    semantic_focus = await self.identify_semantic_focus(semantic_text)
                    
                    # Create contextualized text
                    semantic_contextualized = f"{semantic_summary}\n\n{semantic_text}"
                    
                    # Create semantic chunk
                    semantic_chunk = ChunkData(
                        document_id=document_id,
                        id=semantic_id,
                        chunk_level='semantic',
                        chunk_index=semantic_idx,
                        chunk_text=semantic_text,
                        chunk_size=self.count_tokens(semantic_text),
                        contextual_summary=semantic_summary,
                        contextualized_text=semantic_contextualized,
                        parent_chunk_id=section_id,
                        bm25_tokens=self.tokenize_for_bm25(semantic_contextualized),
                        sentence_count=sentence_count,
                        semantic_focus=semantic_focus,
                        metadata=metadata
                    )
                    
                    all_chunks.append(semantic_chunk)
        
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"  - Page chunks: {len([c for c in all_chunks if c.chunk_level == 'page'])}")
        print(f"  - Paragraph chunks: {len([c for c in all_chunks if c.chunk_level == 'paragraph'])}")
        print(f"  - Semantic chunks: {len([c for c in all_chunks if c.chunk_level == 'semantic'])}")
        
        return all_chunks
    
    async def save_chunks(self, chunks: List[ChunkData]) -> bool:
        """Save chunks to database."""
        try:
            # Convert ChunkData objects to dicts
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                # Convert datetime objects if present
                if chunk_dict.get('created_at'):
                    chunk_dict['created_at'] = chunk_dict['created_at'].isoformat()
                if chunk_dict.get('updated_at'):
                    chunk_dict['updated_at'] = chunk_dict['updated_at'].isoformat()
                
                # Store semantic_focus in metadata if it exists
                semantic_focus = chunk_dict.pop('semantic_focus', None)
                if semantic_focus:
                    if not chunk_dict.get('metadata'):
                        chunk_dict['metadata'] = {}
                    chunk_dict['metadata']['semantic_focus'] = semantic_focus
                
                # Remove fields that don't exist in DB schema
                chunk_dict.pop('sentence_count', None)
                
                chunk_dicts.append(chunk_dict)
            
            # Insert chunks
            response = self.supabase.client.table('chunks').insert(chunk_dicts).execute()
            
            return len(response.data) == len(chunks)
            
        except Exception as e:
            print(f"Error saving chunks: {e}")
            return False
    
    async def process_and_save(
        self,
        document_id: str,
        content: str,
        title: str = "Document",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Process document and save chunks to database."""
        try:
            # Delete existing chunks for this document
            self.supabase.client.table('chunks').delete().eq('document_id', document_id).execute()
            
            # Process document
            chunks = await self.process_document(document_id, content, title, metadata)
            
            # Save chunks
            success = await self.save_chunks(chunks)
            
            if success:
                print(f"Successfully saved {len(chunks)} chunks for document {document_id}")
            else:
                print(f"Failed to save chunks for document {document_id}")
            
            return success
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return False