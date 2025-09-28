"""
Two-tier hierarchical chunking processor (formerly three_tier_chunker.py) for maximum retrieval accuracy.
Implements a simplified version of Anthropic's Contextual Retrieval:

- Parent chunks (page-like, broad context)
- Semantic chunks (small, single-concept) derived directly from each parent

Removes the middle paragraph/section layer to reduce complexity while
preserving precision and context for retrieval.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
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
    chunk_level: Optional[str] = None  # Database doesn't accept custom values, use metadata instead
    chunk_index: int = 0
    chunk_text: str = ''  # Changed from content to chunk_text to match DB
    chunk_size: int = 0  # Changed from token_count to chunk_size to match DB
    chunking_strategy: str = 'semantic'  # Required field in DB
    contextual_summary: str = ''
    contextualized_text: str = ''
    parent_chunk_id: Optional[str] = None
    bm25_tokens: Optional[List[str]] = None
    sentence_count: Optional[int] = None
    semantic_focus: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TwoTierChunker:
    """
    Two-tier hierarchical chunking:
    1. Parent level (page-like, ~1200 tokens) - broad document context
    2. Semantic level (1-3 sentences, ≤ ~100 tokens) - single-concept chunks
    """
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.llm_service = LLMService()
        self.supabase = SupabaseService()
        
        # Chunking parameters
        self.page_token_size = 1200
        self.page_overlap = 200
        # Removed paragraph/section tier
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
    
    def _split_into_heading_blocks(self, text: str) -> List[Tuple[Optional[str], str]]:
        """Split text into blocks by markdown headings.

        Returns list of (heading, body) where heading includes the full
        markdown heading line (e.g., "### Cause") without trailing newlines.
        The body contains text until the next heading. If text exists before
        the first heading, it is merged with the first heading block so the
        preamble stays attached to the first section.
        """
        pattern = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if not matches:
            return [(None, text.strip())] if text.strip() else []

        blocks: List[Tuple[Optional[str], str]] = []

        # Preamble before the first heading
        pre_start = 0
        pre_end = matches[0].start()
        preamble = text[pre_start:pre_end].strip()

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
            block_text = text[start:end].strip()

            # Extract the heading line and the remaining body
            lines = block_text.splitlines()
            heading_line = lines[0].strip() if lines else None
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            # Merge preamble into the first heading block if present
            if i == 0 and preamble:
                merged_body = preamble + ("\n\n" + body if body else "")
                blocks.append((heading_line, merged_body))
            else:
                blocks.append((heading_line, body))

        return blocks

    def create_semantic_chunks(self, text: str, parent_id: str, section_idx: int) -> List[Tuple[str, int]]:
        """Create semantic chunks that are heading-aware.

        Headings form hard boundaries. Each heading block is chunked
        independently so content from different sections never mixes.
        """
        blocks = self._split_into_heading_blocks(text)
        all_chunks: List[Tuple[str, int]] = []

        if not blocks:
            return []

        # Try semantic chunker; if unavailable, fallback per block
        try:
            from app.processors.semantic_chunker import SemanticChunker

            semantic_chunker = SemanticChunker(
                similarity_threshold=0.5,
                min_chunk_size=0,  # No minimum size as requested
                max_chunk_size=self.semantic_max_tokens,
                min_sentences=1,
                max_sentences=self.semantic_max_sentences,
            )

            for heading, body in blocks:
                block_text = body if body else ""
                # If there is no body but there is a heading, keep the heading alone
                if not block_text and heading:
                    all_chunks.append((heading, 0))
                    continue

                semantic_results = semantic_chunker.create_semantic_chunks(
                    block_text, maintain_context=True
                )

                first = True
                for chunk_text, metadata in semantic_results:
                    text_out = chunk_text
                    if first and heading:
                        text_out = f"{heading}\n\n{text_out}" if text_out else heading
                        first = False
                    all_chunks.append((text_out, metadata.get("sentence_count", 0)))

            return all_chunks

        except ImportError:
            print(
                "Warning: SemanticChunker not available, using heading-aware sentence chunking"
            )

            for heading, body in blocks:
                sentences = self.split_into_sentences(body) if body else []
                current_chunk: List[str] = []
                current_tokens = 0
                first = True

                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)

                    if current_chunk and (
                        len(current_chunk) >= self.semantic_max_sentences
                        or current_tokens + sentence_tokens > self.semantic_max_tokens
                    ):
                        chunk_text = " ".join(current_chunk)
                        if first and heading:
                            chunk_text = f"{heading}\n\n{chunk_text}" if chunk_text else heading
                            first = False
                        all_chunks.append((chunk_text, len(current_chunk)))
                        current_chunk = []
                        current_tokens = 0

                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens

                if current_chunk or heading:
                    chunk_text = " ".join(current_chunk)
                    if first and heading:
                        chunk_text = f"{heading}\n\n{chunk_text}" if chunk_text else heading
                    all_chunks.append((chunk_text, len(current_chunk)))

            return all_chunks
    
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
        else:  # parent level
            prompt = f"""Document: {doc_title}

Parent Content Summary: {chunk_text[:600]}

Write 2-3 sentences summarizing the key topics and themes covered in this part of the document."""
        
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
        Process document through two-tier chunking hierarchy.
        Returns all chunks as a flat list with parent-child relationships.
        """
        all_chunks = []
        
        print(f"Processing document '{title}' with two-tier chunking...")
        
        # Tier 1: Create parent chunks (page-like)
        parent_chunks_text = self.create_page_chunks(content)
        print(f"Created {len(parent_chunks_text)} parent chunks")
        
        for parent_idx, parent_text in enumerate(parent_chunks_text):
            parent_id = self.generate_chunk_id(document_id, 'parent', parent_idx)
            
            # Generate contextual summary for page
            parent_summary = await self.generate_contextual_summary(
                parent_text,
                title,
                title,
                'parent'
            )
            
            # Create contextualized text
            parent_contextualized = f"{parent_summary}\n\n{parent_text}"
            
            # Create page chunk
            now = datetime.utcnow()
            parent_metadata = metadata.copy() if metadata else {}
            parent_metadata['tier'] = 'parent'
            parent_chunk = ChunkData(
                document_id=document_id,
                id=parent_id,
                chunk_level='page',  # Use 'page' for parent chunks as per database constraint
                chunk_index=parent_idx,
                chunk_text=parent_text,
                chunk_size=self.count_tokens(parent_text),
                contextual_summary=parent_summary,
                contextualized_text=parent_contextualized,
                bm25_tokens=self.tokenize_for_bm25(parent_contextualized),
                metadata=parent_metadata,
                created_at=now,
                updated_at=now
            )
            
            all_chunks.append(parent_chunk)

            # Tier 2: Create semantic chunks directly from the parent
            semantic_chunks = self.create_semantic_chunks(parent_text, parent_id, parent_idx)
            print(f"  Parent {parent_idx}: Created {len(semantic_chunks)} semantic chunks")

            for semantic_idx, (semantic_text, sentence_count) in enumerate(semantic_chunks):
                semantic_id = self.generate_chunk_id(document_id, 'semantic', semantic_idx, parent_id)

                # Generate contextual summary for semantic chunk
                semantic_summary = await self.generate_contextual_summary(
                    semantic_text,
                    parent_summary,
                    title,
                    'semantic'
                )

                # Identify semantic focus
                semantic_focus = await self.identify_semantic_focus(semantic_text)

                # Create contextualized text
                semantic_contextualized = f"{semantic_summary}\n\n{semantic_text}"

                # Create semantic chunk
                now = datetime.utcnow()
                semantic_metadata = metadata.copy() if metadata else {}
                semantic_metadata['tier'] = 'semantic'
                semantic_metadata['semantic_focus'] = semantic_focus
                semantic_chunk = ChunkData(
                    document_id=document_id,
                    id=semantic_id,
                    chunk_level='semantic',  # Use 'semantic' for semantic chunks as per database constraint
                    chunk_index=semantic_idx,
                    chunk_text=semantic_text,
                    chunk_size=self.count_tokens(semantic_text),
                    contextual_summary=semantic_summary,
                    contextualized_text=semantic_contextualized,
                    parent_chunk_id=parent_id,
                    bm25_tokens=self.tokenize_for_bm25(semantic_contextualized),
                    sentence_count=sentence_count,
                    semantic_focus=semantic_focus,
                    metadata=semantic_metadata,
                    created_at=now,
                    updated_at=now
                )

                all_chunks.append(semantic_chunk)
        
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"  - Parent chunks: {len([c for c in all_chunks if c.metadata and c.metadata.get('tier') == 'parent'])}")
        print(f"  - Semantic chunks: {len([c for c in all_chunks if c.metadata and c.metadata.get('tier') == 'semantic'])}")
        
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
