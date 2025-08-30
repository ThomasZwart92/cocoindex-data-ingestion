"""Chunk model"""
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class ChunkingStrategy(str, Enum):
    """Chunking strategies"""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SENTENCE = "sentence"

class Chunk(BaseModel):
    """Document chunk model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    
    # Chunk details
    chunk_index: int
    chunk_text: str
    chunk_size: Optional[int] = None
    
    # Chunking strategy
    chunking_strategy: ChunkingStrategy
    chunk_overlap: Optional[int] = None
    
    # Parent-child relationship
    parent_chunk_id: Optional[str] = None
    
    # Embeddings reference
    embedding_id: Optional[str] = None  # Qdrant point ID
    embedding_model: Optional[str] = None
    
    # Review status
    is_edited: bool = False
    original_text: Optional[str] = None
    edited_by: Optional[str] = None
    edited_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def edit(self, new_text: str, edited_by: str) -> None:
        """Edit chunk text"""
        if not self.is_edited:
            self.original_text = self.chunk_text
        self.chunk_text = new_text
        self.is_edited = True
        self.edited_by = edited_by
        self.edited_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        # Clear embedding since text changed
        self.embedding_id = None
    
    def to_supabase_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion"""
        data = self.dict(exclude_none=True)
        # Convert enums to strings (handle both enum and string types)
        if hasattr(self.chunking_strategy, 'value'):
            data['chunking_strategy'] = self.chunking_strategy.value
        else:
            data['chunking_strategy'] = self.chunking_strategy
        # Calculate chunk size if not set
        if not data.get('chunk_size'):
            data['chunk_size'] = len(self.chunk_text)
        # Convert datetime to ISO format
        for field in ['created_at', 'updated_at', 'edited_at']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        return data
    
    class Config:
        use_enum_values = True