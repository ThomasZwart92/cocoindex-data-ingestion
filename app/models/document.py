"""Document model and state machine"""
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class DocumentState(str, Enum):
    """Document processing states"""
    DISCOVERED = "discovered"
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    INGESTED = "ingested"
    FAILED = "failed"
    DELETED = "deleted"
    
    @classmethod  
    def valid_transitions(cls) -> Dict[str, List[str]]:
        """Define valid state transitions"""
        return {
            cls.DISCOVERED: [cls.PROCESSING, cls.FAILED],
            cls.PROCESSING: [cls.PENDING_REVIEW, cls.FAILED],
            cls.PENDING_REVIEW: [cls.APPROVED, cls.REJECTED, cls.PROCESSING],
            cls.APPROVED: [cls.INGESTED, cls.FAILED],
            cls.REJECTED: [cls.PROCESSING, cls.DISCOVERED],
            cls.INGESTED: [cls.PROCESSING],  # Allow reprocessing
            cls.FAILED: [cls.PROCESSING, cls.DISCOVERED],
            cls.DELETED: []  # No transitions from deleted
        }
    
    def can_transition_to(self, new_state: 'DocumentState') -> bool:
        """Check if transition to new state is valid"""
        valid_states = self.valid_transitions().get(self, [])
        return new_state in valid_states

# Alias for backwards compatibility
DocumentStatus = DocumentState

class SourceType(str, Enum):
    """Document source types"""
    UPLOAD = "upload"
    NOTION = "notion"
    GDRIVE = "gdrive"
    URL = "url"

class ParseTier(str, Enum):
    """LlamaParse parsing tiers"""
    BALANCED = "balanced"
    AGENTIC = "agentic"
    AGENTIC_PLUS = "agentic_plus"

class Document(BaseModel):
    """Document model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_type: SourceType
    source_url: Optional[str] = None
    source_id: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    
    # Document content
    content: Optional[str] = None
    
    # Security
    security_level: Optional[str] = None
    access_level: Optional[int] = None
    
    # Processing state
    status: DocumentState = DocumentState.DISCOVERED
    processing_error: Optional[str] = None
    retry_count: int = 0
    
    # Parsing details
    parse_tier: Optional[ParseTier] = None
    parse_confidence: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_metadata: Dict[str, Any] = Field(default_factory=dict)  # Alias for Supabase compatibility
    tags: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    
    # Versioning
    version: int = 1
    parent_document_id: Optional[str] = None
    
    def transition_to(self, new_state: DocumentState, error: Optional[str] = None) -> bool:
        """Transition document to new state"""
        if not self.status.can_transition_to(new_state):
            raise ValueError(f"Invalid transition from {self.status} to {new_state}")
        
        self.status = new_state
        self.updated_at = datetime.utcnow()
        
        if new_state == DocumentState.FAILED and error:
            self.processing_error = error
            self.retry_count += 1
        elif new_state == DocumentState.PROCESSING:
            self.processing_error = None
        elif new_state == DocumentState.INGESTED:
            self.processed_at = datetime.utcnow()
        elif new_state == DocumentState.APPROVED:
            self.approved_at = datetime.utcnow()
            
        return True
    
    def to_supabase_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion"""
        data = self.dict(exclude_none=True)
        # Convert enums to strings (handle both enum and string types)
        if hasattr(self.status, 'value'):
            data['status'] = self.status.value
        else:
            data['status'] = self.status
            
        if hasattr(self.source_type, 'value'):
            data['source_type'] = self.source_type.value
        else:
            data['source_type'] = self.source_type
            
        if self.parse_tier:
            if hasattr(self.parse_tier, 'value'):
                data['parse_tier'] = self.parse_tier.value
            else:
                data['parse_tier'] = self.parse_tier
        # Convert datetime to ISO format
        for field in ['created_at', 'updated_at', 'processed_at', 'approved_at']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        return data
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """Document chunk model for text segments"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_number: int
    chunk_text: str
    chunk_size: int
    start_position: int = 0
    end_position: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    title: str
    source_type: str
    source_id: str
    security_level: str
    access_level: int
    department: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)