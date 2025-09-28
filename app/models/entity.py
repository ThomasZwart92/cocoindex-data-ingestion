"""Entity and relationship models"""
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class EntityType(Enum):
    """Types of entities that can be extracted"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    PRODUCT = "product"
    CONCEPT = "concept"
    EVENT = "event"
    COMPONENT = "component"  # Hardware/software components
    TECHNOLOGY = "technology"  # Frameworks, tools, platforms
    CHEMICAL = "chemical"  # Chemical substances, materials
    PROCEDURE = "procedure"  # Methods, processes, techniques
    SPECIFICATION = "specification"  # Standards, requirements, specs
    SYSTEM = "system"  # Complex systems, subsystems
    MEASUREMENT = "measurement"  # Measurements, dimensions, quantities
    PROBLEM = "problem"  # Issues, symptoms, error states
    STATE = "state"  # Operational modes or statuses (active, locked, failed)
    CONDITION = "condition"  # States of wear, degradation, or quality (corrosion, contamination)
    TOOL = "tool"  # Physical or software tools used in procedures
    MATERIAL = "material"  # Raw materials, supplies, consumables
    OTHER = "other"

class Entity(BaseModel):
    """Entity model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    
    # Entity details
    name: str  # Changed from entity_name for API compatibility
    type: str  # Changed from entity_type and made string for flexibility
    entity_name: Optional[str] = None  # Keep for backwards compatibility
    entity_type: Optional[EntityType] = None  # Keep for backwards compatibility
    confidence: Optional[float] = None
    
    # Source information
    source_chunk_id: Optional[str] = None
    source_model: Optional[str] = None  # Which LLM extracted this
    
    # Review status
    is_verified: bool = False
    is_edited: bool = False
    original_name: Optional[str] = None
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    # Entity resolution
    canonical_entity_id: Optional[str] = None  # Points to canonical version
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def verify(self, verified_by: str) -> None:
        """Mark entity as verified"""
        self.is_verified = True
        self.verified_by = verified_by
        self.verified_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def edit(self, new_name: str) -> None:
        """Edit entity name"""
        if not self.is_edited:
            self.original_name = self.entity_name
        self.entity_name = new_name
        self.is_edited = True
        self.updated_at = datetime.utcnow()
    
    def to_supabase_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion"""
        data = self.dict(exclude_none=True)
        # Convert datetime to ISO format
        for field in ['created_at', 'updated_at', 'verified_at']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        return data

class EntityRelationship(BaseModel):
    """Entity relationship model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence_score: Optional[float] = None
    
    # Source information
    source_chunk_id: Optional[str] = None
    source_model: Optional[str] = None
    
    # Review status
    is_verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def verify(self, verified_by: str) -> None:
        """Mark relationship as verified"""
        self.is_verified = True
        self.verified_by = verified_by
        self.verified_at = datetime.utcnow()
    
    def to_supabase_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion"""
        data = self.dict(exclude_none=True)
        # Convert datetime to ISO format
        for field in ['created_at', 'verified_at']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        return data