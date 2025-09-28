from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ExtractionRun:
    id: str
    document_id: str
    pipeline_version: str
    prompt_version: Optional[str] = None
    model: Optional[str] = None
    status: str = "pending"
    input_hash: Optional[str] = None
    cache_hits: int = 0
    mentions_extracted: int = 0
    entities_canonicalized: int = 0
    relationships_inferred: int = 0


@dataclass
class EntityMention:
    """Single occurrence of an entity in text with offsets."""

    # Required fields expected from LLM output
    text: str
    type: str
    start_offset: int
    end_offset: int
    confidence: float

    # Optional fields we fill in downstream
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    extraction_run_id: Optional[str] = None
    context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    canonical_entity_id: Optional[str] = None
    canonicalization_score: Optional[float] = None


@dataclass
class CanonicalEntity:
    """Deduplicated canonical entity across mentions/documents."""

    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    quality_score: float = 0.5
    is_validated: bool = False
    definition: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRelationship:
    """Relationship between canonical entities."""

    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
