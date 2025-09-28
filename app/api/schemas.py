from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OperationResponse(BaseModel):
    message: str
    class Config:
        schema_extra = {"example": {"message": "Document updated successfully"}}


class ErrorResponse(BaseModel):
    detail: str
    class Config:
        schema_extra = {"example": {"detail": "Document not found"}}


class DocumentListItem(BaseModel):
    id: str
    title: Optional[str] = None
    name: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processed_at: Optional[str] = None
    chunk_count: int = 0
    entity_count: int = 0
    class Config:
        schema_extra = {
            "example": {
                "id": "2b0c1c68-7a0c-4f59-9b88-0e5d6c2a9a10",
                "title": "NC2056 Display Controller Troubleshooting",
                "name": "NC2056 Display Controller Troubleshooting",
                "source_type": "gdrive",
                "source_id": "1AbC...",
                "status": "pending_review",
                "metadata": {"category": "support", "tags": ["NC2056", "display"]},
                "created_at": "2025-09-02T12:34:56Z",
                "updated_at": "2025-09-02T13:01:00Z",
                "processed_at": "2025-09-02T12:59:00Z",
                "chunk_count": 42,
                "entity_count": 17
            }
        }


class ChunkOut(BaseModel):
    id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_number: Optional[int] = None
    chunk_index: Optional[int] = None
    chunk_text: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_level: Optional[str] = None
    contextual_summary: Optional[str] = None
    contextualized_text: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    parent_context: Optional[str] = None
    position_in_parent: Optional[int] = None
    bm25_tokens: Optional[Any] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    class Config:
        schema_extra = {
            "example": {
                "id": "a1b2c3",
                "document_id": "2b0c1c68-7a0c-4f59-9b88-0e5d6c2a9a10",
                "chunk_index": 3,
                "chunk_text": "The temperature sensor depends on the control board for proper operation.",
                "chunk_size": 96,
                "chunk_level": "semantic",
                "contextual_summary": "Dependency between sensor and control board",
                "start_position": 512,
                "end_position": 608,
                "metadata": {"hierarchy_level": "semantic"},
                "created_at": "2025-09-02T12:58:00Z"
            }
        }


class EntityOut(BaseModel):
    id: Optional[str] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        schema_extra = {
            "example": {
                "id": "e_123",
                "entity_name": "temperature sensor",
                "entity_type": "component",
                "confidence_score": 0.92,
                "metadata": {"start_offset": 512, "end_offset": 528}
            }
        }


class RelationshipOut(BaseModel):
    id: Optional[str] = None
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence_score: Optional[float] = None
    is_verified: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    class Config:
        schema_extra = {
            "example": {
                "id": "r_456",
                "source_entity_id": "e_123",
                "target_entity_id": "e_456",
                "relationship_type": "DEPENDS_ON",
                "confidence_score": 0.78,
                "is_verified": False,
                "metadata": {"proposal_method": "rule_based", "chunk_id": "a1b2c3"},
                "created_at": "2025-09-02T13:00:00Z"
            }
        }


class DocumentDetail(BaseModel):
    id: str
    title: Optional[str] = None
    name: Optional[str] = None
    content: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    mime_type: Optional[str] = None
    author: Optional[str] = None
    security_level: Optional[str] = None
    access_level: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processed_at: Optional[str] = None
    ingested_at: Optional[str] = None
    chunks: Optional[List[ChunkOut]] = None
    entities: Optional[List[EntityOut]] = None
    class Config:
        schema_extra = {
            "example": {
                "id": "2b0c1c68-7a0c-4f59-9b88-0e5d6c2a9a10",
                "title": "NC2056 Display Controller Troubleshooting",
                "content": "# NC2056 Troubleshooting...",
                "source_type": "gdrive",
                "status": "pending_review",
                "metadata": {"category": "support", "tags": ["NC2056", "display"]},
                "created_at": "2025-09-02T12:34:56Z",
                "updated_at": "2025-09-02T13:01:00Z",
                "processed_at": "2025-09-02T12:59:00Z",
                "chunks": [
                    {
                        "id": "a1b2c3",
                        "chunk_index": 1,
                        "chunk_text": "The NC2056 is a display controller board...",
                        "chunk_size": 120
                    }
                ],
                "entities": [
                    {
                        "id": "e_123",
                        "entity_name": "temperature sensor",
                        "entity_type": "component",
                        "confidence_score": 0.92
                    }
                ]
            }
        }


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    name: Optional[str] = None
    author: Optional[str] = None
    mime_type: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    class Config:
        schema_extra = {
            "example": {
                "title": "NC2056 Controller Troubleshooting (v2)",
                "metadata": {"tags": ["NC2056", "controller"], "department": "Support"}
            }
        }


class ExtractMetadataResponse(BaseModel):
    message: str
    document_id: str
    class Config:
        schema_extra = {"example": {"message": "Metadata extraction started", "document_id": "2b0c1c68-..."}}


class DocumentProcessResponse(BaseModel):
    document_id: str
    status: str
    message: str
    job_id: Optional[str] = None
    celery_task_id: Optional[str] = None
    class Config:
        schema_extra = {
            "example": {
                "document_id": "2b0c1c68-...",
                "status": "processing",
                "job_id": "b8d9e2d0-...",
                "celery_task_id": "a1b2c3d4e5",
                "message": "Document queued for processing via Celery"
            }
        }


class ApproveDocumentResponse(BaseModel):
    message: str
    status: str
    ingested_at: Optional[str] = None
    class Config:
        schema_extra = {
            "example": {
                "message": "Document 2b0c1c68-... approved successfully",
                "status": "ingested",
                "ingested_at": "2025-09-02T14:10:00Z"
            }
        }


class RejectDocumentResponse(BaseModel):
    message: str
    status: str
    reason: str
    can_reprocess: bool = True
    class Config:
        schema_extra = {
            "example": {
                "message": "Document 2b0c1c68-... rejected",
                "status": "rejected",
                "reason": "Parsing quality too low",
                "can_reprocess": True
            }
        }


class ReviewStatusOut(BaseModel):
    document_id: str
    status: Optional[str] = None
    can_approve: bool = False
    can_reject: bool = False
    can_reprocess: bool = False
    chunks_count: int = 0
    entities_count: Dict[str, int] = Field(default_factory=dict)
    review_info: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        schema_extra = {
            "example": {
                "document_id": "2b0c1c68-...",
                "status": "pending_review",
                "can_approve": True,
                "can_reject": True,
                "can_reprocess": False,
                "chunks_count": 42,
                "entities_count": {"total": 17, "verified": 5, "unverified": 12},
                "review_info": {"reviewed_at": None, "review_action": None},
                "metadata": {"category": "support"}
            }
        }


class BatchChunkUpdateItem(BaseModel):
    id: str
    text: str


class BatchUpdateChunksRequest(BaseModel):
    updates: List[BatchChunkUpdateItem]
    class Config:
        schema_extra = {
            "example": {
                "updates": [
                    {"id": "a1b2c3", "text": "Updated chunk text..."},
                    {"id": "d4e5f6", "text": "Another updated chunk text..."}
                ]
            }
        }


class BatchUpdateChunksResponse(BaseModel):
    message: str
    updated_count: int
    total_requested: int
    class Config:
        schema_extra = {
            "example": {
                "message": "Batch update completed",
                "updated_count": 2,
                "total_requested": 2
            }
        }
