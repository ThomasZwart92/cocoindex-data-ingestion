"""
Relationship Management API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import logging
from datetime import datetime

from app.services.supabase_service import SupabaseService
from app.api.schemas import RelationshipOut, OperationResponse, ErrorResponse

logger = logging.getLogger(__name__)

# Request models
class CreateRelationshipRequest(BaseModel):
    document_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence_score: Optional[float] = 1.0
    metadata: Optional[dict] = None

class UpdateRelationshipRequest(BaseModel):
    relationship_type: Optional[str] = None
    source_entity_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    confidence_score: Optional[float] = None
    approved: Optional[bool] = None
    metadata: Optional[dict] = None

class BulkRelationshipAction(BaseModel):
    ids: list[str]
    rationale: Optional[str] = None

router = APIRouter(prefix="/api/relationships", tags=["relationships"])

# Initialize Supabase service
supabase_service = SupabaseService()


@router.post(
    "/",
    response_model=RelationshipOut,
    summary="Create relationship",
    description="Create a custom relationship between two entities. Custom relationships are marked verified by default.",
    responses={500: {"model": ErrorResponse}},
)
async def create_relationship(request: CreateRelationshipRequest):
    """Create a new custom relationship between entities"""
    try:
        import uuid
        
        # Create the relationship data
        # NOTE: We intentionally do NOT store document_id for manually created relationships
        # This ensures they are preserved during document reprocessing
        metadata = request.metadata or {}
        metadata["manual"] = True  # Mark as manually created
        metadata["created_via"] = "api"
        # Store document context but not as document_id to preserve during reprocessing
        if request.document_id:
            metadata["document_context"] = request.document_id

        relationship_data = {
            "id": str(uuid.uuid4()),
            "source_entity_id": request.source_entity_id,
            "target_entity_id": request.target_entity_id,
            "relationship_type": request.relationship_type,
            "confidence_score": request.confidence_score,
            "is_verified": True,  # Custom relationships are pre-verified
            "verified_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Insert into canonical relationships
        response = supabase_service.client.table("canonical_relationships").insert(
            relationship_data
        ).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create relationship")
        
        logger.info(f"Created custom relationship: {relationship_data['id']}")
        return response.data[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{relationship_id}",
    response_model=RelationshipOut,
    summary="Update relationship",
    description="Update relationship attributes including type, endpoints, confidence, and approval status.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_relationship(
    relationship_id: str,
    request: UpdateRelationshipRequest
):
    """Update a relationship"""
    try:
        # Build update data
        update_data = {}
        
        if request.relationship_type is not None:
            update_data["relationship_type"] = request.relationship_type
        
        if request.source_entity_id is not None:
            update_data["source_entity_id"] = request.source_entity_id
            
        if request.target_entity_id is not None:
            update_data["target_entity_id"] = request.target_entity_id
        
        if request.confidence_score is not None:
            update_data["confidence_score"] = request.confidence_score
        
        if request.approved is not None:
            update_data["is_verified"] = request.approved
            if request.approved:
                update_data["verified_at"] = datetime.utcnow().isoformat()
        
        if request.metadata is not None:
            update_data["metadata"] = request.metadata
        
        if not update_data:
            # No fields to update - just return success without making DB call
            # This can happen when the same value is selected in a dropdown
            check_response = supabase_service.client.table("canonical_relationships").select("*").eq(
                "id", relationship_id
            ).execute()
            
            if not check_response.data:
                raise HTTPException(status_code=404, detail="Relationship not found")
            
            logger.info(f"No changes needed for relationship {relationship_id}")
            return check_response.data[0]
        
        # Update in Supabase
        response = supabase_service.client.table("canonical_relationships").update(
            update_data
        ).eq("id", relationship_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Relationship not found")
        
        logger.info(f"Updated relationship {relationship_id}")
        return response.data[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating relationship {relationship_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{relationship_id}",
    response_model=OperationResponse,
    summary="Delete relationship",
    description="Delete a relationship by its ID.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_relationship(relationship_id: str):
    """Delete a relationship"""
    try:
        # Check if relationship exists
        check_response = supabase_service.client.table("canonical_relationships").select("id").eq(
            "id", relationship_id
        ).execute()
        
        if not check_response.data:
            raise HTTPException(status_code=404, detail="Relationship not found")
        
        # Delete from Supabase
        response = supabase_service.client.table("canonical_relationships").delete().eq(
            "id", relationship_id
        ).execute()
        
        logger.info(f"Deleted relationship {relationship_id}")
        return {"message": "Relationship deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting relationship {relationship_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/bulk-approve",
    response_model=OperationResponse,
    summary="Bulk approve relationships",
    description="Mark multiple relationships as verified in a single request.",
)
async def bulk_approve_relationships(payload: BulkRelationshipAction):
    try:
        if not payload.ids:
            return {"message": "No relationship ids provided"}
        updated = 0
        for rid in payload.ids:
            updates = {
                "is_verified": True,
                "verified_at": datetime.utcnow().isoformat(),
            }
            if payload.rationale:
                # Merge rationale into metadata if present
                try:
                    existing = supabase_service.client.table("canonical_relationships").select("metadata").eq("id", rid).execute()
                    meta = (existing.data[0].get("metadata") if existing.data else {}) or {}
                    meta["review_rationale"] = payload.rationale
                    updates["metadata"] = meta
                except Exception:
                    pass
            resp = supabase_service.client.table("canonical_relationships").update(updates).eq("id", rid).execute()
            if resp.data:
                updated += 1
        return {"message": f"Approved {updated}/{len(payload.ids)} relationships"}
    except Exception as e:
        logger.error(f"Bulk approve failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/bulk-reject",
    response_model=OperationResponse,
    summary="Bulk reject relationships",
    description="Reject multiple relationships by marking a rejection flag in metadata.",
)
async def bulk_reject_relationships(payload: BulkRelationshipAction):
    try:
        if not payload.ids:
            return {"message": "No relationship ids provided"}
        updated = 0
        for rid in payload.ids:
            try:
                existing = supabase_service.client.table("canonical_relationships").select("metadata").eq("id", rid).execute()
                meta = (existing.data[0].get("metadata") if existing.data else {}) or {}
            except Exception:
                meta = {}
            meta["proposal_rejected"] = True
            if payload.rationale:
                meta["review_rationale"] = payload.rationale
            resp = supabase_service.client.table("canonical_relationships").update({
                "metadata": meta,
                "is_verified": False
            }).eq("id", rid).execute()
            if resp.data:
                updated += 1
        return {"message": f"Rejected {updated}/{len(payload.ids)} relationships"}
    except Exception as e:
        logger.error(f"Bulk reject failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
