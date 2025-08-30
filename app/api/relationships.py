"""
Relationship Management API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime

from app.services.supabase_service import SupabaseService

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

router = APIRouter(prefix="/api/relationships", tags=["relationships"])

# Initialize Supabase service
supabase_service = SupabaseService()


@router.post("/")
async def create_relationship(request: CreateRelationshipRequest):
    """Create a new custom relationship between entities"""
    try:
        import uuid
        
        # Create the relationship data
        relationship_data = {
            "id": str(uuid.uuid4()),
            "document_id": request.document_id,
            "source_entity_id": request.source_entity_id,
            "target_entity_id": request.target_entity_id,
            "relationship_type": request.relationship_type,
            "confidence_score": request.confidence_score,
            "is_verified": True,  # Custom relationships are pre-verified
            "verified_at": datetime.utcnow().isoformat(),
            "metadata": request.metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Insert into Supabase
        response = supabase_service.client.table("entity_relationships").insert(
            relationship_data
        ).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create relationship")
        
        logger.info(f"Created custom relationship: {relationship_data['id']}")
        return {"message": "Relationship created successfully", "data": response.data[0]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{relationship_id}")
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
            check_response = supabase_service.client.table("entity_relationships").select("*").eq(
                "id", relationship_id
            ).execute()
            
            if not check_response.data:
                raise HTTPException(status_code=404, detail="Relationship not found")
            
            logger.info(f"No changes needed for relationship {relationship_id}")
            return {"message": "No changes needed", "data": check_response.data[0]}
        
        # Update in Supabase
        response = supabase_service.client.table("entity_relationships").update(
            update_data
        ).eq("id", relationship_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Relationship not found")
        
        logger.info(f"Updated relationship {relationship_id}")
        return {"message": "Relationship updated successfully", "data": response.data[0]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating relationship {relationship_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{relationship_id}")
async def delete_relationship(relationship_id: str):
    """Delete a relationship"""
    try:
        # Check if relationship exists
        check_response = supabase_service.client.table("entity_relationships").select("id").eq(
            "id", relationship_id
        ).execute()
        
        if not check_response.data:
            raise HTTPException(status_code=404, detail="Relationship not found")
        
        # Delete from Supabase
        response = supabase_service.client.table("entity_relationships").delete().eq(
            "id", relationship_id
        ).execute()
        
        logger.info(f"Deleted relationship {relationship_id}")
        return {"message": "Relationship deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting relationship {relationship_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))