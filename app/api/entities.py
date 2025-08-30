"""
Entity Management API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
import logging

from app.models.entity import Entity as EntityModel
from app.models.document import Document, DocumentStatus
from app.services.supabase_service import SupabaseService
from app.config import settings
# Import bridge endpoint for alias
from app.api.bridge import get_document_entities as bridge_get_entities

logger = logging.getLogger(__name__)

# Request models
class CreateEntityRequest(BaseModel):
    document_id: str
    entity_name: str
    entity_type: str
    confidence: float = 1.0
    metadata: Optional[dict] = None
    manual: Optional[bool] = False

class UpdateEntityRequest(BaseModel):
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[dict] = None
    merge_metadata: Optional[bool] = True

router = APIRouter(prefix="/api/entities", tags=["entities"])

# Initialize Supabase service
supabase_service = SupabaseService()


# Note: This alias is placed after specific entity routes to avoid conflicts
# We'll add it later in the file to ensure proper route ordering


@router.get("/", response_model=List[dict])
async def list_entities(
    document_id: Optional[str] = Query(None, description="Filter by document"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score"),
    limit: int = Query(100, ge=1, le=500, description="Maximum entities to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List entities with optional filtering
    """
    try:
        # Build query with Supabase
        query = supabase_service.client.table("entities").select("*")
        
        # Apply filters
        if document_id:
            query = query.eq("document_id", document_id)
        if entity_type:
            query = query.eq("entity_type", entity_type)
        if min_confidence > 0:
            query = query.gte("confidence_score", min_confidence)
        
        # Execute query with pagination
        query = query.range(offset, offset + limit - 1)
        response = query.execute()
        
        # Convert to expected format
        result = []
        for entity in response.data:
            result.append({
                "id": entity.get("id"),
                "document_id": entity.get("document_id"),
                "name": entity.get("entity_name"),  # Note: column name is entity_name
                "type": entity.get("entity_type"),  # Note: column name is entity_type
                "confidence": entity.get("confidence_score", 0.0),  # Note: column name is confidence_score
                "metadata": entity.get("metadata") or {},
                "created_at": entity.get("created_at"),
                "updated_at": entity.get("updated_at")
            })
        
        logger.info(f"Listed {len(result)} entities")
        return result
        
    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        return []  # Return empty list on error to prevent crash


@router.get("/{entity_id}")
async def get_entity(
    entity_id: str,  # Changed from UUID to str to handle both entity_id and document_id
    include_relationships: bool = Query(False, description="Include entity relationships")
):
    """
    Get a single entity or entities for a document.
    
    Note: The frontend may call this with a document_id instead of entity_id.
    We'll handle both cases properly.
    """
    try:
        # First check if this is actually a document_id by checking length and format
        # Document IDs are UUIDs, entity IDs might be different
        logger.info(f"Getting entity/document entities for ID: {entity_id}")
        
        # Try to fetch entities from Supabase
        response = supabase_service.client.table("entities").select("*").eq("document_id", entity_id).execute()
        
        if response.data:
            # This is a document_id, return entities for the document
            entities = response.data
            logger.info(f"Found {len(entities)} entities for document {entity_id}")
            
            # Convert to expected format (frontend expects array directly)
            result = []
            for entity in entities:
                result.append({
                    "id": entity.get("id"),
                    "document_id": entity.get("document_id"),
                    "entity_name": entity.get("entity_name"),
                    "entity_type": entity.get("entity_type"),
                    "confidence": entity.get("confidence_score", 0.0),
                    "metadata": entity.get("metadata") or {},
                    "created_at": entity.get("created_at"),
                    "updated_at": entity.get("updated_at")
                })
            
            # Return array directly for frontend compatibility
            return result
        else:
            # Try to fetch as a single entity
            entity_response = supabase_service.client.table("entities").select("*").eq("id", entity_id).execute()
            
            if entity_response.data and len(entity_response.data) > 0:
                entity = entity_response.data[0]
                
                # Convert to expected format and return single entity as array
                return [{
                    "id": entity.get("id"),
                    "document_id": entity.get("document_id"),
                    "entity_name": entity.get("entity_name"),
                    "entity_type": entity.get("entity_type"),
                    "confidence": entity.get("confidence_score", 0.0),
                    "metadata": entity.get("metadata") or {},
                    "created_at": entity.get("created_at"),
                    "updated_at": entity.get("updated_at")
                }]
            else:
                # No entity found, return empty array
                return []
        
    except Exception as e:
        logger.error(f"Error getting entity {entity_id}: {e}")
        # Return empty array instead of error to prevent frontend crash
        return []


@router.post("/", response_model=dict)
async def create_entity(request: CreateEntityRequest):
    """
    Create a new entity
    """
    try:
        # Verify document exists
        doc_response = supabase_service.client.table("documents").select("*").eq("id", request.document_id).execute()
        if not doc_response.data or len(doc_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
        
        # Check for duplicate
        existing_response = supabase_service.client.table("entities").select("*").eq("document_id", request.document_id).eq("entity_name", request.entity_name).eq("entity_type", request.entity_type).execute()
        
        if existing_response.data and len(existing_response.data) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Entity '{request.entity_name}' of type '{request.entity_type}' already exists for this document"
            )
        
        # Prepare metadata
        metadata = request.metadata or {}
        if request.manual:
            metadata["manually_created"] = True
        
        # Create entity
        entity_data = {
            "id": str(uuid4()),
            "document_id": request.document_id,
            "entity_name": request.entity_name,
            "entity_type": request.entity_type,
            "confidence_score": request.confidence,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat()
        }
        
        entity_response = supabase_service.client.table("entities").insert(entity_data).execute()
        
        if not entity_response.data or len(entity_response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create entity")
        
        entity = entity_response.data[0]
        
        # Update document status
        supabase_service.client.table("documents").update({
            "status": "pending_review",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", request.document_id).execute()
        
        logger.info(f"Created entity {entity['id']}: {request.entity_name} ({request.entity_type})")
        return {
            "id": entity["id"],
            "document_id": entity["document_id"],
            "entity_name": entity["entity_name"],
            "entity_type": entity["entity_type"],
            "confidence": entity["confidence_score"],
            "metadata": entity.get("metadata", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{entity_id}")
async def update_entity(
    entity_id: str,
    request: UpdateEntityRequest
):
    """
    Update an entity
    """
    try:
        # Fetch entity from Supabase
        entity_response = supabase_service.client.table("entities").select("*").eq("id", entity_id).execute()
        if not entity_response.data or len(entity_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
        
        entity = entity_response.data[0]
        
        # Store original values for history
        original = {
            "name": entity.get("entity_name"),
            "type": entity.get("entity_type"),
            "confidence": entity.get("confidence_score", 0.0)
        }
        
        # Build update data
        update_data = {}
        if request.entity_name is not None:
            update_data["entity_name"] = request.entity_name
        if request.entity_type is not None:
            update_data["entity_type"] = request.entity_type
        if request.confidence is not None:
            update_data["confidence_score"] = request.confidence
        
        # Update metadata
        existing_metadata = entity.get("metadata", {})
        if request.metadata is not None:
            if request.merge_metadata and existing_metadata:
                updated_metadata = {**existing_metadata, **request.metadata}
            else:
                updated_metadata = request.metadata
        else:
            updated_metadata = existing_metadata
        
        # Add edit history
        if "edit_history" not in updated_metadata:
            updated_metadata["edit_history"] = []
        
        updated_metadata["edit_history"].append({
            "edited_at": datetime.utcnow().isoformat(),
            "original": original,
            "manually_edited": True
        })
        
        update_data["metadata"] = updated_metadata
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Update entity in Supabase
        supabase_service.client.table("entities").update(update_data).eq("id", entity_id).execute()
        
        # Update document status
        document_id = entity.get("document_id")
        if document_id:
            supabase_service.client.table("documents").update({
                "status": "pending_review",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", document_id).execute()
        
        logger.info(f"Updated entity {entity_id}")
        return {
            "message": f"Entity {entity_id} updated successfully",
            "entity": {
                "id": entity_id,
                "name": update_data.get("entity_name", entity.get("entity_name")),
                "type": update_data.get("entity_type", entity.get("entity_type")),
                "confidence": update_data.get("confidence_score", entity.get("confidence_score", 0.0))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{entity_id}")
async def delete_entity(
    entity_id: str
):
    """
    Delete an entity
    """
    try:
        # Get entity to find document_id
        entity_response = supabase_service.client.table("entities").select("*").eq("id", entity_id).execute()
        if not entity_response.data or len(entity_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
        
        entity = entity_response.data[0]
        document_id = entity.get("document_id")
        
        # Delete entity from Supabase
        supabase_service.client.table("entities").delete().eq("id", entity_id).execute()
        
        # Update document status
        if document_id:
            supabase_service.client.table("documents").update({
                "status": "pending_review",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", document_id).execute()
        
        logger.info(f"Deleted entity {entity_id}")
        return {"message": f"Entity {entity_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge")
async def merge_entities(
    entity_ids: List[str],
    target_name: str,
    target_type: str
):
    """
    Merge multiple entities into one (entity resolution)
    """
    try:
        if len(entity_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 entities required for merging")
        
        # Get all entities from Supabase
        entities_response = supabase_service.client.table("entities").select("*").in_("id", entity_ids).execute()
        entities = entities_response.data if entities_response.data else []
        
        if len(entities) != len(entity_ids):
            raise HTTPException(status_code=404, detail="One or more entities not found")
        
        # Create merged entity data
        merged_confidence = max(e.get("confidence_score", 0.0) for e in entities)
        merged_metadata = {
            "merged_from": [e["id"] for e in entities],
            "original_names": [e.get("entity_name", "") for e in entities],
            "original_types": list(set(e.get("entity_type", "") for e in entities))
        }
        
        # Use first entity's document_id
        document_id = entities[0].get("document_id")
        
        # Update first entity as the merged entity
        merged_id = entities[0]["id"]
        existing_metadata = entities[0].get("metadata", {})
        
        update_data = {
            "entity_name": target_name,
            "entity_type": target_type,
            "confidence_score": merged_confidence,
            "metadata": {**existing_metadata, **merged_metadata},
            "updated_at": datetime.utcnow().isoformat()
        }
        
        supabase_service.client.table("entities").update(update_data).eq("id", merged_id).execute()
        
        # Delete other entities
        other_ids = [e["id"] for e in entities[1:]]
        if other_ids:
            supabase_service.client.table("entities").delete().in_("id", other_ids).execute()
        
        # Update document status
        if document_id:
            supabase_service.client.table("documents").update({
                "status": "pending_review",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", document_id).execute()
        
        logger.info(f"Merged {len(entities)} entities into {merged_id}")
        return {
            "message": f"Successfully merged {len(entities)} entities",
            "merged_entity": {
                "id": merged_id,
                "name": target_name,
                "type": target_type,
                "confidence": merged_confidence
            },
            "deleted_entities": len(entities) - 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/entities")
async def get_document_entities(
    document_id: str,
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence")
):
    """
    Get all entities for a specific document
    """
    try:
        # Verify document exists
        doc_response = supabase_service.client.table("documents").select("*").eq("id", document_id).execute()
        if not doc_response.data or len(doc_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Query entities
        query = supabase_service.client.table("entities").select("*").eq("document_id", document_id)
        
        if entity_type:
            query = query.eq("entity_type", entity_type)
        if min_confidence > 0:
            query = query.gte("confidence_score", min_confidence)
        
        # Order by confidence descending (Supabase syntax)
        query = query.order("confidence_score", desc=True)
        
        entities_response = query.execute()
        entities = entities_response.data if entities_response.data else []
        
        # Group by type for summary
        by_type = {}
        for entity in entities:
            entity_type_val = entity.get("entity_type", "Unknown")
            if entity_type_val not in by_type:
                by_type[entity_type_val] = []
            by_type[entity_type_val].append({
                "id": entity.get("id"),
                "name": entity.get("entity_name"),
                "confidence": entity.get("confidence_score", 0.0),
                "metadata": entity.get("metadata", {})
            })
        
        return {
            "document_id": document_id,
            "total_entities": len(entities),
            "entities_by_type": by_type,
            "all_entities": [
                {
                    "id": e.get("id"),
                    "name": e.get("entity_name"),
                    "type": e.get("entity_type"),
                    "confidence": e.get("confidence_score", 0.0),
                    "metadata": e.get("metadata", {})
                }
                for e in entities
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entities for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))