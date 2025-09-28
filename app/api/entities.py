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
from app.utils.entity_deduplication import EntityDeduplicator
# Import bridge endpoint for alias
from app.api.bridge import get_document_entities_neo4j as bridge_get_entities

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
    document_id: Optional[str] = Query(None, description="Filter by document (mentions)"),
    entity_type: Optional[str] = Query(None, description="Filter by canonical entity type"),
    min_quality: float = Query(0.0, ge=0.0, le=1.0, description="Minimum canonical quality score"),
    limit: int = Query(100, ge=1, le=500, description="Maximum entities to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List canonical entities with optional filtering.
    If document_id is provided, returns canonical entities that have mentions in that document.
    """
    try:
        client = supabase_service.client
        canonical_ids: Optional[list[str]] = None
        
        # If filtering by document, fetch canonical ids that have mentions in that document
        if document_id:
            logger.info(f"Filtering entities for document: {document_id}")
            mention_res = (
                client.table("entity_mentions")
                .select("canonical_entity_id")
                .eq("document_id", document_id)
                .not_.is_("canonical_entity_id", "null")
                .execute()
            )
            logger.info(f"Found {len(mention_res.data or [])} mentions with canonical_entity_id")
            ids = [m.get("canonical_entity_id") for m in (mention_res.data or []) if m.get("canonical_entity_id")]
            canonical_ids = list({i for i in ids}) if ids else []
            logger.info(f"Extracted {len(canonical_ids)} unique canonical entity IDs")
            if not canonical_ids:
                logger.warning("No canonical entity IDs found, returning empty list")
                return []

        # Build canonical_entities query
        cquery = client.table("canonical_entities").select("*")
        if canonical_ids is not None:
            logger.info(f"Filtering canonical_entities by {len(canonical_ids)} IDs")
            cquery = cquery.in_("id", canonical_ids)
        if entity_type:
            cquery = cquery.eq("type", entity_type)
        if min_quality > 0:
            cquery = cquery.gte("quality_score", min_quality)
        cquery = cquery.range(offset, offset + limit - 1)
        cres = cquery.execute()
        logger.info(f"Retrieved {len(cres.data or [])} canonical entities")

        # Map canonical entities and include mention counts when doc filter is present
        result = []
        for ce in (cres.data or []):
            item = {
                "id": ce.get("id"),
                "name": ce.get("name"),
                "type": ce.get("type"),
                "quality_score": ce.get("quality_score", 0.0),
                "is_validated": ce.get("is_validated", False),
                "aliases": ce.get("aliases", []),
                "metadata": ce.get("metadata", {}),
                "document_id": document_id,
            }
            if document_id:
                # Count mentions for this doc
                count_res = (
                    client.table("entity_mentions").select("id")
                    .eq("document_id", document_id)
                    .eq("canonical_entity_id", ce.get("id"))
                    .execute()
                )
                item["mentions_in_document"] = len(count_res.data or [])
            result.append(item)
        logger.info(f"Returning {len(result)} entities for document {document_id}")
        return result
    except Exception as e:
        logger.error(f"Error listing canonical entities: {e}", exc_info=True)
        return []


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
        
        client = supabase_service.client
        # Treat as document_id first: return canonical entities present in that document via mentions
        mention_res = client.table("entity_mentions").select("canonical_entity_id").eq("document_id", entity_id).neq("canonical_entity_id", None).execute()
        logger.info(f"Found {len(mention_res.data or [])} mentions with canonical_entity_id for document {entity_id}")
        if mention_res.data:
            ids = list({row.get("canonical_entity_id") for row in mention_res.data if row.get("canonical_entity_id")})
            logger.info(f"Extracted {len(ids)} unique canonical entity IDs")
            if not ids:
                logger.warning("No canonical entity IDs found in mentions, returning empty")
                return []
            ce_res = client.table("canonical_entities").select("*").in_("id", ids).execute()
            logger.info(f"Retrieved {len(ce_res.data or [])} canonical entities")
            result = []
            for ce in (ce_res.data or []):
                result.append({
                    "id": ce.get("id"),
                    "name": ce.get("name"),
                    "type": ce.get("type"),
                    "quality_score": ce.get("quality_score", 0.0),
                    "is_validated": ce.get("is_validated", False),
                    "metadata": ce.get("metadata", {}),
                    "document_id": entity_id,
                })
            logger.info(f"Returning {len(result)} entities for document {entity_id}")
            return result
        # Otherwise treat as canonical entity id and return it with mentions
        ce_res = client.table("canonical_entities").select("*").eq("id", entity_id).execute()
        if ce_res.data:
            ce = ce_res.data[0]
            # Fetch mentions across documents
            mentions_res = client.table("entity_mentions").select("id,document_id,chunk_id,text,type,start_offset,end_offset,confidence").eq("canonical_entity_id", entity_id).execute()
            return [{
                "id": ce.get("id"),
                "name": ce.get("name"),
                "type": ce.get("type"),
                "quality_score": ce.get("quality_score", 0.0),
                "is_validated": ce.get("is_validated", False),
                "metadata": ce.get("metadata", {}),
                "mentions": mentions_res.data or []
            }]
        return []
        
    except Exception as e:
        logger.error(f"Error getting entity {entity_id}: {e}")
        # Bubble up as HTTP error so the client can react appropriately
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/find-duplicates")
async def find_duplicate_entities(
    document_id: str,
    threshold: float = Query(0.85, ge=0.0, le=1.0, description="Similarity threshold"),
    auto_merge: bool = Query(False, description="Automatically merge high-confidence duplicates")
):
    """
    Find duplicate entities in a document and optionally auto-merge them.
    """
    try:
        # Get all entities for the document
        entities_response = supabase_service.client.table("entities")\
            .select("*")\
            .eq("document_id", document_id)\
            .execute()
        
        entities = entities_response.data if entities_response.data else []
        
        if len(entities) < 2:
            return {
                "duplicates": [],
                "merged": 0,
                "message": "Not enough entities to check for duplicates"
            }
        
        # Find duplicate groups
        duplicate_groups = EntityDeduplicator.find_duplicates(entities, threshold)
        
        merged_count = 0
        groups_for_review = []
        
        if auto_merge:
            # Auto-merge high confidence duplicates
            for group in duplicate_groups:
                # Calculate average similarity in group
                names = [e.get('entity_name', e.get('name', '')) for e in group]
                total_sim = 0
                count = 0
                
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        sim = EntityDeduplicator.calculate_similarity(names[i], names[j])
                        total_sim += sim
                        count += 1
                
                avg_similarity = total_sim / count if count > 0 else 0
                
                # Auto-merge if very high similarity
                if avg_similarity >= 0.95:
                    # Merge the group
                    merged_data = EntityDeduplicator.merge_entity_data(group)
                    
                    # Update first entity with merged data
                    first_id = group[0]['id']
                    update_data = {
                        'entity_name': merged_data.get('entity_name'),
                        'confidence_score': merged_data.get('confidence_score'),
                        'metadata': merged_data.get('metadata'),
                        'updated_at': datetime.utcnow().isoformat()
                    }
                    
                    supabase_service.client.table("entities")\
                        .update(update_data)\
                        .eq("id", first_id)\
                        .execute()
                    
                    # Delete other entities in group
                    other_ids = [e['id'] for e in group[1:]]
                    if other_ids:
                        supabase_service.client.table("entities")\
                            .delete()\
                            .in_("id", other_ids)\
                            .execute()
                    
                    merged_count += len(group) - 1
                    
                    logger.info(f"Auto-merged {len(group)} entities: {names}")
                else:
                    # Add to review list
                    groups_for_review.append({
                        "entities": group,
                        "similarity": avg_similarity,
                        "names": names
                    })
        else:
            # Return all groups for review
            for group in duplicate_groups:
                names = [e.get('entity_name', e.get('name', '')) for e in group]
                total_sim = 0
                count = 0
                
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        sim = EntityDeduplicator.calculate_similarity(names[i], names[j])
                        total_sim += sim
                        count += 1
                
                avg_similarity = total_sim / count if count > 0 else 0
                
                groups_for_review.append({
                    "entities": group,
                    "similarity": avg_similarity,
                    "names": names
                })
        
        # Update document entity count if entities were merged
        if merged_count > 0:
            # Get updated entity count
            count_response = supabase_service.client.table("entities")\
                .select("id", count="exact")\
                .eq("document_id", document_id)\
                .execute()
            
            new_count = count_response.count if count_response else 0
            
            # Update document
            supabase_service.client.table("documents")\
                .update({"entity_count": new_count})\
                .eq("id", document_id)\
                .execute()
        
        return {
            "duplicates": groups_for_review,
            "merged": merged_count,
            "total_groups": len(duplicate_groups),
            "message": f"Found {len(duplicate_groups)} duplicate groups, merged {merged_count} entities"
        }
        
    except Exception as e:
        logger.error(f"Error finding duplicates: {e}")
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
        
        # Find canonical ids with mentions in this doc
        client = supabase_service.client
        mentions = client.table("entity_mentions").select("canonical_entity_id,confidence,type").eq("document_id", document_id).not_.is_("canonical_entity_id", "null").execute()
        logger.info(f"Found {len(mentions.data) if mentions.data else 0} mentions for document {document_id}")
        ids = [m.get("canonical_entity_id") for m in (mentions.data or []) if m.get("canonical_entity_id")]
        logger.info(f"Extracted {len(ids)} canonical entity IDs")
        if not ids:
            return {"document_id": document_id, "total_entities": 0, "entities_by_type": {}, "all_entities": []}
        # Fetch canonical and filter by type if requested
        ce_query = client.table("canonical_entities").select("*").in_("id", list(set(ids)))
        if entity_type:
            ce_query = ce_query.eq("type", entity_type)
        ce_res = ce_query.execute()
        ces = ce_res.data or []
        # Build by-type mapping
        by_type = {}
        for ce in ces:
            t = ce.get("type", "Unknown")
            by_type.setdefault(t, []).append({
                "id": ce.get("id"),
                "name": ce.get("name"),
                "quality_score": ce.get("quality_score", 0.0),
                "is_validated": ce.get("is_validated", False),
                "metadata": ce.get("metadata", {}),
            })
        return {
            "document_id": document_id,
            "total_entities": len(ces),
            "entities_by_type": by_type,
            "all_entities": [
                {
                    "id": ce.get("id"),
                    "name": ce.get("name"),
                    "type": ce.get("type"),
                    "quality_score": ce.get("quality_score", 0.0),
                    "is_validated": ce.get("is_validated", False),
                    "metadata": ce.get("metadata", {})
                }
                for ce in ces
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entities for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/by-document/{document_id}")
async def list_entities_by_document(
    document_id: str,
    entity_type: Optional[str] = Query(None, description="Filter by canonical entity type"),
    min_quality: float = Query(0.0, ge=0.0, le=1.0, description="Minimum canonical quality score")
):
    """
    Explicit endpoint to list canonical entities for a document by joining mentions → canonical.
    """
    try:
        client = supabase_service.client
        mention_res = client.table("entity_mentions").select("canonical_entity_id").eq("document_id", document_id).neq("canonical_entity_id", None).execute()
        ids = [m.get("canonical_entity_id") for m in (mention_res.data or []) if m.get("canonical_entity_id")]
        if not ids:
            return []
        q = client.table("canonical_entities").select("*").in_("id", list(set(ids)))
        if entity_type:
            q = q.eq("type", entity_type)
        if min_quality > 0:
            q = q.gte("quality_score", min_quality)
        cres = q.execute()
        return [{
            "id": ce.get("id"),
            "name": ce.get("name"),
            "type": ce.get("type"),
            "quality_score": ce.get("quality_score", 0.0),
            "is_validated": ce.get("is_validated", False),
            "metadata": ce.get("metadata", {}),
            "document_id": document_id,
        } for ce in (cres.data or [])]
    except Exception as e:
        logger.error(f"Error listing entities for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
