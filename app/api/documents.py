"""
Document Management API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
import logging
import asyncio
import json

from app.services.supabase_service import SupabaseService
from app.models.job import ProcessingJob, JobType, JobStatus
from app.models.document import Document, DocumentState
from app.config import settings
from app.flows.entity_extraction_runner_v2 import run_extract_mentions, ChunkInput
from app.models.entity_v2 import EntityMention, CanonicalEntity
from pydantic import BaseModel, Field

from app.api.schemas import (
    DocumentListItem,
    DocumentDetail,
    DocumentUpdateRequest,
    ExtractMetadataResponse,
    DocumentProcessResponse,
    ChunkOut,
    ErrorResponse,
    OperationResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# IMPORTANT: Specific routes must come before parameterized routes
# This test endpoint must be defined before /{document_id}


class ProcessDocumentRequest(BaseModel):
    force_reprocess: bool = False


class DocumentRelationshipCreate(BaseModel):
    source_entity_id: str = Field(..., description="Canonical entity id for the source")
    target_entity_id: str = Field(..., description="Canonical entity id for the target")
    relationship_type: str = Field(..., description="Relationship label")
    confidence_score: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

@router.get(
    "/",
    response_model=List[DocumentListItem],
    summary="List documents",
    description="List documents with optional filters for status and source. Returns counts for chunks and entities.",
)
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),  # Fixed: Changed from DocumentStatus to str
    source: Optional[str] = Query(None, description="Filter by source type"),
    limit: int = Query(100, ge=1, le=500, description="Maximum documents to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all documents with optional filtering - using Supabase directly
    """
    try:
        # Use Supabase service with singleton client
        supabase_service = SupabaseService()
        
        # Get documents from Supabase
        documents = supabase_service.list_documents(status=status, limit=limit)
        logger.debug(f"Retrieved {len(documents)} documents from Supabase")
        
        # Filter by source if provided
        if source:
            documents = [doc for doc in documents if doc.source_type == source]
        
        # Apply offset manually (Supabase service doesn't support offset yet)
        if offset > 0:
            documents = documents[offset:]
        
        # Convert to dict format expected by frontend
        result = []
        for doc in documents:
            # Get chunk count for this document
            chunk_count = 0
            try:
                # Direct count from chunks table (avoid missing helper)
                cnt = supabase_service.client.table("chunks").select("id", count="exact").eq("document_id", str(doc.id)).execute()
                chunk_count = int(getattr(cnt, 'count', 0) or 0)
            except Exception as e:
                logger.warning(f"Failed to get chunk count for document {doc.id}: {e}")
            
            # Get entity count for this document
            entity_count = 0
            try:
                # Count ALL entity mentions for this document (not just canonicalized ones)
                # This matches what the detail page shows
                res = supabase_service.client.table("entity_mentions").select("id", count="exact").eq("document_id", str(doc.id)).execute()
                entity_count = int(getattr(res, 'count', 0) or 0)

                # Log if many mentions lack canonical assignment (for monitoring)
                if entity_count > 0:
                    canonical_res = supabase_service.client.table("entity_mentions").select("canonical_entity_id").eq("document_id", str(doc.id)).not_.is_("canonical_entity_id", "null").execute()
                    canonical_count = len({row.get("canonical_entity_id") for row in (canonical_res.data or []) if row.get("canonical_entity_id")})
                    if canonical_count < entity_count * 0.5:  # Less than 50% canonicalized
                        logger.info(f"Document {doc.id}: {entity_count} mentions but only {canonical_count} canonicalized")
            except Exception as e:
                logger.warning(f"Failed to get entity count for document {doc.id}: {e}")
            
            # Check metadata completeness
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            metadata_complete = False
            if metadata:
                # Check if key metadata fields are present and non-empty
                required_fields = ['title', 'author', 'summary', 'key_topics']
                metadata_complete = all(
                    field in metadata and metadata.get(field) and str(metadata.get(field)).strip()
                    for field in required_fields
                )
            
            doc_dict = {
                "id": str(doc.id),
                "title": doc.name,  # Map name to title for frontend compatibility
                "name": doc.name,
                "source_type": doc.source_type,
                "source_id": doc.source_id,
                "source_url": doc.source_url if hasattr(doc, 'source_url') else None,
                "status": doc.status.value if hasattr(doc.status, 'value') else doc.status,
                "metadata": metadata,
                "metadata_complete": metadata_complete,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "chunk_count": chunk_count,
                "entity_count": entity_count
            }
            result.append(doc_dict)
        
        logger.info(f"Listed {len(result)} documents with filters: status={status}, source={source}")
        return result
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        # Return empty list instead of error to allow frontend to work
        return []


@router.get(
    "/{document_id}/progress",
    summary="Stream processing progress",
    description="Server-Sent Events stream with step-by-step processing progress for a document.",
)
async def stream_document_progress(document_id: str):
    """
    Stream real-time processing progress updates for a document via Server-Sent Events (SSE).
    This includes step-by-step progress with time elapsed for each step.
    """
    from app.services.progress_tracker import ProgressTracker
    
    # Queue for in-process progress updates (legacy/simple path); Celery emits via DB polling below
    progress_queue = asyncio.Queue()
    supabase_service = SupabaseService()
    
    # Identify latest job for this document (if any)
    try:
        jobs = supabase_service.get_document_jobs(document_id)
        current_job = jobs[0] if jobs else None
    except Exception:
        current_job = None
    
    async def generate():
        """Generate SSE events with processing progress updates"""
        try:
            # Send initial event
            yield f"data: {json.dumps({'event': 'connected', 'document_id': document_id})}\n\n"
            
            # Listen for progress updates (in-process) and always poll job status for Celery-driven runs
            timeout_counter = 0
            max_timeouts = 30  # 30 seconds without updates
            
            while timeout_counter < max_timeouts:
                try:
                    # Wait for progress update with timeout
                    progress_data = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    
                    # Send progress event
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    # Check if processing is complete
                    if progress_data.get('percentage') == 100 or progress_data.get('status') in ['success', 'failed']:
                        yield f"data: {json.dumps({'event': 'complete', 'final_status': progress_data.get('status', 'completed')})}\n\n"
                        break
                    
                    timeout_counter = 0  # Reset timeout counter
                    
                except asyncio.TimeoutError:
                    timeout_counter += 1
                    # Also poll job status and stream it (works across Celery processes)
                    try:
                        nonlocal_job = current_job
                        if not nonlocal_job:
                            jobs = supabase_service.get_document_jobs(document_id)
                            nonlocal_job = jobs[0] if jobs else None
                        if nonlocal_job:
                            job = supabase_service.get_job(nonlocal_job.id)
                            if job:
                                job_event = {
                                    "event": "job_update",
                                    "document_id": document_id,
                                    "job_id": job.id,
                                    "status": job.job_status.value if hasattr(job.job_status, 'value') else job.job_status,
                                    "progress": job.progress,
                                    "current_step": job.current_step,
                                }
                                yield f"data: {json.dumps(job_event)}\n\n"
                    except Exception:
                        pass
                    # Send heartbeat to keep connection alive periodically
                    if timeout_counter % 5 == 0:
                        yield f"data: {json.dumps({'event': 'heartbeat', 'document_id': document_id})}\n\n"
                    
            if timeout_counter >= max_timeouts:
                yield f"data: {json.dumps({'event': 'timeout', 'message': 'No updates received for 30 seconds'})}\n\n"
                
        except Exception as e:
            logger.error(f"Error in progress stream for document {document_id}: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    # Store the queue for this document so the processor can send updates
    tracker = ProgressTracker()
    tracker.register_queue(document_id, progress_queue)
    
    try:
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    finally:
        # Clean up queue when connection closes
        tracker.unregister_queue(document_id)


@router.get(
    "/{document_id}/chunks",
    response_model=List[ChunkOut],
    summary="Get document chunks",
    description="Fetch all chunks for a document. Optionally include surrounding content for each chunk.",
    responses={500: {"model": ErrorResponse}},
)
async def get_document_chunks(
    document_id: UUID,
    include_context: bool = Query(False, description="Include surrounding text context"),
    context_size: int = Query(200, description="Characters of context before/after")
):
    """
    Get all chunks for a document - using Supabase directly
    """
    try:
        # Use Supabase service instead of SQLAlchemy
        supabase_service = SupabaseService()
        
        # Get chunks from Supabase
        response = supabase_service.client.table("chunks").select("*").eq("document_id", str(document_id)).order("chunk_index").execute()
        chunks = response.data if response.data else []
        
        result = []
        document = None
        
        # Get document content if context is requested
        if include_context and chunks:
            doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
            document = doc_response.data if doc_response.data else None
        
        for chunk in chunks:
            chunk_data = {
                "id": str(chunk.get("id")),
                "document_id": str(chunk.get("document_id")),
                "chunk_number": chunk.get("chunk_index", 0),  # Map to expected field
                "chunk_index": chunk.get("chunk_index", 0),  # Include both for compatibility
                "chunk_text": chunk.get("chunk_text", ""),  # Keep consistent field name
                "chunk_size": chunk.get("chunk_size", 0),
                "chunk_level": chunk.get("chunk_level"),  # Include contextual retrieval fields
                "contextual_summary": chunk.get("contextual_summary"),
                "contextualized_text": chunk.get("contextualized_text"),
                "parent_chunk_id": chunk.get("parent_chunk_id"),
                "parent_context": chunk.get("parent_context"),
                "position_in_parent": chunk.get("position_in_parent"),
                "bm25_tokens": chunk.get("bm25_tokens"),
                "start_position": chunk.get("start_position", 0),
                "end_position": chunk.get("end_position", 0),
                "metadata": chunk.get("metadata") or {},
                "created_at": chunk.get("created_at")
            }
            
            # Add context if requested
            if include_context and document and document.get('content'):
                content = document.get('content', '')
                start_pos = chunk.get("start_position", 0)
                end_pos = chunk.get("end_position", 0)
                
                # Get text before chunk
                before_start = max(0, start_pos - context_size)
                before_text = content[before_start:start_pos]
                
                # Get text after chunk
                after_end = min(len(content), end_pos + context_size)
                after_text = content[end_pos:after_end]
                
                chunk_data["context"] = {
                    "before": before_text,
                    "after": after_text
                }
            
            result.append(chunk_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get a document",
    description="Fetch a document with its chunks and entities.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_document(
    document_id: UUID,
    include_chunks: bool = Query(True, description="Include document chunks"),
    include_entities: bool = Query(True, description="Include extracted entities")
):
    """
    Get a single document with its chunks and entities - using Supabase directly
    """
    try:
        # Use Supabase service instead of SQLAlchemy
        supabase_service = SupabaseService()
        
        # Get document from Supabase
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).execute()
        documents = doc_response.data if doc_response.data else []
        document = documents[0] if documents else None
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Build response
        result = {
            "id": str(document.get("id")),
            "title": document.get("name"),  # Map name to title for frontend
            "name": document.get("name"),
            "content": document.get("content", ""),  # Add content field
            "source_type": document.get("source_type"),
            "source_id": document.get("source_id"),
            "source_url": document.get("source_url"),
            "status": document.get("status", "unknown"),
            "metadata": document.get("metadata") or {},
            "mime_type": document.get("mime_type") or (document.get("metadata") or {}).get("mime_type"),
            "author": document.get("author") or (document.get("metadata") or {}).get("author"),
            "security_level": document.get("security_level"),
            "access_level": document.get("access_level"),
            "created_at": document.get("created_at"),
            "updated_at": document.get("updated_at"),
            "processed_at": document.get("processed_at"),
            "ingested_at": document.get("ingested_at")
        }
        
        # Add chunks if requested
        if include_chunks:
            chunks_response = supabase_service.client.table("chunks").select("*").eq(
                "document_id", str(document_id)
            ).order("chunk_index").execute()
            chunks = chunks_response.data if chunks_response.data else []
            
            result["chunks"] = [
                {
                    "id": str(chunk.get("id")),
                    "chunk_number": chunk.get("chunk_index", 0),  # Keep legacy compatibility
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "chunk_size": chunk.get("chunk_size", 0),
                    "start_position": chunk.get("start_position", 0),
                    "end_position": chunk.get("end_position", 0),
                    "metadata": chunk.get("metadata") or {}
                }
                for chunk in chunks
            ]
        
        # Add entities if requested
        if include_entities:
            entities_response = supabase_service.client.table("entity_mentions").select("*").eq(
                "document_id", str(document_id)
            ).execute()
            entities = entities_response.data if entities_response.data else []

            canonical_ids = sorted({
                str(entity.get("canonical_entity_id"))
                for entity in entities
                if entity.get("canonical_entity_id")
            })
            canonical_details: dict[str, dict] = {}
            if canonical_ids:
                canonical_resp = (
                    supabase_service.client.table("canonical_entities")
                    .select("id,name,type,metadata")
                    .in_("id", canonical_ids)
                    .execute()
                )
                canonical_details = {
                    str(row.get("id")): row
                    for row in (canonical_resp.data or [])
                    if row.get("id") is not None
                }

            enriched_entities = []
            for entity in entities:
                cid = str(entity.get("canonical_entity_id")) if entity.get("canonical_entity_id") else None
                canonical_info = canonical_details.get(cid) if cid else None
                base_metadata = entity.get("metadata") or {}
                attributes = entity.get("attributes") or base_metadata.get("attributes")
                metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}
                if attributes and "attributes" not in metadata:
                    metadata["attributes"] = attributes

                entity_type = None
                if canonical_info and canonical_info.get("type"):
                    entity_type = canonical_info.get("type")
                else:
                    entity_type = entity.get("entity_type") or entity.get("type")

                enriched_entities.append({
                    "id": str(entity.get("id")),
                    "entity_name": entity.get("entity_name") or entity.get("name"),
                    "entity_type": entity_type,
                    "confidence_score": entity.get("confidence_score", entity.get("confidence", 1.0)),
                    "metadata": metadata,
                    "canonical_entity_id": cid,
                    "canonical_name": canonical_info.get("name") if canonical_info else None,
                    "canonical_type": canonical_info.get("type") if canonical_info else None,
                    "canonical_metadata": canonical_info.get("metadata") if canonical_info else None,
                })

            result["entities"] = enriched_entities
        
        logger.info(f"Retrieved document {document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred")


@router.get(
    "/{document_id}/entities",
    summary="Get document entities (Supabase)",
    description="Fetch all entities for a document from Supabase in standardized shape.",
)
async def get_document_entities(document_id: UUID):
    try:
        supabase_service = SupabaseService()
        entities_response = supabase_service.client.table("entity_mentions").select("*").eq(
            "document_id", str(document_id)
        ).execute()
        entities = entities_response.data or []

        canonical_ids = sorted({
            str(entity.get("canonical_entity_id"))
            for entity in entities
            if entity.get("canonical_entity_id")
        })
        canonical_details: dict[str, dict] = {}
        if canonical_ids:
            canonical_resp = (
                supabase_service.client.table("canonical_entities")
                .select("id,name,type,metadata")
                .in_("id", canonical_ids)
                .execute()
            )
            canonical_details = {
                str(row.get("id")): row
                for row in (canonical_resp.data or [])
                if row.get("id") is not None
            }

        normalized = []
        for e in entities:
            cid = str(e.get("canonical_entity_id")) if e.get("canonical_entity_id") else None
            canonical_info = canonical_details.get(cid) if cid else None
            base_metadata = e.get("metadata") or {}
            attributes = e.get("attributes") or (base_metadata.get("attributes") if isinstance(base_metadata, dict) else None)
            metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}
            if attributes and "attributes" not in metadata:
                metadata["attributes"] = attributes

            entity_type = None
            if canonical_info and canonical_info.get("type"):
                entity_type = canonical_info.get("type")
            else:
                entity_type = e.get("entity_type") or e.get("type")

            item = {
                "id": str(e.get("id")),
                "document_id": str(e.get("document_id")),
                "entity_name": e.get("entity_name") or e.get("name") or e.get("text"),  # Also check 'text' field
                "entity_type": entity_type,
                "confidence_score": e.get("confidence_score", e.get("confidence", 1.0)),
                "metadata": metadata,
                "canonical_entity_id": cid,
                "canonical_name": canonical_info.get("name") if canonical_info else None,
                "canonical_type": canonical_info.get("type") if canonical_info else None,
                "canonical_metadata": canonical_info.get("metadata") if canonical_info else None,
            }
            # Add context if available (from entity_mentions table)
            if e.get("context"):
                item["context"] = e.get("context")
            # Legacy alias for frontend/tests expecting 'confidence'
            item["confidence"] = item["confidence_score"]
            normalized.append(item)

        return normalized
    except Exception as e:
        logger.error(f"Error getting entities for document {document_id}: {e}")
        return []


@router.get(
    "/{document_id}/relationship-proposals",
    summary="List relationship proposals",
    description="Fetch unverified relationship proposals for a document with optional type and confidence filters.",
)
async def get_relationship_proposals(
    document_id: UUID,
    type: Optional[str] = Query(None, description="Filter by relationship_type label"),
    min_conf: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence_score")
):
    try:
        supabase_service = SupabaseService()
        client = supabase_service.client
        # Get canonical ids present via mentions for this doc
        mentions = client.table("entity_mentions").select("canonical_entity_id").eq("document_id", str(document_id)).neq("canonical_entity_id", None).execute()
        ids = [m.get("canonical_entity_id") for m in (mentions.data or []) if m.get("canonical_entity_id")]
        if not ids:
            return []
        ent_ids = list(set(ids))

        # Fetch proposals where BOTH ends are in this doc's canonical ids and is not validated
        rels = client.table("canonical_relationships").select("*")\
            .in_("source_entity_id", ent_ids)\
            .in_("target_entity_id", ent_ids)\
            .eq("is_validated", False)\
            .execute()

        combined = rels.data or []

        # Optional filters
        if type:
            combined = [r for r in combined if str(r.get("relationship_type", "")).upper() == type.upper()]
        if min_conf:
            def _conf(x):
                v = x.get("confidence_score")
                try:
                    return float(v or 0.0)
                except Exception:
                    return 0.0
            combined = [r for r in combined if _conf(r) >= min_conf]

        return combined
    except Exception as e:
        logger.error(f"Error fetching relationship proposals for {document_id}: {e}")
        return []


@router.get(
    "/{document_id}/relationships",
    summary="List document relationships",
    description="Fetch canonical relationships associated with a document (both proposed and verified).",
)
async def get_document_relationships(document_id: UUID):
    supabase_service = SupabaseService()
    try:
        relationships = supabase_service.get_document_relationships(str(document_id))

        # Collect all unique entity IDs
        entity_ids = set()
        for rel in relationships:
            if rel.get("source_entity_id"):
                entity_ids.add(rel["source_entity_id"])
            if rel.get("target_entity_id"):
                entity_ids.add(rel["target_entity_id"])

        # Fetch canonical entities for these IDs
        entity_map = {}
        if entity_ids:
            try:
                logger.info(f"Fetching canonical entities for {len(entity_ids)} IDs")
                entities_response = supabase_service.client.table("canonical_entities").select("id, name, type").in_("id", list(entity_ids)).execute()
                logger.info(f"Got {len(entities_response.data or [])} canonical entities")
                for entity in (entities_response.data or []):
                    entity_map[entity["id"]] = entity
                    logger.debug(f"Mapped entity {entity['id'][:8]}... -> {entity['name']}")
            except Exception as e:
                logger.warning(f"Failed to fetch canonical entities for relationship enrichment: {e}")

        # Enrich relationships with entity names from canonical_entities
        for rel in relationships:
            metadata = rel.get("metadata", {}) or {}
            # Expose raw_relationship_type at the top level if it exists
            if "raw_relationship_type" in metadata:
                rel["raw_relationship_type"] = metadata["raw_relationship_type"]

            # Add entity names
            source_id = rel.get("source_entity_id")
            target_id = rel.get("target_entity_id")

            if source_id and source_id in entity_map:
                rel["source_entity_name"] = entity_map[source_id]["name"]
                rel["source_entity_type"] = entity_map[source_id]["type"]
            else:
                # Fallback to metadata if available
                rel["source_entity_name"] = metadata.get("source_name", None)

            if target_id and target_id in entity_map:
                rel["target_entity_name"] = entity_map[target_id]["name"]
                rel["target_entity_type"] = entity_map[target_id]["type"]
            else:
                # Fallback to metadata if available
                rel["target_entity_name"] = metadata.get("target_name", None)

        return relationships
    except Exception as exc:
        logger.error("Error fetching relationships for document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch document relationships")


@router.post(
    "/{document_id}/relationships",
    summary="Create document relationship",
    description="Create a canonical relationship tied to a document.",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_document_relationship(document_id: UUID, payload: DocumentRelationshipCreate):
    if payload.source_entity_id == payload.target_entity_id:
        raise HTTPException(status_code=400, detail="Source and target must differ")

    supabase_service = SupabaseService()

    try:
        record = supabase_service.create_document_relationship(
            str(document_id),
            source_entity_id=payload.source_entity_id,
            target_entity_id=payload.target_entity_id,
            relationship_type=payload.relationship_type,
            confidence_score=float(payload.confidence_score or 0.0),
            metadata=payload.metadata,
        )
        return record
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create relationship for document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to create relationship")


@router.delete(
    "/{document_id}",
    summary="Delete a document",
    description="Soft delete by default. Optionally perform a hard delete of the document and related data.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: UUID,
    hard_delete: bool = Query(False, description="Permanently delete (vs soft delete)")
):
    """
    Delete a document (soft delete by default) - using Supabase directly
    """
    try:
        supabase_service = SupabaseService()
        
        # Check if document exists
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        document = doc_response.data if doc_response.data else None
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        if hard_delete:
            # Delete related data first (foreign key constraints)
            supabase_service.client.table("chunks").delete().eq("document_id", str(document_id)).execute()
            supabase_service.client.table("entity_mentions").delete().eq("document_id", str(document_id)).execute()
            
            # Delete document
            supabase_service.client.table("documents").delete().eq("id", str(document_id)).execute()
            
            logger.info(f"Hard deleted document {document_id}")
            return {"message": f"Document {document_id} permanently deleted"}
        else:
            # Soft delete - just update status to REJECTED (since DELETED causes constraint violation)
            supabase_service.client.table("documents").update({
                "status": DocumentState.REJECTED.value,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", str(document_id)).execute()
            
            logger.info(f"Soft deleted document {document_id}")
            return {"message": f"Document {document_id} marked as deleted"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    reparse: bool = Query(False, description="Re-parse source document"),
    rechunk: bool = Query(True, description="Re-chunk document"),
    reextract: bool = Query(True, description="Re-extract entities"),
    chunk_size: Optional[int] = Query(None, description="Override chunk size"),
    chunk_overlap: Optional[int] = Query(None, description="Override chunk overlap")
):
    """
    Trigger reprocessing of a document - using Supabase directly
    """
    try:
        supabase_service = SupabaseService()
        
        # Check if document exists
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        document = doc_response.data if doc_response.data else None
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Create a ProcessingJob for reprocess
        from app.models.job import ProcessingJob, JobType, JobStatus
        job = ProcessingJob(document_id=str(document_id), job_type=JobType.FULL_PIPELINE)
        job = supabase_service.create_job(job)

        # Update status to processing
        supabase_service.update_document_status(str(document_id), DocumentState.PROCESSING)

        # Enqueue Celery processing (reuse full pipeline)
        try:
            from app.tasks.document_tasks import process_document
            try:
                async_result = process_document.apply_async(
                    args=[str(document_id), job.id],
                    kwargs={"force_reprocess": True},
                )
            except TypeError as exc_kwargs:
                if "force_reprocess" in str(exc_kwargs):
                    logger.warning(
                        "process_document task rejected force_reprocess kwarg during reprocess; falling back to positional args: %s",
                        exc_kwargs,
                    )
                    try:
                        async_result = process_document.apply_async(
                            args=[str(document_id), job.id, True],
                        )
                    except TypeError as exc_args:
                        logger.warning(
                            "process_document task rejected positional force_reprocess during reprocess; falling back to legacy invocation: %s",
                            exc_args,
                        )
                        async_result = process_document.apply_async(
                            args=[str(document_id), job.id]
                        )
                else:
                    raise

            supabase_service.update_job(job.id, {"celery_task_id": async_result.id, "job_status": "running", "current_step": "Queued for reprocessing", "progress": 1})
            logger.info(f"Queued reprocessing via Celery for document {document_id} (job {job.id}, task {async_result.id})")
        except Exception as e:
            logger.error(f"Failed to queue Celery reprocess task: {e}")
            supabase_service.update_job(job.id, {"job_status": "failed", "error_message": str(e)})
            supabase_service.update_document_status(str(document_id), DocumentState.FAILED, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to queue reprocessing task")

        return {
            "message": f"Document {document_id} queued for reprocessing",
            "job_id": job.id,
            "celery_task_id": async_result.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{document_id}",
    response_model=OperationResponse,
    summary="Update a document",
    description="Update document fields and/or merge metadata. Use this to set status, author, or metadata fields.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_document(
    document_id: UUID,
    update_data: DocumentUpdateRequest
):
    """
    Update document fields including metadata - using Supabase directly
    """
    try:
        supabase_service = SupabaseService()
        
        # Get existing document
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        document = doc_response.data if doc_response.data else None
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle metadata specifically - always merge unless explicitly set to replace
        payload = update_data.dict(exclude_unset=True)

        if "metadata" in payload:
            existing_metadata = document.get("metadata", {}) or {}
            new_metadata = payload["metadata"] or {}
            
            # Extract special fields that may be in metadata
            if "author" in new_metadata:
                updates["author"] = new_metadata.pop("author")
            if "mime_type" in new_metadata:
                updates["mime_type"] = new_metadata.pop("mime_type")
            if "security_level" in new_metadata:
                updates["security_level"] = new_metadata.pop("security_level")
            if "access_level" in new_metadata:
                updates["access_level"] = new_metadata.pop("access_level")

            # Merge remaining metadata
            updates["metadata"] = {**existing_metadata, **new_metadata}
        
        # Handle other direct fields
        for field in ["title", "name", "author", "mime_type", "status", "security_level", "access_level"]:
            if field in payload:
                updates[field] = payload[field]

        # Map title to name for storage if provided
        if "title" in payload and "name" not in updates:
            updates["name"] = payload["title"]

        # If security_level is updated, also update access_level
        if "security_level" in updates:
            security_mapping = {
                "public": 1,
                "client": 2,
                "partner": 3,
                "employee": 4,
                "management": 5
            }
            updates["access_level"] = security_mapping.get(updates["security_level"], 1)

        updates["updated_at"] = datetime.utcnow().isoformat()
        
        # Update document
        supabase_service.client.table("documents").update(updates).eq("id", str(document_id)).execute()
        
        logger.info(f"Updated document {document_id} with fields: {list(updates.keys())}")
        return {"message": f"Document {document_id} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{document_id}/metadata",
    response_model=OperationResponse,
    summary="Update document metadata",
    description="Convenience endpoint to update document metadata only.",
)
async def update_metadata(
    document_id: UUID,
    metadata: dict,
    merge: bool = Query(True, description="Merge with existing metadata (vs replace)")
):
    """
    Update document metadata - using Supabase directly
    """
    # This is a convenience endpoint that delegates to the main update endpoint
    return await update_document(document_id, {"metadata": metadata})


@router.post(
    "/{document_id}/extract-metadata",
    response_model=ExtractMetadataResponse,
    summary="Extract metadata",
    description="Run AI metadata extraction for a document (async background).",
)
async def extract_metadata(
    document_id: UUID,
    background_tasks: BackgroundTasks
):
    """
    Extract metadata from document content using AI
    """
    logger.info(f"Metadata extraction requested for document {document_id}")
    
    try:
        # Verify document exists
        supabase_service = SupabaseService()
        doc_response = supabase_service.client.table("documents").select("*").eq(
            "id", str(document_id)
        ).single().execute()
        
        if not doc_response.data:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        document = doc_response.data
        
        # Check if document has content
        if not document.get("content"):
            raise HTTPException(
                status_code=400,
                detail="Document has no content for metadata extraction"
            )
        
        # Queue background extraction task
        background_tasks.add_task(
            extract_metadata_task,
            document_id=str(document_id)
        )
        
        return {
            "message": f"Metadata extraction started for document {document_id}",
            "document_id": str(document_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting metadata extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{document_id}/suggested-metadata",
    summary="Get suggested metadata",
    description="Retrieve previously AI-extracted metadata suggestions for a document.",
)
async def get_suggested_metadata(document_id: UUID):
    """
    Get AI-suggested metadata for a document
    """
    try:
        supabase_service = SupabaseService()
        
        # Get document
        doc_response = supabase_service.client.table("documents").select("*").eq(
            "id", str(document_id)
        ).single().execute()
        
        if not doc_response.data:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        document = doc_response.data
        metadata = document.get("metadata", {}) or {}
        
        # Check if we have AI-extracted metadata
        if metadata.get("ai_extracted"):
            return {
                "document_id": str(document_id),
                "suggestions": {
                    "category": metadata.get("category"),
                    "tags": metadata.get("tags", []),
                    "author": metadata.get("author"),
                    "department": metadata.get("department"),
                    "version": metadata.get("version"),
                    "description": metadata.get("description")
                },
                "confidence_scores": metadata.get("confidence_scores", {}),
                "extraction_timestamp": metadata.get("extraction_timestamp")
            }
        else:
            return {
                "document_id": str(document_id),
                "suggestions": None,
                "message": "No AI-extracted metadata available. Use /extract-metadata endpoint first."
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting suggested metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessResponse,
    summary="Trigger document processing",
    description="Start processing for a document: chunking, embeddings, and entity extraction with progress streaming available.",
)
async def trigger_document_processing(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    request: ProcessDocumentRequest = Body(
        default=ProcessDocumentRequest(),
        description="Processing options including force reprocess flag",
    ),
):
    """
    Trigger processing for a specific document.
    This will chunk the document, extract entities, and generate embeddings.
    """
    force_reprocess = request.force_reprocess if request else False
    logger.info(
        "Processing endpoint called for %s force_reprocess=%s request=%s",
        document_id,
        force_reprocess,
        request.dict() if request else None,
    )
    try:
        # Verify document exists using Supabase
        supabase_service = SupabaseService()
        document = supabase_service.get_document(str(document_id))
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Check if document is in a state that can be processed
        if force_reprocess:
            # Allow reprocessing from any state except processing
            if document.status == "processing":
                raise HTTPException(
                    status_code=400,
                    detail="Document is currently being processed. Please wait for it to complete."
                )

            logger.info(f"Force reprocessing: Cleaning up existing data for document {document_id}")
            try:
                supabase_service.delete_document_chunks(str(document_id))
            except Exception as cleanup_err:
                logger.warning(f"Error deleting chunks: {cleanup_err}")

            try:
                deleted_mentions = supabase_service.delete_document_entity_mentions(str(document_id))
                if deleted_mentions:
                    logger.info(f"Removed {deleted_mentions} existing entity mentions for document {document_id}")
            except Exception as cleanup_err:
                logger.warning(f"Error deleting entity mentions: {cleanup_err}")

            try:
                deleted_relationships = supabase_service.delete_document_relationships(str(document_id))
                if deleted_relationships:
                    logger.info(f"Removed {deleted_relationships} canonical relationships for document {document_id}")
            except Exception as cleanup_err:
                logger.warning(f"Error deleting relationships: {cleanup_err}")
        else:
            # Normal processing - only allow from discovered or failed states
            valid_states = ["discovered", "failed"]
            if document.status not in valid_states:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document cannot be processed from status: {document.status}. Use force_reprocess=true to reprocess."
                )


        # Create a ProcessingJob row in Supabase
        job = ProcessingJob(document_id=str(document_id), job_type=JobType.FULL_PIPELINE)
        job = supabase_service.create_job(job)

        # Update document status to processing
        supabase_service.update_document_status(str(document_id), DocumentState.PROCESSING)

        # Enqueue Celery pipeline (preferred path)
        try:
            from app.tasks.document_tasks import process_document
            try:
                async_result = process_document.apply_async(
                    args=[str(document_id), job.id],
                    kwargs={"force_reprocess": force_reprocess},
                )
            except TypeError as exc_kwargs:
                if "force_reprocess" in str(exc_kwargs):
                    logger.warning(
                        "process_document task rejected force_reprocess kwarg; falling back to positional args: %s",
                        exc_kwargs,
                    )
                    try:
                        async_result = process_document.apply_async(
                            args=[str(document_id), job.id, force_reprocess],
                        )
                    except TypeError as exc_args:
                        logger.warning(
                            "process_document task rejected positional force_reprocess; falling back to legacy invocation: %s",
                            exc_args,
                        )
                        async_result = process_document.apply_async(
                            args=[str(document_id), job.id]
                        )
                else:
                    raise

            # Store Celery task id on the job
            supabase_service.update_job(job.id, {"celery_task_id": async_result.id, "job_status": "running", "progress": 1, "current_step": "Queued for processing"})
            logger.info(f"Queued Celery processing for document {document_id} (job {job.id}, task {async_result.id})")
        except Exception as e:
            logger.error(f"Failed to queue Celery task: {e}")
            # Fail the job and revert document status
            supabase_service.update_job(job.id, {"job_status": "failed", "error_message": str(e)})
            supabase_service.update_document_status(str(document_id), DocumentState.FAILED, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to queue processing task")

        return {
            "document_id": str(document_id),
            "status": "processing",
            "job_id": job.id,
            "celery_task_id": async_result.id,
            "message": "Document queued for processing via Celery"
        }
        
    except HTTPException as e:
        logger.error(f"HTTPException in process endpoint: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error triggering document processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


## ---------------------------------------------
## LEGACY (commented out): process_document_pipeline
## The simple CocoIndex-based pipeline has been removed in favor of Celery.
## Keeping a placeholder here for reference without executable code.
## async def process_document_pipeline(document_id: str, document_dict: dict):
##     pass


## ---------------------------------------------
## LEGACY (commented out): process_document_pipeline_old
## Historical manual pipeline, no longer in use.
## async def process_document_pipeline_old(document_id: str, document_dict: dict):
##     pass


async def extract_metadata_task(document_id: str):
    """
    Background task to extract metadata from a document
    """
    logger.info(f"Starting metadata extraction task for document {document_id}")
    
    try:
        from app.services.metadata_extraction import extract_metadata_for_document
        
        # Run the metadata extraction
        result = await extract_metadata_for_document(document_id)
        
        if result.get("success"):
            logger.info(f"Successfully extracted metadata for document {document_id}")
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"Failed to extract metadata for document {document_id}: {error}")
            
    except Exception as e:
        logger.error(f"Error in metadata extraction task: {e}", exc_info=True)


async def reprocess_document_task(
    document_id: str,
    reparse: bool,
    rechunk: bool,
    reextract: bool,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int]
):
    """
    DEPRECATED: Background task for reprocessing. Use Celery process_document instead.
    """
    logger.warning("reprocess_document_task is DEPRECATED; forwarding to Celery pipeline.")
    logger.info(f"=== REPROCESS TASK STARTED for document {document_id} ===")
    logger.info(f"Options: reparse={reparse}, rechunk={rechunk}, reextract={reextract}")
    
    try:
        supabase_service = SupabaseService()
        
        # Get the document
        doc_response = supabase_service.client.table("documents").select("*").eq("id", document_id).single().execute()
        document = doc_response.data if doc_response.data else None
        
        if not document:
            logger.error(f"Document {document_id} not found")
            return
        
        # Update status to processing
        supabase_service.client.table("documents").update({
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", document_id).execute()
        
        logger.info(f"Document {document_id} status updated to 'processing'")
        
        # Convert document dict to match expected format
        document_dict = {
            "id": document["id"],
            "source_type": document.get("source_type", ""),
            "source_id": document.get("source_id", ""),
            "title": document.get("name", "Untitled"),
            "name": document.get("name", "Untitled"),
            "content": document.get("content", ""),
            "metadata": document.get("metadata", {}),
            "doc_metadata": document.get("doc_metadata", {})
        }
        
        # Queue Celery full pipeline instead of calling the simple flow
        from app.tasks.document_tasks import process_document
        from app.models.job import ProcessingJob, JobType, JobStatus
        job = ProcessingJob(document_id=document_id, job_type=JobType.FULL_PIPELINE)
        job = supabase_service.create_job(job)
        async_result = process_document.apply_async(
            args=[document_id, job.id],
            kwargs={"force_reprocess": True},
        )
        supabase_service.update_job(job.id, {"celery_task_id": async_result.id, "job_status": "running", "current_step": "Queued for reprocessing", "progress": 1})
        logger.info(f"Queued Celery reprocess task for {document_id} (job {job.id})")
        return {"status": "queued", "job_id": job.id, "celery_task_id": async_result.id}
        
        if False and result and result.get("status") == "success":
            logger.info(f"Successfully reprocessed document {document_id}")
            logger.info(f"Chunks: {result.get('chunks_created', 0)}, Entities: {result.get('entities_extracted', 0)}")

            # Run v2 entity extraction if enabled
            if settings.entity_pipeline_version == "v2":
                try:
                    logger.info(f"Running v2 entity extraction for {document_id}")
                    # Fetch chunks
                    chunks = supabase_service.get_document_chunks(document_id)
                    inputs = [
                        ChunkInput(id=c.id, document_id=c.document_id, text=c.chunk_text)
                        for c in chunks
                    ]
                    # Create extraction run
                    run_id = supabase_service.create_extraction_run(
                        document_id=document_id,
                        pipeline_version="v2",
                        model="gpt-4o-mini",
                    )

                    # Extract mentions
                    mentions_per_chunk = run_extract_mentions(inputs)
                    mentions: list[EntityMention] = []
                    for chunk, mlist in zip(chunks, mentions_per_chunk):
                        if not mlist:
                            continue
                        for m in mlist:
                            m.document_id = document_id
                            m.chunk_id = chunk.id
                            mentions.append(m)

                    # Canonicalize naively by (name,type)
                    seen: set[tuple[str, str]] = set()
                    canonicals: list[CanonicalEntity] = []
                    for m in mentions:
                        key = (m.text.strip().lower(), m.type)
                        if key not in seen:
                            seen.add(key)
                            canonicals.append(CanonicalEntity(name=m.text.strip(), type=m.type))

                    # Upsert and map ids
                    canonical_map = supabase_service.upsert_canonical_entities_map(canonicals)
                    for m in mentions:
                        key = (m.text.strip().lower(), m.type)
                        m.canonical_entity_id = canonical_map.get(key)

                    # Insert mentions and complete run
                    inserted = supabase_service.insert_entity_mentions(mentions, extraction_run_id=run_id)
                    supabase_service.complete_extraction_run(
                        run_id,
                        mentions=len(mentions),
                        canonical=len(canonical_map),
                        relationships=0,
                    )
                    logger.info(f"v2 extraction completed: mentions={len(mentions)}, canonical={len(canonical_map)}")
                except Exception as ve:
                    logger.error(f"v2 entity extraction failed for {document_id}: {ve}")
            
            # Update document status to pending_review
            supabase_service.client.table("documents").update({
                "status": "pending_review",  # Changed from ingested to require review
                "chunk_count": result.get("chunks_created", 0),
                "entity_count": result.get("entities_extracted", 0),
                "updated_at": datetime.utcnow().isoformat(),
                "processed_at": datetime.utcnow().isoformat()  # Changed from ingested_at to processed_at
            }).eq("id", document_id).execute()
        else:
            error_msg = result.get("error", "Unknown error") if result else "Processing returned no result"
            logger.error(f"Failed to reprocess document {document_id}: {error_msg}")
            
            # Update status to failed
            supabase_service.client.table("documents").update({
                "status": "failed",
                "updated_at": datetime.utcnow().isoformat(),
                "metadata": {**document.get("metadata", {}), "processing_error": error_msg}
            }).eq("id", document_id).execute()
            
    except Exception as e:
        logger.error(f"Error in reprocess task for document {document_id}: {e}", exc_info=True)
        
        # Update status to failed
        try:
            supabase_service = SupabaseService()
            supabase_service.client.table("documents").update({
                "status": "failed",
                "updated_at": datetime.utcnow().isoformat(),
                "metadata": {**document.get("metadata", {}), "processing_error": str(e)}
            }).eq("id", document_id).execute()
        except:
            pass

# Force reload
