"""
Document Management API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional
from datetime import datetime
from uuid import UUID
import logging

from app.services.supabase_service import SupabaseService
from app.services.document_processor import DocumentProcessor
from app.models.document import Document, DocumentState
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# IMPORTANT: Specific routes must come before parameterized routes
# This test endpoint must be defined before /{document_id}

@router.get("/", response_model=List[dict])
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),  # Fixed: Changed from DocumentStatus to str
    source: Optional[str] = Query(None, description="Filter by source type"),
    limit: int = Query(100, ge=1, le=500, description="Maximum documents to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all documents with optional filtering - using Supabase directly
    """
    logger.info(f"=== LIST DOCUMENTS ENDPOINT CALLED ===")
    logger.info(f"Parameters: status={status}, source={source}, limit={limit}, offset={offset}")
    
    try:
        logger.info(f"Creating SupabaseService instance...")
        # Use Supabase service instead of SQLAlchemy
        supabase_service = SupabaseService()
        logger.info(f"SupabaseService instance created successfully")
        
        # Get documents from Supabase
        logger.info(f"Calling list_documents on SupabaseService...")
        documents = supabase_service.list_documents(status=status, limit=limit)
        logger.info(f"Retrieved {len(documents)} documents from Supabase")
        
        # Filter by source if provided
        if source:
            documents = [doc for doc in documents if doc.source_type == source]
        
        # Apply offset manually (Supabase service doesn't support offset yet)
        if offset > 0:
            documents = documents[offset:]
        
        # Convert to dict format expected by frontend
        result = []
        for doc in documents:
            doc_dict = {
                "id": str(doc.id),
                "title": doc.name,  # Map name to title for frontend compatibility
                "name": doc.name,
                "source_type": doc.source_type,
                "source_id": doc.source_id,
                "source_url": doc.source_url if hasattr(doc, 'source_url') else None,
                "status": doc.status.value if hasattr(doc.status, 'value') else doc.status,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "chunk_count": 0,  # TODO: Fix chunk count query
                "entity_count": 0  # TODO: Fix entity count query
            }
            result.append(doc_dict)
        
        logger.info(f"Listed {len(result)} documents with filters: status={status}, source={source}")
        return result
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        # Return empty list instead of error to allow frontend to work
        return []


@router.get("/{document_id}/chunks", response_model=list)
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


@router.get("/{document_id}", response_model=dict)
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
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        document = doc_response.data if doc_response.data else None
        
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
                    "chunk_number": chunk.get("chunk_index", 0),  # Map to frontend expected field
                    "text": chunk.get("chunk_text", ""),
                    "size": chunk.get("chunk_size", 0),
                    "start_position": chunk.get("start_position", 0),
                    "end_position": chunk.get("end_position", 0),
                    "metadata": chunk.get("metadata") or {}
                }
                for chunk in chunks
            ]
        
        # Add entities if requested
        if include_entities:
            entities_response = supabase_service.client.table("entities").select("*").eq(
                "document_id", str(document_id)
            ).execute()
            entities = entities_response.data if entities_response.data else []
            
            result["entities"] = [
                {
                    "id": str(entity.get("id")),
                    "name": entity.get("name"),
                    "type": entity.get("type"),
                    "confidence": entity.get("confidence", 1.0),
                    "metadata": entity.get("metadata") or {}
                }
                for entity in entities
            ]
        
        logger.info(f"Retrieved document {document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred")


@router.delete("/{document_id}")
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
            supabase_service.client.table("entities").delete().eq("document_id", str(document_id)).execute()
            
            # Delete document
            supabase_service.client.table("documents").delete().eq("id", str(document_id)).execute()
            
            logger.info(f"Hard deleted document {document_id}")
            return {"message": f"Document {document_id} permanently deleted"}
        else:
            # Soft delete - just update status
            supabase_service.client.table("documents").update({
                "status": "deleted",
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
        
        # Update status
        supabase_service.client.table("documents").update({
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", str(document_id)).execute()
        
        # Queue background processing
        background_tasks.add_task(
            reprocess_document_task,
            document_id=str(document_id),
            reparse=reparse,
            rechunk=rechunk,
            reextract=reextract,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Queued reprocessing for document {document_id}")
        return {
            "message": f"Document {document_id} queued for reprocessing",
            "job_id": f"reprocess_{document_id}_{datetime.utcnow().timestamp()}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}")
async def update_document(
    document_id: UUID,
    update_data: dict
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
        if "metadata" in update_data:
            existing_metadata = document.get("metadata", {}) or {}
            new_metadata = update_data["metadata"]
            
            # Extract special fields that may be in metadata
            if "author" in new_metadata:
                updates["author"] = new_metadata.pop("author")
            if "mime_type" in new_metadata:
                updates["mime_type"] = new_metadata.pop("mime_type")
            
            # Merge remaining metadata
            updates["metadata"] = {**existing_metadata, **new_metadata}
        
        # Handle other direct fields
        for field in ["title", "name", "author", "mime_type", "status"]:
            if field in update_data:
                updates[field] = update_data[field]
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        # Update document
        supabase_service.client.table("documents").update(updates).eq("id", str(document_id)).execute()
        
        logger.info(f"Updated document {document_id} with fields: {list(updates.keys())}")
        return {
            "message": f"Document {document_id} updated successfully",
            "updated_fields": list(updates.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}/metadata")
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


@router.post("/{document_id}/extract-metadata")
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


@router.get("/{document_id}/suggested-metadata")
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


@router.post("/{document_id}/process")
async def trigger_document_processing(
    document_id: UUID,
    background_tasks: BackgroundTasks
):
    """
    Trigger processing for a specific document.
    This will chunk the document, extract entities, and generate embeddings.
    """
    logger.info(f"Processing endpoint called for document: {document_id}")
    try:
        # Verify document exists using Supabase
        supabase_service = SupabaseService()
        document = supabase_service.get_document(str(document_id))
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Check if document is in a state that can be processed
        valid_states = ["discovered", "failed"]
        if document.status not in valid_states:
            raise HTTPException(
                status_code=400, 
                detail=f"Document cannot be processed from status: {document.status}"
            )
        
        # Update status to processing
        supabase_service.update_document(
            str(document_id),
            {"status": "processing"}
        )
        
        # Queue the processing task - pass document as dict for serialization
        background_tasks.add_task(
            process_document_pipeline,
            document_id=str(document_id),
            document_dict=document.dict() if hasattr(document, 'dict') else document.__dict__
        )
        
        logger.info(f"Queued processing for document {document_id}")
        
        return {
            "document_id": str(document_id),
            "status": "processing",
            "message": f"Document processing has been started"
        }
        
    except HTTPException as e:
        logger.error(f"HTTPException in process endpoint: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error triggering document processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_pipeline(document_id: str, document_dict: dict):
    """
    Process a document through the CocoIndex pipeline
    Now using the full CocoIndex flow for incremental processing and multi-DB sync
    """
    logger.info(f"=== PROCESS_DOCUMENT_PIPELINE CALLED for {document_id} ===")
    logger.info(f"Using CocoIndex flow for document processing")
    
    try:
        # Import the CocoIndex flow
        from app.flows.document_processor import create_and_run_flow
        
        # Run the CocoIndex flow
        result = await create_and_run_flow(document_id)
        
        if result["status"] == "success":
            logger.info(f"Successfully processed document {document_id} with CocoIndex")
            logger.info(f"Created {result['chunks_created']} chunks and {result['entities_extracted']} entities")
        else:
            logger.error(f"Failed to process document {document_id}: {result.get('error', 'Unknown error')}")
            # The flow already updates the document status to FAILED
        
        return result
        
    except Exception as e:
        logger.error(f"Error in CocoIndex pipeline: {e}", exc_info=True)
        
        # Update document status to failed
        try:
            supabase_service = SupabaseService()
            supabase_service.update_document(
                document_id,
                {"status": "failed", "metadata": {"error": str(e)}}
            )
        except:
            pass
        
        raise


async def process_document_pipeline_old(document_id: str, document_dict: dict):
    """
    DEPRECATED: Old manual processing pipeline - kept for reference
    This is replaced by the CocoIndex flow above
    """
    logger.info(f"=== OLD PROCESS_DOCUMENT_PIPELINE CALLED for {document_id} ===")
    try:
        supabase_service = SupabaseService()
        from app.processors.parser import DocumentParser
        from app.processors.chunker import DocumentChunker
        from app.processors.entity_extractor import EntityExtractor
        
        logger.info(f"Processing document {document_id}: Starting pipeline...")
        
        # Step 1: Extract content using LlamaParse
        source_type = document_dict.get('source_type', '')
        source_id = document_dict.get('source_id', '')
        title = document_dict.get('title', 'Untitled')
        
        content = None
        
        # Check if document already has content (from scan)
        if document_dict.get('content'):
            content = document_dict['content']
            logger.info(f"Using existing content for document {document_id}")
        else:
            # Need to fetch content based on source type
            if source_type == 'notion':
                # For Notion, content should have been fetched during scan
                logger.warning(f"Notion document {document_id} has no content - may need to re-scan")
                content = f"# {title}\n\nContent needs to be fetched from Notion."
            elif source_type == 'google_drive':
                # For Google Drive, we need to download and parse the file
                logger.info(f"Fetching Google Drive document for parsing: {source_id}")
                try:
                    # TODO: Download file from Google Drive and parse with LlamaParse
                    # For now, use placeholder
                    parser = DocumentParser()
                    # This will return test content since file doesn't exist locally
                    parse_result = await parser.parse_document(
                        file_path=f"/tmp/{source_id}",  # Placeholder path
                        tier="balanced"
                    )
                    if parse_result["success"]:
                        content = parse_result["content"]
                        logger.info(f"Successfully parsed document {document_id}")
                    else:
                        content = f"# {title}\n\nFailed to parse document: {parse_result.get('error', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Error parsing document: {e}")
                    content = f"# {title}\n\nError parsing document: {str(e)}"
            else:
                content = f"# {title}\n\nUnknown source type: {source_type}"
        
        # Step 2: Chunk the document
        logger.info(f"Chunking document {document_id}...")
        chunker = DocumentChunker()
        from app.models.chunk import ChunkingStrategy
        chunks = chunker.chunk_text(
            text=content,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=1500,
            chunk_overlap=200
        )
        
        # Save chunks to database
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "document_id": document_id,
                "chunk_index": i + 1,  # Changed from chunk_number to chunk_index
                "chunk_text": chunk["text"],
                "chunk_size": len(chunk["text"]),
                "chunking_strategy": "recursive",  # Added required field
                "metadata": chunk.get("metadata", {})
            }
            supabase_service.client.table("chunks").insert(chunk_data).execute()
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        
        # Step 3: Extract entities
        logger.info(f"Extracting entities from document {document_id}...")
        extractor = EntityExtractor()
        from app.models.chunk import Chunk
        # Convert chunks to Chunk objects for entity extractor
        chunk_objects = [
            Chunk(
                id=f"{document_id}_chunk_{i+1}",
                document_id=document_id,
                chunk_index=i+1,  # Changed from chunk_number to chunk_index
                chunk_text=chunk["text"],
                chunk_size=len(chunk["text"]),
                chunking_strategy="recursive",
                metadata=chunk.get("metadata", {})
            )
            for i, chunk in enumerate(chunks)
        ]
        entities, relationships = extractor.extract(chunk_objects, document_id)
        
        # Save entities to database
        for entity in entities:
            entity_data = {
                "document_id": document_id,
                "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                "entity_name": entity.name,
                "confidence_score": entity.confidence,  # Changed from confidence to confidence_score
                "metadata": entity.metadata or {}
            }
            supabase_service.client.table("entities").insert(entity_data).execute()
        
        logger.info(f"Extracted {len(entities)} entities from document {document_id}")
        
        # Step 4: Generate embeddings (TODO)
        # embeddings = generate_embeddings(chunks)
        
        # Step 5: Store in vector DB (TODO)
        # store_in_qdrant(chunks, embeddings)
        
        # Step 6: Store entities in Neo4j (TODO)
        # store_in_neo4j(entities)
        
        # Update status to pending_review
        update_data = {
            "status": "pending_review",
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store content if it was extracted
        if content and not document_dict.get('content'):
            update_data["content"] = content[:10000]  # Store first 10k chars for preview
        
        supabase_service.update_document(document_id, update_data)
        
        logger.info(f"Document {document_id} processing complete, pending review")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        # Update status to failed
        supabase_service = SupabaseService()
        supabase_service.update_document(
            document_id,
            {"status": "failed", "processing_error": str(e)}
        )


async def extract_metadata_task(document_id: str):
    """
    Background task to extract metadata from a document
    """
    logger.info(f"Starting metadata extraction task for document {document_id}")
    
    try:
        from app.flows.metadata_extraction_flow import extract_metadata_for_document
        
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
    Background task to reprocess a document
    """
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
        
        # Call the main processing pipeline
        logger.info(f"Calling process_document_pipeline for {document_id}")
        result = await process_document_pipeline(document_id, document_dict)
        
        if result and result.get("status") == "success":
            logger.info(f"Successfully reprocessed document {document_id}")
            logger.info(f"Chunks: {result.get('chunks_created', 0)}, Entities: {result.get('entities_extracted', 0)}")
            
            # Update document status to ingested
            supabase_service.client.table("documents").update({
                "status": "ingested",
                "chunk_count": result.get("chunks_created", 0),
                "entity_count": result.get("entities_extracted", 0),
                "updated_at": datetime.utcnow().isoformat(),
                "ingested_at": datetime.utcnow().isoformat()
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