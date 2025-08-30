"""
Chunk Management API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from typing import Optional, List
from datetime import datetime
from uuid import UUID
import logging

from app.models.document import DocumentStatus
from app.services.document_processor import DocumentProcessor
from app.services.database import get_db_session, DocumentChunkTable as DocumentChunk, DocumentTable as Document
from app.services.supabase_client_db import document_service, chunk_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chunks", tags=["chunks"])


@router.get("/document/{document_id}")
async def get_document_chunks_supabase(document_id: str):
    """
    Get all chunks for a document using Supabase client
    """
    try:
        chunks = chunk_service.get_chunks(document_id)
        return chunks
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{chunk_id}")
async def update_chunk_supabase(chunk_id: str, request: dict):
    """
    Update chunk text using Supabase client
    """
    try:
        text = request.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        success = chunk_service.update_chunk(chunk_id, text)
        if success:
            # Note: We would need to get the document_id from the chunk first
            # For now, just return success
            # TODO: Add method to get single chunk by ID to retrieve document_id
            
            return {
                "message": f"Chunk {chunk_id} updated successfully",
                "success": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found or update failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{chunk_id}")
async def delete_chunk_supabase(chunk_id: str):
    """
    Delete a chunk using Supabase client
    """
    try:
        success = chunk_service.delete_chunk(chunk_id)
        if success:
            return {
                "message": f"Chunk {chunk_id} deleted successfully",
                "success": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/batch")
async def batch_update_chunks(request: dict):
    """
    Batch update multiple chunks
    """
    try:
        updates = request.get("updates", [])
        if not updates:
            raise HTTPException(status_code=400, detail="Updates array is required")
        
        updated_count = 0
        for update in updates:
            chunk_id = update.get("id")
            text = update.get("text")
            if chunk_id and text:
                if chunk_service.update_chunk(chunk_id, text):
                    updated_count += 1
        
        return {
            "message": f"Batch update completed",
            "updated_count": updated_count,
            "total_requested": len(updates)
        }
    except Exception as e:
        logger.error(f"Error in batch update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Original endpoints below (using old database service)
@router.get("/{chunk_id}", response_model=dict)
async def get_chunk(
    chunk_id: UUID,
    include_context: bool = Query(False, description="Include surrounding text context"),
    context_size: int = Query(200, description="Characters of context before/after"),
    db=Depends(get_db_session)
):
    """
    Get a single chunk with optional context
    """
    try:
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        result = {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "chunk_number": chunk.chunk_number,
            "text": chunk.chunk_text,
            "size": chunk.chunk_size,
            "start_position": chunk.start_position,
            "end_position": chunk.end_position,
            "metadata": chunk.chunk_metadata or {},
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None
        }
        
        # Add context if requested
        if include_context:
            document = db.query(Document).filter(Document.id == chunk.document_id).first()
            if document and document.content:
                # Get text before chunk
                before_start = max(0, chunk.start_position - context_size)
                before_text = document.content[before_start:chunk.start_position]
                
                # Get text after chunk
                after_end = min(len(document.content), chunk.end_position + context_size)
                after_text = document.content[chunk.end_position:after_end]
                
                result["context"] = {
                    "before": before_text,
                    "after": after_text
                }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{chunk_id}")
async def update_chunk(
    chunk_id: UUID,
    text: str,
    update_embeddings: bool = Query(True, description="Regenerate embeddings after update"),
    db=Depends(get_db_session)
):
    """
    Update the text content of a chunk
    """
    try:
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Store old text for logging
        old_text = chunk.chunk_text
        
        # Update chunk
        chunk.chunk_text = text
        chunk.chunk_size = len(text)
        chunk.updated_at = datetime.utcnow()
        
        # Add edit history to metadata
        if not chunk.metadata:
            chunk.metadata = {}
        
        if "edit_history" not in chunk.metadata:
            chunk.metadata["edit_history"] = []
        
        chunk.metadata["edit_history"].append({
            "edited_at": datetime.utcnow().isoformat(),
            "old_length": len(old_text),
            "new_length": len(text)
        })
        
        # Mark as manually edited
        chunk.metadata["manually_edited"] = True
        
        # Update document status
        document = db.query(Document).filter(Document.id == chunk.document_id).first()
        if document:
            document.status = DocumentStatus.PENDING_REVIEW
            document.updated_at = datetime.utcnow()
        
        db.commit()
        
        # TODO: Regenerate embeddings if requested
        if update_embeddings:
            # This would trigger embedding regeneration
            logger.info(f"Would regenerate embeddings for chunk {chunk_id}")
        
        logger.info(f"Updated chunk {chunk_id} (size: {len(old_text)} -> {len(text)})")
        return {
            "message": f"Chunk {chunk_id} updated successfully",
            "old_size": len(old_text),
            "new_size": len(text),
            "embeddings_updated": update_embeddings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chunk {chunk_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{chunk_id}")
async def delete_chunk(
    chunk_id: UUID,
    renumber: bool = Query(True, description="Renumber remaining chunks"),
    db=Depends(get_db_session)
):
    """
    Delete a chunk and optionally renumber remaining chunks
    """
    try:
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        document_id = chunk.document_id
        chunk_number = chunk.chunk_number
        
        # Delete the chunk
        db.delete(chunk)
        
        # Renumber remaining chunks if requested
        if renumber:
            remaining_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_number > chunk_number
            ).order_by(DocumentChunk.chunk_number).all()
            
            for remaining_chunk in remaining_chunks:
                remaining_chunk.chunk_number -= 1
        
        # Update document status
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = DocumentStatus.PENDING_REVIEW
            document.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Deleted chunk {chunk_id} from document {document_id}")
        return {
            "message": f"Chunk {chunk_id} deleted successfully",
            "renumbered": renumber,
            "affected_chunks": len(remaining_chunks) if renumber else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chunk {chunk_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/rechunk")
async def rechunk_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    method: str = Query("recursive", description="Chunking method: recursive or semantic"),
    chunk_size: int = Query(1500, ge=100, le=10000, description="Target chunk size"),
    chunk_overlap: int = Query(200, ge=0, le=1000, description="Overlap between chunks"),
    min_chunk_size: Optional[int] = Query(None, description="Minimum chunk size"),
    language: str = Query("markdown", description="Language for syntax-aware chunking"),
    db=Depends(get_db_session)
):
    """
    Rechunk an entire document with new parameters
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        if not document.content:
            raise HTTPException(status_code=400, detail=f"Document {document_id} has no content to chunk")
        
        # Update document status
        document.status = DocumentStatus.PROCESSING
        document.updated_at = datetime.utcnow()
        db.commit()
        
        # Queue background rechunking
        background_tasks.add_task(
            rechunk_document_task,
            document_id=str(document_id),
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            language=language
        )
        
        logger.info(f"Queued rechunking for document {document_id} with method={method}, size={chunk_size}")
        return {
            "message": f"Document {document_id} queued for rechunking",
            "parameters": {
                "method": method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "min_chunk_size": min_chunk_size,
                "language": language
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rechunking document {document_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{chunk_id}/split")
async def split_chunk(
    chunk_id: UUID,
    split_position: int = Query(..., description="Position in chunk text to split at"),
    db=Depends(get_db_session)
):
    """
    Split a chunk into two chunks at the specified position
    """
    try:
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        if split_position <= 0 or split_position >= len(chunk.chunk_text):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid split position. Must be between 1 and {len(chunk.chunk_text)-1}"
            )
        
        # Split the text
        first_text = chunk.chunk_text[:split_position]
        second_text = chunk.chunk_text[split_position:]
        
        # Update first chunk
        chunk.chunk_text = first_text
        chunk.chunk_size = len(first_text)
        chunk.end_position = chunk.start_position + len(first_text)
        chunk.updated_at = datetime.utcnow()
        
        # Renumber subsequent chunks
        subsequent_chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == chunk.document_id,
            DocumentChunk.chunk_number > chunk.chunk_number
        ).order_by(DocumentChunk.chunk_number).all()
        
        for subsequent in subsequent_chunks:
            subsequent.chunk_number += 1
        
        # Create second chunk
        new_chunk = DocumentChunk(
            document_id=chunk.document_id,
            chunk_number=chunk.chunk_number + 1,
            chunk_text=second_text,
            chunk_size=len(second_text),
            start_position=chunk.end_position,
            end_position=chunk.end_position + len(second_text),
            metadata={"split_from": str(chunk_id)}
        )
        db.add(new_chunk)
        
        # Update document status
        document = db.query(Document).filter(Document.id == chunk.document_id).first()
        if document:
            document.status = DocumentStatus.PENDING_REVIEW
            document.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Split chunk {chunk_id} at position {split_position}")
        return {
            "message": f"Chunk {chunk_id} split successfully",
            "first_chunk": {
                "id": str(chunk.id),
                "number": chunk.chunk_number,
                "size": len(first_text)
            },
            "second_chunk": {
                "id": str(new_chunk.id),
                "number": new_chunk.chunk_number,
                "size": len(second_text)
            },
            "renumbered_chunks": len(subsequent_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error splitting chunk {chunk_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge")
async def merge_chunks(
    chunk_ids: List[UUID],
    db=Depends(get_db_session)
):
    """
    Merge multiple chunks into a single chunk
    """
    try:
        if len(chunk_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 chunks required for merging")
        
        # Get all chunks
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.id.in_(chunk_ids)
        ).order_by(DocumentChunk.chunk_number).all()
        
        if len(chunks) != len(chunk_ids):
            raise HTTPException(status_code=404, detail="One or more chunks not found")
        
        # Verify all chunks are from same document
        document_ids = set(chunk.document_id for chunk in chunks)
        if len(document_ids) > 1:
            raise HTTPException(status_code=400, detail="All chunks must be from the same document")
        
        # Merge text
        merged_text = " ".join(chunk.chunk_text for chunk in chunks)
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        
        # Update first chunk with merged content
        first_chunk.chunk_text = merged_text
        first_chunk.chunk_size = len(merged_text)
        first_chunk.end_position = last_chunk.end_position
        first_chunk.updated_at = datetime.utcnow()
        
        if not first_chunk.metadata:
            first_chunk.metadata = {}
        first_chunk.metadata["merged_from"] = [str(c.id) for c in chunks[1:]]
        
        # Delete other chunks
        for chunk in chunks[1:]:
            db.delete(chunk)
        
        # Renumber remaining chunks
        remaining_chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == first_chunk.document_id,
            DocumentChunk.chunk_number > last_chunk.chunk_number
        ).order_by(DocumentChunk.chunk_number).all()
        
        chunks_removed = len(chunks) - 1
        for remaining in remaining_chunks:
            remaining.chunk_number -= chunks_removed
        
        # Update document status
        document = db.query(Document).filter(Document.id == first_chunk.document_id).first()
        if document:
            document.status = DocumentStatus.PENDING_REVIEW
            document.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Merged {len(chunks)} chunks into chunk {first_chunk.id}")
        return {
            "message": f"Successfully merged {len(chunks)} chunks",
            "merged_chunk": {
                "id": str(first_chunk.id),
                "number": first_chunk.chunk_number,
                "size": len(merged_text)
            },
            "deleted_chunks": len(chunks) - 1,
            "renumbered_chunks": len(remaining_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging chunks: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


async def rechunk_document_task(
    document_id: str,
    method: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: Optional[int],
    language: str
):
    """
    Background task to rechunk a document
    """
    try:
        processor = DocumentProcessor()
        db = next(get_db_session())
        
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document {document_id} not found for rechunking")
            return
        
        # Perform chunking
        chunks = await processor.chunk_document(
            content=document.content,
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            language=language
        )
        
        # Delete old chunks
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # Create new chunks
        for i, chunk_data in enumerate(chunks):
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_number=i,
                chunk_text=chunk_data["text"],
                chunk_size=len(chunk_data["text"]),
                start_position=chunk_data.get("start", 0),
                end_position=chunk_data.get("end", 0),
                metadata={
                    "method": method,
                    "chunk_size_target": chunk_size,
                    "overlap": chunk_overlap,
                    "language": language
                }
            )
            db.add(chunk)
        
        # Update document
        document.status = DocumentStatus.PENDING_REVIEW
        document.updated_at = datetime.utcnow()
        
        if not document.metadata:
            document.metadata = {}
        document.metadata["last_chunking"] = {
            "method": method,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        db.commit()
        logger.info(f"Rechunked document {document_id}: created {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error in rechunk_document_task for {document_id}: {e}")
        try:
            db = next(get_db_session())
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.status = DocumentStatus.FAILED
                document.updated_at = datetime.utcnow()
                db.commit()
        except:
            pass