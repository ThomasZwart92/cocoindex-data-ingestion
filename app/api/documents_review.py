"""
Document Review API Endpoints
Handles document approval and rejection workflows
"""
from datetime import datetime, timezone
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query
import logging

from app.models.document import DocumentState
from app.services.supabase_service import SupabaseService
from app.config import settings
from app.api.schemas import ApproveDocumentResponse, RejectDocumentResponse, ReviewStatusOut, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{document_id}/approve",
    summary="Approve a document",
    description="Approve a pending document and transition it to APPROVED status. Triggers the publishing pipeline to store embeddings in Qdrant and entities/relationships in Neo4j.",
    response_model=ApproveDocumentResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def approve_document(document_id: UUID):
    """
    Approve a document that is pending review, moving it to APPROVED status.
    Triggers the publishing pipeline to store embeddings in Qdrant and entities/relationships in Neo4j.
    """
    logger.info(f"Approving document: {document_id}")
    try:
        supabase_service = SupabaseService()
        
        # Get document and verify status
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        if not doc_response.data:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        document = doc_response.data
        
        # Verify document is in pending_review status
        if document.get("status") != DocumentState.PENDING_REVIEW.value:
            raise HTTPException(
                status_code=400,
                detail=f"Document cannot be approved from status: {document.get('status')}. Must be in 'pending_review' status."
            )
        
        # Update document status to APPROVED
        update_result = supabase_service.client.table("documents").update({
            "status": DocumentState.APPROVED.value,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", str(document_id)).execute()
        
        # Trigger publishing task to push to vector and graph stores
        try:
            from app.tasks.document_tasks import publish_approved_document

            # Queue the publishing task
            task = publish_approved_document.delay(str(document_id))
            logger.info(f"Triggered publishing task {task.id} for document {document_id}")

        except Exception as e:
            logger.error(f"Failed to trigger publishing task: {e}")
            # Don't fail the approval if publishing task fails to queue

        logger.info(f"Document {document_id} approved and publishing task queued")
        
        return {
            "message": f"Document {document_id} approved successfully and publishing task queued",
            "status": DocumentState.APPROVED.value,
            "approved_at": update_result.data[0]["approved_at"] if update_result.data else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{document_id}/reject",
    summary="Reject a document",
    description="Reject a pending document and transition it to REJECTED with a reason. Document can be reprocessed later.",
    response_model=RejectDocumentResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def reject_document(
    document_id: UUID,
    reason: str = Query(..., description="Reason for rejection")
):
    """
    Reject a document that is pending review, moving it to REJECTED status.
    The document can be reprocessed after addressing the rejection reason.
    """
    logger.info(f"Rejecting document: {document_id} with reason: {reason}")
    try:
        supabase_service = SupabaseService()
        
        # Get document and verify status
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        if not doc_response.data:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        document = doc_response.data
        
        # Verify document is in pending_review status
        if document.get("status") != DocumentState.PENDING_REVIEW.value:
            raise HTTPException(
                status_code=400,
                detail=f"Document cannot be rejected from status: {document.get('status')}. Must be in 'pending_review' status."
            )
        
        # Update document status to REJECTED with reason
        existing_metadata = document.get("metadata", {}) or {}
        updated_metadata = {
            **existing_metadata,
            "rejection_reason": reason,
            "rejected_at": datetime.now(timezone.utc).isoformat()
        }
        
        update_result = supabase_service.client.table("documents").update({
            "status": DocumentState.REJECTED.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": updated_metadata
        }).eq("id", str(document_id)).execute()
        
        logger.info(f"Document {document_id} rejected with reason: {reason}")
        
        return {
            "message": f"Document {document_id} rejected",
            "status": DocumentState.REJECTED.value,
            "reason": reason,
            "can_reprocess": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{document_id}/review-status",
    summary="Get review status",
    description="Fetch the current review status, counts, and review metadata for a document.",
    response_model=ReviewStatusOut,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_review_status(document_id: UUID):
    """
    Get the review status and metadata for a document.
    """
    try:
        supabase_service = SupabaseService()
        
        # Get document
        doc_response = supabase_service.client.table("documents").select("*").eq("id", str(document_id)).single().execute()
        if not doc_response.data:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        document = doc_response.data
        metadata = document.get("metadata", {}) or {}
        
        # Get counts
        chunks_response = supabase_service.client.table("chunks").select("id").eq("document_id", str(document_id)).execute()
        entities_response = supabase_service.client.table("entities").select("id, is_verified").eq("document_id", str(document_id)).execute()
        
        verified_entities = [e for e in (entities_response.data or []) if e.get("is_verified")]
        unverified_entities = [e for e in (entities_response.data or []) if not e.get("is_verified")]
        
        return {
            "document_id": str(document_id),
            "status": document.get("status"),
            "can_approve": document.get("status") == DocumentState.PENDING_REVIEW.value,
            "can_reject": document.get("status") == DocumentState.PENDING_REVIEW.value,
            "can_reprocess": document.get("status") in [DocumentState.REJECTED.value, DocumentState.FAILED.value],
            "chunks_count": len(chunks_response.data) if chunks_response.data else 0,
            "entities_count": {
                "total": len(entities_response.data) if entities_response.data else 0,
                "verified": len(verified_entities),
                "unverified": len(unverified_entities)
            },
            "review_info": {
                "reviewed_at": document.get("reviewed_at"),
                "review_action": document.get("review_action"),
                "rejection_reason": metadata.get("rejection_reason")
            },
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting review status for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
