"""
Processing API Endpoints for Document Processing and Job Management
Updated to include document processing trigger endpoint
"""
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
import logging
import asyncio
import json

from app.connectors.notion_connector import NotionConnector
from app.connectors.google_drive_connector import GoogleDriveConnector
from app.services.supabase_service import SupabaseService
from app.services.job_tracker import JobTracker, JobStatus
from app.services.state_manager_simple import SimpleDocumentStateManager
from app.models.document_state import DocumentState
from app.models.document import Document
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["processing"])

# In-memory job tracker (in production, use Redis or database)
job_tracker = JobTracker()


# Document processing endpoint moved to documents.py to avoid path conflicts


@router.get("/documents/{document_id}/stream")
async def stream_document_status(document_id: str):
    """
    Stream real-time status updates for a document via Server-Sent Events (SSE).
    The client can connect to this endpoint to receive live updates as the document
    progresses through the processing pipeline.
    """
    state_manager = SimpleDocumentStateManager()
    
    async def generate():
        """Generate SSE events with document status updates"""
        consecutive_errors = 0
        max_errors = 5
        
        while consecutive_errors < max_errors:
            try:
                # Get current state
                state_info = await state_manager.get_state(document_id)
                
                if state_info:
                    # Prepare data to send
                    data = {
                        "document_id": document_id,
                        "state": state_info.current_state.value,
                        "updated_at": state_info.updated_at.isoformat(),
                        "error_count": state_info.error_count,
                        "retry_count": state_info.retry_count,
                        "time_in_state": state_info.get_time_in_state()
                    }
                    
                    # Add transition history if available
                    if state_info.transition_history:
                        last_transition = state_info.transition_history[-1]
                        data["last_transition"] = {
                            "from": last_transition.from_state.value,
                            "to": last_transition.to_state.value,
                            "timestamp": last_transition.timestamp.isoformat(),
                            "reason": last_transition.reason
                        }
                    
                    # Send SSE event
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # Check if we're in a terminal state
                    if state_info.current_state in [
                        DocumentState.INGESTED,
                        DocumentState.FAILED,
                        DocumentState.REJECTED
                    ]:
                        # Send final event and close
                        yield f"data: {json.dumps({'event': 'complete', 'final_state': state_info.current_state.value})}\n\n"
                        break
                    
                    consecutive_errors = 0  # Reset error counter on success
                else:
                    # Document not found
                    yield f"data: {json.dumps({'error': 'Document not found', 'document_id': document_id})}\n\n"
                    break
                
                # Wait before next check
                await asyncio.sleep(2)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in SSE stream for document {document_id}: {e}")
                
                # Send error event
                yield f"data: {json.dumps({'error': str(e), 'retry_count': consecutive_errors})}\n\n"
                
                if consecutive_errors >= max_errors:
                    yield f"data: {json.dumps({'event': 'error', 'message': 'Max errors reached, closing stream'})}\n\n"
                    break
                
                # Exponential backoff on errors
                await asyncio.sleep(2 ** consecutive_errors)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/jobs/{job_id}/status")
async def get_celery_job_status(job_id: str):
    """
    Get the status of a Celery job.
    """
    try:
        from celery.result import AsyncResult
        from app.worker import celery_app
        
        result = AsyncResult(job_id, app=celery_app)
        
        # Get job info
        job_info = {
            "job_id": job_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None
        }
        
        # Add result or error info if available
        if result.ready():
            if result.successful():
                job_info["result"] = result.result
            elif result.failed():
                job_info["error"] = str(result.info)
                job_info["traceback"] = result.traceback
        
        return job_info
        
    except Exception as e:
        logger.error(f"Error checking job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/notion")
async def process_notion(
    background_tasks: BackgroundTasks,
    security_level: str = Query("employee", description="Security level to use"),
    workspace_id: Optional[str] = Query(None, description="Specific workspace to scan"),
    force_update: bool = Query(False, description="Force update even if no changes"),

):
    """
    Scan Notion workspace and queue new/changed documents for processing
    """
    try:
        # Validate security level
        valid_levels = ["public", "client", "partner", "employee", "management"]
        if security_level not in valid_levels:
            raise HTTPException(status_code=400, detail=f"Invalid security level. Must be one of: {valid_levels}")
        
        # Create job
        job_id = str(uuid4())
        job_tracker.create_job(job_id, "notion_scan", {"security_level": security_level})
        
        # Queue background processing
        background_tasks.add_task(
            scan_notion_workspace,
            job_id=job_id,
            security_level=security_level,
            workspace_id=workspace_id,
            force_update=force_update
        )
        
        logger.info(f"Queued Notion scan job {job_id} with security level {security_level}")
        return {
            "job_id": job_id,
            "status": "queued",
            "source": "notion",
            "security_level": security_level,
            "message": "Notion scan queued for processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing Notion scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/gdrive")
async def process_gdrive(
    background_tasks: BackgroundTasks,
    security_level: str = Query("employee", description="Security level to use"),
    folder_id: Optional[str] = Query(None, description="Specific folder to scan"),
    file_types: Optional[str] = Query(".pdf,.docx,.txt,.md", description="Comma-separated file types"),
    force_update: bool = Query(False, description="Force update even if no changes"),

):
    """
    Scan Google Drive and queue new/changed documents for processing
    """
    try:
        # Validate security level
        valid_levels = ["public", "client", "partner", "employee", "management"]
        if security_level not in valid_levels:
            raise HTTPException(status_code=400, detail=f"Invalid security level. Must be one of: {valid_levels}")
        
        # Parse file types
        file_type_list = [ft.strip() for ft in file_types.split(",")] if file_types else []
        
        # Create job
        job_id = str(uuid4())
        job_tracker.create_job(job_id, "gdrive_scan", {
            "security_level": security_level,
            "file_types": file_type_list
        })
        
        # Queue background processing
        background_tasks.add_task(
            scan_google_drive,
            job_id=job_id,
            security_level=security_level,
            folder_id=folder_id,
            file_types=file_type_list,
            force_update=force_update
        )
        
        logger.info(f"Queued Google Drive scan job {job_id} with security level {security_level}")
        return {
            "job_id": job_id,
            "status": "queued",
            "source": "google_drive",
            "security_level": security_level,
            "file_types": file_type_list,
            "message": "Google Drive scan queued for processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing Google Drive scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/process/jobs/{job_id}/status")
async def get_process_job_status(job_id: UUID):
    """
    Get the status of a processing job from job tracker
    """
    try:
        job = job_tracker.get_job(str(job_id))
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return {
            "job_id": str(job_id),
            "type": job["type"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "metadata": job.get("metadata", {}),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "completed_at": job.get("completed_at"),
            "error": job.get("error"),
            "result": job.get("result", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all processing jobs
    """
    try:
        jobs = job_tracker.list_jobs(status=status, job_type=type, limit=limit, offset=offset)
        
        return {
            "jobs": jobs,
            "total": len(jobs),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: UUID,
    background_tasks: BackgroundTasks
):
    """
    Retry a failed job
    """
    try:
        job = job_tracker.get_job(str(job_id))
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job["status"] != JobStatus.FAILED:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not in failed state")
        
        # Reset job status
        job_tracker.update_job(str(job_id), JobStatus.QUEUED, message="Retrying job")
        
        # Re-queue based on job type
        if job["type"] == "notion_scan":
            background_tasks.add_task(
                scan_notion_workspace,
                job_id=str(job_id),
                **job.get("metadata", {})
            )
        elif job["type"] == "gdrive_scan":
            background_tasks.add_task(
                scan_google_drive,
                job_id=str(job_id),
                **job.get("metadata", {})
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown job type: {job['type']}")
        
        logger.info(f"Retrying job {job_id}")
        return {
            "job_id": str(job_id),
            "status": "queued",
            "message": f"Job {job_id} queued for retry"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def scan_notion_workspace(
    job_id: str,
    security_level: str,
    workspace_id: Optional[str] = None,
    force_update: bool = False
):
    """
    Background task to scan Notion workspace using real API
    """
    try:
        job_tracker.update_job(job_id, JobStatus.RUNNING, message="Connecting to Notion")
        
        # Use the real NotionConnector with security level
        # The connector will use the appropriate API key from environment variables
        connector = NotionConnector(security_level=security_level)
        
        # Scan for documents
        job_tracker.update_job(job_id, JobStatus.RUNNING, message="Scanning workspace", progress=10)
        documents = await connector.scan_workspace(workspace_id=workspace_id)
        
        # Check for changes
        supabase_service = SupabaseService()
        new_count = 0
        updated_count = 0
        
        for idx, doc_data in enumerate(documents):
            progress = 10 + (80 * idx // len(documents))
            job_tracker.update_job(
                job_id, 
                JobStatus.RUNNING, 
                message=f"Processing document {idx+1}/{len(documents)}", 
                progress=progress
            )
            
            # Check if document exists using Supabase
            existing_docs = supabase_service.list_documents(
                source_type="notion",
                source_id=doc_data["source_id"]
            )
            
            if existing_docs:
                # Check for updates
                existing = existing_docs[0]
                # Handle both Document objects and dicts
                existing_updated = existing.updated_at if hasattr(existing, 'updated_at') else existing.get("updated_at", "")
                if force_update or doc_data.get("last_edited") > existing_updated:
                    # Update existing document
                    existing_id = existing.id if hasattr(existing, 'id') else existing["id"]
                    existing_metadata = existing.metadata if hasattr(existing, 'metadata') else existing.get("doc_metadata", {})
                    supabase_service.update_document(
                        existing_id,
                        {
                            "content": doc_data["content"],
                            "doc_metadata": {**existing_metadata, **doc_data.get("metadata", {})},
                            "status": "processing",
                            "updated_at": datetime.utcnow().isoformat()
                        }
                    )
                    updated_count += 1
            else:
                # Create new document
                new_doc = Document(
                    name=doc_data["title"],
                    source_type="notion",
                    source_id=doc_data["source_id"],
                    content=doc_data["content"],
                    status=DocumentState.DISCOVERED,
                    security_level=security_level,
                    access_level=connector.get_access_level(security_level),
                    doc_metadata={
                        **doc_data.get("metadata", {}),
                        "content_hash": connector.get_content_hash(doc_data["content"])
                    }
                )
                created_doc = supabase_service.create_document(new_doc)
                new_count += 1
        
        # Update job with results
        job_tracker.update_job(
            job_id,
            JobStatus.COMPLETED,
            message=f"Scan complete: {new_count} new, {updated_count} updated",
            progress=100,
            result={
                "documents_found": len(documents),
                "new_documents": new_count,
                "updated_documents": updated_count
            }
        )
        
        logger.info(f"Notion scan job {job_id} completed: {new_count} new, {updated_count} updated")
        
    except Exception as e:
        logger.error(f"Error in Notion scan job {job_id}: {e}")
        job_tracker.update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Scan failed: {str(e)}",
            error=str(e)
        )


async def scan_google_drive(
    job_id: str,
    security_level: str,
    folder_id: Optional[str] = None,
    file_types: list = None,
    force_update: bool = False
):
    """
    Background task to scan Google Drive using real API
    """
    try:
        job_tracker.update_job(job_id, JobStatus.RUNNING, message="Connecting to Google Drive")
        
        # Use the real GoogleDriveConnector with security level
        # The connector will use the appropriate service account from environment variables
        connector = GoogleDriveConnector(security_level=security_level)
        
        # Scan for documents
        job_tracker.update_job(job_id, JobStatus.RUNNING, message="Scanning drive", progress=10)
        documents = await connector.scan_drive(
            folder_id=folder_id,
            file_types=file_types or [".pdf", ".docx", ".txt", ".md"]
        )
        
        # Check for changes
        supabase_service = SupabaseService()
        new_count = 0
        updated_count = 0
        
        for idx, doc_data in enumerate(documents):
            progress = 10 + (80 * idx // len(documents))
            job_tracker.update_job(
                job_id,
                JobStatus.RUNNING,
                message=f"Processing document {idx+1}/{len(documents)}",
                progress=progress
            )
            
            # Check if document exists using Supabase
            existing_docs = supabase_service.list_documents(
                source_type="google_drive",
                source_id=doc_data["source_id"]
            )
            
            if existing_docs:
                # Check for updates
                existing = existing_docs[0]
                # Handle both Document objects and dicts
                existing_updated = existing.updated_at if hasattr(existing, 'updated_at') else existing.get("updated_at", "")
                if force_update or doc_data.get("modified_time") > existing_updated:
                    # Update existing document
                    existing_id = existing.id if hasattr(existing, 'id') else existing["id"]
                    existing_metadata = existing.metadata if hasattr(existing, 'metadata') else existing.get("doc_metadata", {})
                    supabase_service.update_document(
                        existing_id,
                        {
                            "content": doc_data.get("content", ""),
                            "doc_metadata": {**existing_metadata, **doc_data.get("metadata", {})},
                            "status": "processing",
                            "updated_at": datetime.utcnow().isoformat()
                        }
                    )
                    updated_count += 1
            else:
                # Create new document
                new_doc = Document(
                    name=doc_data["title"],
                    source_type="google_drive",
                    source_id=doc_data["source_id"],
                    content=doc_data.get("content", ""),
                    status=DocumentState.DISCOVERED,
                    security_level=security_level,
                    access_level=connector.get_access_level(security_level),
                    doc_metadata={
                        **doc_data.get("metadata", {}),
                        "content_hash": connector.get_content_hash(doc_data.get("content", "").encode() if doc_data.get("content") else b"")
                    }
                )
                created_doc = supabase_service.create_document(new_doc)
                new_count += 1
        
        # Update job with results
        job_tracker.update_job(
            job_id,
            JobStatus.COMPLETED,
            message=f"Scan complete: {new_count} new, {updated_count} updated",
            progress=100,
            result={
                "documents_found": len(documents),
                "new_documents": new_count,
                "updated_documents": updated_count
            }
        )
        
        logger.info(f"Google Drive scan job {job_id} completed: {new_count} new, {updated_count} updated")
        
    except Exception as e:
        logger.error(f"Error in Google Drive scan job {job_id}: {e}")
        job_tracker.update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Scan failed: {str(e)}",
            error=str(e)
        )