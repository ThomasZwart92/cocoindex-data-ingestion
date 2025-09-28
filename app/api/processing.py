"""
Processing API Endpoints for Document Processing and Job Management
Updated to include document processing trigger endpoint
"""
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
import os
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

NOTION_SECURITY_LEVELS = ["public", "client", "partner", "employee", "management"]


# Document processing endpoint moved to documents.py to avoid path conflicts


@router.post("/sources/scan")
async def sources_scan(
    background_tasks: BackgroundTasks,
    source: Optional[str] = Query(None, description="Source to scan: gdrive|notion|all"),
    security_level: str = Query("all", description="Security level to use"),
    folder_id: Optional[str] = Query(None, description="Google Drive folder to scan (optional)"),
    file_types: Optional[str] = Query(".pdf,.docx,.txt,.md,.gdoc,.gsheet,.gslides", description="Comma-separated file types for Drive"),
    force_update: bool = Query(False, description="Force update even if no changes"),
):
    """Compatibility endpoint used by some frontends for 'Scan now'.

    Delegates to the real process endpoints so it works under any app entrypoint.
    """
    try:
        source = (source or "all").lower()
        results: Dict[str, Any] = {}

        if source in ("gdrive", "google_drive", "drive", "all"):
            results["gdrive"] = await process_gdrive(
                background_tasks=background_tasks,
                security_level=security_level,
                folder_id=folder_id,
                file_types=file_types,
                force_update=force_update,
            )

        if source in ("notion", "all"):
            results["notion"] = await process_notion(
                background_tasks=background_tasks,
                security_level=security_level,
                workspace_id=None,
                force_update=force_update,
            )

        if not results:
            raise HTTPException(status_code=400, detail="Unknown source. Use gdrive, notion, or all")

        return {
            "status": "queued",
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sources_scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/stream", include_in_schema=False)
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
    security_level: str = Query("all", description="Security level to use or 'all'"),
    workspace_id: Optional[str] = Query(None, description="Specific workspace to scan"),
    force_update: bool = Query(False, description="Force update even if no changes"),

):
    """
    Scan Notion workspace and queue new/changed documents for processing
    """
    try:
        valid_levels = NOTION_SECURITY_LEVELS + ["all"]
        if security_level not in valid_levels:
            raise HTTPException(status_code=400, detail=f"Invalid security level. Must be one of: {valid_levels}")

        # Create job metadata so retries know what to re-run
        job_id = str(uuid4())
        metadata = {
            "security_level": security_level,
            "workspace_id": workspace_id,
            "force_update": force_update,
        }
        job_tracker.create_job(job_id, "notion_scan", metadata)

        # Queue background processing
        if security_level == "all":
            background_tasks.add_task(
                scan_notion_all,
                job_id=job_id,
                workspace_id=workspace_id,
                force_update=force_update,
            )
        else:
            background_tasks.add_task(
                scan_notion_workspace,
                job_id=job_id,
                security_level=security_level,
                workspace_id=workspace_id,
                force_update=force_update,
            )

        logger.info(f"Queued Notion scan job {job_id} with security level {security_level}")
        return {
            "job_id": job_id,
            "status": "queued",
            "source": "notion",
            "security_level": security_level,
            "message": "Notion scan queued for processing",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing Notion scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/gdrive")
async def process_gdrive(
    background_tasks: BackgroundTasks,
    security_level: str = Query("all", description="Security level to use or 'all'"),
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

        # Queue background processing for single or all security levels
        if security_level.lower() == "all":
            background_tasks.add_task(
                scan_google_drive_all,
                job_id=job_id,
                folder_id=folder_id,
                file_types=file_type_list,
                force_update=force_update,
            )
        else:
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


@router.get("/process/jobs/{job_id}/status", include_in_schema=False)
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
            metadata = job.get("metadata", {})
            if metadata.get("security_level") == "all":
                background_tasks.add_task(
                    scan_notion_all,
                    job_id=str(job_id),
                    workspace_id=metadata.get("workspace_id"),
                    force_update=metadata.get("force_update", False),
                )
            else:
                background_tasks.add_task(
                    scan_notion_workspace,
                    job_id=str(job_id),
                    security_level=metadata.get("security_level", "employee"),
                    workspace_id=metadata.get("workspace_id"),
                    force_update=metadata.get("force_update", False),
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



async def _scan_notion_level(
    job_id: str,
    security_level: str,
    workspace_id: Optional[str],
    force_update: bool,
    *,
    supabase_service: SupabaseService,
    progress_start: int,
    progress_span: int,
) -> tuple[int, int, int]:
    connector = NotionConnector(security_level=security_level)
    progress_span = max(progress_span, 1)
    progress_end = min(100, progress_start + progress_span)

    job_tracker.update_job(
        job_id,
        JobStatus.RUNNING,
        message=f"Scanning Notion workspace ({security_level})",
        progress=min(progress_start, progress_end),
    )

    documents = await connector.scan_workspace(workspace_id=workspace_id)
    total_docs = len(documents)
    new_count = 0
    updated_count = 0

    for idx, doc_data in enumerate(documents):
        progress = progress_end
        if total_docs:
            progress = progress_start + int(((idx + 1) / total_docs) * progress_span)
            progress = min(progress, progress_end)

        job_tracker.update_job(
            job_id,
            JobStatus.RUNNING,
            message=f"Processing {security_level} document {idx + 1}/{total_docs or 1}",
            progress=progress,
        )

        existing_docs = supabase_service.list_documents(
            source_type="notion",
            source_id=doc_data["source_id"],
        )

        doc_last_edited = doc_data.get("last_edited")
        doc_last_edited_iso = doc_last_edited.isoformat() if hasattr(doc_last_edited, "isoformat") else str(doc_last_edited or "")

        if existing_docs:
            existing = existing_docs[0]
            existing_updated = existing.updated_at if hasattr(existing, "updated_at") else existing.get("updated_at", "")
            existing_updated_iso = existing_updated.isoformat() if hasattr(existing_updated, "isoformat") else str(existing_updated or "")

            if force_update or doc_last_edited_iso > existing_updated_iso:
                existing_id = existing.id if hasattr(existing, "id") else existing["id"]
                existing_metadata = existing.metadata if hasattr(existing, "metadata") else existing.get("doc_metadata", {}) or {}
                supabase_service.update_document(
                    existing_id,
                    {
                        "content": doc_data["content"],
                        "doc_metadata": {**existing_metadata, **(doc_data.get("metadata", {}) or {})},
                        "status": "processing",
                        "updated_at": datetime.utcnow().isoformat(),
                    },
                )
                updated_count += 1
        else:
            new_doc = Document(
                name=doc_data["title"],
                source_type="notion",
                source_id=doc_data["source_id"],
                content=doc_data["content"],
                status=DocumentState.DISCOVERED,
                security_level=security_level,
                access_level=connector.get_access_level(security_level),
                doc_metadata={
                    **(doc_data.get("metadata", {}) or {}),
                    "content_hash": connector.get_content_hash(doc_data["content"]),
                },
            )
            supabase_service.create_document(new_doc)
            new_count += 1

    job_tracker.update_job(
        job_id,
        JobStatus.RUNNING,
        message=f"Finished Notion scan for {security_level}",
        progress=progress_end,
    )

    return total_docs, new_count, updated_count


async def scan_notion_workspace(
    job_id: str,
    security_level: str,
    workspace_id: Optional[str] = None,
    force_update: bool = False,
):
    try:
        job_tracker.update_job(
            job_id,
            JobStatus.RUNNING,
            message=f"Preparing Notion scan ({security_level})",
        )
        supabase_service = SupabaseService()
        total_docs, new_count, updated_count = await _scan_notion_level(
            job_id=job_id,
            security_level=security_level,
            workspace_id=workspace_id,
            force_update=force_update,
            supabase_service=supabase_service,
            progress_start=0,
            progress_span=100,
        )

        job_tracker.update_job(
            job_id,
            JobStatus.COMPLETED,
            message=f"Scan complete ({security_level}): {new_count} new, {updated_count} updated",
            progress=100,
            result={
                "documents_found": total_docs,
                "new_documents": new_count,
                "updated_documents": updated_count,
                "security_level": security_level,
            },
        )

        logger.info(
            f"Notion scan job {job_id} ({security_level}) completed: {new_count} new, {updated_count} updated"
        )

    except ValueError as e:
        logger.error(f"Notion scan job {job_id} failed for level {security_level}: {e}")
        job_tracker.update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Scan failed: {str(e)}",
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Error in Notion scan job {job_id}: {e}")
        job_tracker.update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Scan failed: {str(e)}",
            error=str(e),
        )


async def scan_notion_all(
    job_id: str,
    workspace_id: Optional[str] = None,
    force_update: bool = False,
):
    try:
        job_tracker.update_job(
            job_id,
            JobStatus.RUNNING,
            message="Preparing multi-connector Notion scan",
        )
        supabase_service = SupabaseService()
        active_levels = [level for level in NOTION_SECURITY_LEVELS if settings.notion_tokens.get(level)]

        if not active_levels:
            message = "No Notion connectors configured"
            job_tracker.update_job(
                job_id,
                JobStatus.FAILED,
                message=message,
                error=message,
            )
            logger.error(f"Notion scan job {job_id} failed: {message}")
            return

        totals = {"found": 0, "new": 0, "updated": 0}
        processed_levels: list[str] = []
        progress_cursor = 0

        for idx, level in enumerate(active_levels):
            remaining = len(active_levels) - idx
            span = max(1, (100 - progress_cursor) // remaining) if remaining else 0
            if idx == len(active_levels) - 1:
                span = max(1, 100 - progress_cursor)
            span = min(span, max(1, 100 - progress_cursor))

            try:
                level_totals = await _scan_notion_level(
                    job_id=job_id,
                    security_level=level,
                    workspace_id=workspace_id,
                    force_update=force_update,
                    supabase_service=supabase_service,
                    progress_start=progress_cursor,
                    progress_span=span,
                )
                totals["found"] += level_totals[0]
                totals["new"] += level_totals[1]
                totals["updated"] += level_totals[2]
                processed_levels.append(level)
            except ValueError as level_err:
                logger.warning(f"Skipping Notion scan for level {level}: {level_err}")
            except Exception as level_err:
                logger.warning(f"Error scanning Notion level {level}: {level_err}")
            progress_cursor = min(100, progress_cursor + span)

        if not processed_levels:
            message = "Unable to scan any Notion connectors"
            job_tracker.update_job(
                job_id,
                JobStatus.FAILED,
                message=message,
                error=message,
            )
            logger.error(f"Notion scan job {job_id} failed: {message}")
            return

        job_tracker.update_job(
            job_id,
            JobStatus.COMPLETED,
            message=f"Scan complete across connectors: {totals['new']} new, {totals['updated']} updated",
            progress=100,
            result={
                "documents_found": totals["found"],
                "new_documents": totals["new"],
                "updated_documents": totals["updated"],
                "security_levels": processed_levels,
            },
        )

        logger.info(
            f"Notion multi-connector scan job {job_id} completed across {processed_levels}: {totals['new']} new, {totals['updated']} updated"
        )

    except Exception as e:
        logger.error(f"Error in Notion multi-connector scan job {job_id}: {e}")
        job_tracker.update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Scan failed: {str(e)}",
            error=str(e),
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
        # Resolve folder IDs: prefer explicit, else configured list, else default (root/shared)
        folder_ids = None
        try:
            from app.config import settings as _settings
            if folder_id:
                folder_ids = [folder_id]
            elif _settings.google_drive_folder_ids:
                folder_ids = _settings.google_drive_folder_ids
            elif os.getenv("GOOGLE_DRIVE_FOLDER_ID"):
                folder_ids = [os.getenv("GOOGLE_DRIVE_FOLDER_ID")]  # legacy singular env var
        except Exception:
            folder_ids = [folder_id] if folder_id else None

        # Include Google Docs pseudo-extensions to ensure they aren't filtered out
        default_types = [".pdf", ".docx", ".txt", ".md", ".gdoc", ".gsheet", ".gslides"]

        documents = await connector.scan_drive(
            folder_id=None,  # use folder_ids instead
            folder_ids=folder_ids,
            file_types=file_types or default_types,
        )
        # Fallback: if no docs found and a configured folder_id was used, scan root + shared
        if not documents and folder_ids:
            logger.warning("No documents found in configured Google Drive folder(s); falling back to root + shared scan")
            documents = await connector.scan_drive(
                folder_id=None,
                folder_ids=None,
                file_types=file_types or default_types,
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
                source_type="gdrive",
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
                    updates = {
                        "content": doc_data.get("content", ""),
                        "doc_metadata": {**existing_metadata, **doc_data.get("metadata", {}), "security_level": security_level},
                        "status": "processing",
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                    # Also update top-level security fields if changed
                    if getattr(existing, 'security_level', None) != security_level:
                        updates["security_level"] = security_level
                        updates["access_level"] = connector.get_access_level(security_level)
                    supabase_service.update_document(
                        existing_id,
                        updates
                    )
                    updated_count += 1
            else:
                # Create new document
                new_doc = Document(
                    name=doc_data["title"],
                    source_type="gdrive",
                    source_id=doc_data["source_id"],
                    content=doc_data.get("content", ""),
                    status=DocumentState.DISCOVERED,
                    security_level=security_level,
                    access_level=connector.get_access_level(security_level),
                    doc_metadata={
                        **doc_data.get("metadata", {}),
                        "security_level": security_level,
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


async def scan_google_drive_all(
    job_id: str,
    folder_id: Optional[str] = None,
    file_types: list | None = None,
    force_update: bool = False,
):
    """Scan Google Drive across all configured security levels.

    Iterates over all service accounts (public, client, partner, employee, management)
    that have credentials configured, and aggregates results into one job.
    """
    levels = ["public", "client", "partner", "employee", "management"]
    totals = {"found": 0, "new": 0, "updated": 0}
    try:
        supabase_service = SupabaseService()

        processed_source_ids: set[str] = set()  # avoid double counting same file id

        for level in levels:
            try:
                job_tracker.update_job(job_id, JobStatus.RUNNING, message=f"Scanning Google Drive ({level})")

                connector = GoogleDriveConnector(security_level=level)

                # Resolve folder IDs per existing logic
                folder_ids = None
                try:
                    from app.config import settings as _settings
                    if folder_id:
                        folder_ids = [folder_id]
                    elif _settings.google_drive_folder_ids:
                        folder_ids = _settings.google_drive_folder_ids
                    elif os.getenv("GOOGLE_DRIVE_FOLDER_ID"):
                        folder_ids = [os.getenv("GOOGLE_DRIVE_FOLDER_ID")]  # legacy singular env var
                except Exception:
                    folder_ids = [folder_id] if folder_id else None

                default_types = [".pdf", ".docx", ".txt", ".md", ".gdoc", ".gsheet", ".gslides"]
                docs = await connector.scan_drive(
                    folder_id=None,
                    folder_ids=folder_ids,
                    file_types=file_types or default_types,
                )
                if not docs and folder_ids:
                    docs = await connector.scan_drive(
                        folder_id=None,
                        folder_ids=None,
                        file_types=file_types or default_types,
                    )

                new_count = 0
                updated_count = 0

                for doc_data in docs:
                    sid = doc_data["source_id"]
                    if sid in processed_source_ids:
                        continue

                    # Check existing doc
                    existing_docs = supabase_service.list_documents(
                        source_type="gdrive",
                        source_id=sid,
                    )

                    if existing_docs:
                        existing = existing_docs[0]
                        existing_updated = existing.updated_at if hasattr(existing, 'updated_at') else existing.get("updated_at", "")
                        if force_update or doc_data.get("modified_time") > existing_updated:
                            existing_id = existing.id if hasattr(existing, 'id') else existing["id"]
                            existing_metadata = existing.metadata if hasattr(existing, 'metadata') else existing.get("doc_metadata", {})
                            updates = {
                                "content": doc_data.get("content", ""),
                                "doc_metadata": {**existing_metadata, **doc_data.get("metadata", {}), "security_level": level},
                                "status": "processing",
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                            if getattr(existing, 'security_level', None) != level:
                                updates["security_level"] = level
                                updates["access_level"] = connector.get_access_level(level)
                            supabase_service.update_document(existing_id, updates)
                            updated_count += 1
                    else:
                        # Create new document with level metadata
                        new_doc = Document(
                            name=doc_data["title"],
                            source_type="gdrive",
                            source_id=sid,
                            content=doc_data.get("content", ""),
                            status=DocumentState.DISCOVERED,
                            security_level=level,
                            access_level=connector.get_access_level(level),
                            doc_metadata={
                                **doc_data.get("metadata", {}),
                                "security_level": level,
                                "content_hash": connector.get_content_hash(doc_data.get("content", "").encode() if doc_data.get("content") else b""),
                            },
                        )
                        supabase_service.create_document(new_doc)
                        new_count += 1

                    processed_source_ids.add(sid)

                totals["found"] += len(docs)
                totals["new"] += new_count
                totals["updated"] += updated_count

            except Exception as level_err:
                # Skip levels without configured credentials or access issues
                logger.warning(f"Skipping Google Drive scan for level {level}: {level_err}")
                continue

        job_tracker.update_job(
            job_id,
            JobStatus.COMPLETED,
            message=f"Scan complete across accounts: {totals['new']} new, {totals['updated']} updated",
            progress=100,
            result={
                "documents_found": totals["found"],
                "new_documents": totals["new"],
                "updated_documents": totals["updated"],
            },
        )

        logger.info(
            f"Google Drive multi-account scan job {job_id} completed: {totals['new']} new, {totals['updated']} updated"
        )

    except Exception as e:
        logger.error(f"Error in Google Drive multi-account scan job {job_id}: {e}")
        job_tracker.update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Scan failed: {str(e)}",
            error=str(e),
        )
