"""
Ingestion API Endpoints
API routes for triggering and monitoring document ingestion
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from celery.result import AsyncResult
import logging

from app.tasks.ingestion_tasks import celery_app, ingest_notion_pages
from app.config import settings
# from app.services.auth_service import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


class NotionIngestionRequest(BaseModel):
    """Request model for Notion ingestion"""
    token: Optional[str] = Field(default=None, description="Notion API token (uses env var if not provided)")
    database_ids: Optional[List[str]] = Field(default=None, description="Database IDs to ingest")
    page_ids: Optional[List[str]] = Field(default=None, description="Specific page IDs to ingest")
    full_scan: bool = Field(default=False, description="Process all pages regardless of changes")
    auto_approve: bool = Field(default=False, description="Auto-approve high confidence results")


class IngestionResponse(BaseModel):
    """Response model for ingestion requests"""
    task_id: str
    status: str
    message: str
    check_url: str


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    state: str
    current: Optional[int] = None
    total: Optional[int] = None
    info: Optional[Dict[str, Any]] = None


@router.post("/notion", response_model=IngestionResponse)
async def trigger_notion_ingestion(
    request: NotionIngestionRequest,
    # current_user: Dict = Depends(get_current_user)
):
    """
    Trigger Notion document ingestion
    
    Starts an asynchronous task to ingest documents from Notion.
    Returns a task ID that can be used to check progress.
    """
    try:
        # Use provided token or fall back to environment variable
        notion_token = request.token or settings.notion_api_key
        
        if not notion_token:
            raise HTTPException(
                status_code=400,
                detail="Notion API token not provided and not configured in environment"
            )
        
        # Validate at least one source is specified
        if not request.database_ids and not request.page_ids:
            # Use defaults from settings if available
            database_ids = settings.notion_database_ids
            if not database_ids:
                raise HTTPException(
                    status_code=400,
                    detail="No database IDs or page IDs specified"
                )
        else:
            database_ids = request.database_ids
        
        # Start async task
        task = ingest_notion_pages.delay(
            notion_token=notion_token,
            database_ids=database_ids,
            page_ids=request.page_ids,
            full_scan=request.full_scan,
            auto_approve=request.auto_approve
        )
        
        logger.info(f"Started Notion ingestion task: {task.id}")
        
        return IngestionResponse(
            task_id=task.id,
            status="started",
            message="Notion ingestion task started",
            check_url=f"/api/ingestion/status/{task.id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start Notion ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of an ingestion task
    
    Check the progress and results of an async ingestion task.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        response = TaskStatusResponse(
            task_id=task_id,
            state=result.state
        )
        
        if result.state == 'PENDING':
            response.info = {'status': 'Task not found or pending'}
        elif result.state == 'PROCESSING':
            response.info = result.info
        elif result.state == 'SUCCESS':
            response.info = result.result
        elif result.state == 'FAILURE':
            response.info = {
                'error': str(result.info),
                'status': 'failed'
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running ingestion task
    
    Attempts to cancel an in-progress ingestion task.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        result.revoke(terminate=True)
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancellation requested"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_active_tasks():
    """
    List all active ingestion tasks
    
    Returns information about currently running and queued tasks.
    """
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()
        
        return {
            "active": active,
            "scheduled": scheduled,
            "reserved": reserved,
            "stats": {
                "active_count": sum(len(tasks) for tasks in (active or {}).values()),
                "scheduled_count": sum(len(tasks) for tasks in (scheduled or {}).values()),
                "reserved_count": sum(len(tasks) for tasks in (reserved or {}).values())
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_ingestion():
    """
    Test ingestion with sample data
    
    Runs a quick test to verify the ingestion pipeline is working.
    """
    try:
        # Create test content
        test_content = """
        # Test Document
        
        This is a test document for the Model X500 water dispenser.
        
        ## Features
        - Advanced filtration system
        - IoT connectivity
        - Temperature control
        
        ## Troubleshooting
        If you encounter error code E501, check the pump system.
        
        The Engineering team has developed firmware updates to address this issue.
        """
        
        # Process through pipeline (simplified for testing)
        from app.services.document_processor import DocumentProcessor
        from app.services.embedding_service import EmbeddingService
        
        processor = DocumentProcessor()
        embedder = EmbeddingService()
        
        # Chunk the content
        chunks = await processor.chunk_document(
            content=test_content,
            method="recursive",
            chunk_size=500
        )
        
        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            embedding = await embedder.embed_text(chunk["text"])
            embeddings.append(len(embedding))  # Just return dimension
        
        return {
            "status": "success",
            "message": "Test ingestion completed",
            "results": {
                "chunks_created": len(chunks),
                "embedding_dimensions": embeddings,
                "sample_chunk": chunks[0]["text"][:100] if chunks else None
            }
        }
        
    except Exception as e:
        logger.error(f"Test ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def list_configured_sources():
    """
    List configured data sources
    
    Returns information about configured Notion and Google Drive sources.
    """
    sources = {
        "notion": {
            "configured": bool(settings.notion_api_key),
            "database_ids": settings.notion_database_ids if settings.notion_api_key else [],
            "scan_interval": "30 minutes" if settings.notion_api_key else "disabled"
        },
        "google_drive": {
            "configured": bool(settings.google_drive_credentials_path),
            "folder_ids": settings.google_drive_folder_ids if settings.google_drive_credentials_path else [],
            "scan_interval": "30 minutes" if settings.google_drive_credentials_path else "disabled"
        }
    }
    
    return sources