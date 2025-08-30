"""
Celery Worker with Real CocoIndex Integration
Handles async document processing through the proper dataflow pipeline
"""
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from celery import Celery, Task
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Celery configuration
celery_app = Celery(
    'cocoindex_worker',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (prevent memory leaks)
)

# Import services and models
from app.models.document_state import DocumentState
from app.services.state_manager_simple import SimpleDocumentStateManager


class CallbackTask(Task):
    """Task with callbacks for state management"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        document_id = kwargs.get('document_id')
        if document_id:
            logger.info(f"Task succeeded for document {document_id}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        document_id = kwargs.get('document_id')
        if document_id:
            logger.error(f"Task failed for document {document_id}: {exc}")
            # Update state to FAILED
            asyncio.run(update_document_state(document_id, DocumentState.FAILED, str(exc)))
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        document_id = kwargs.get('document_id')
        if document_id:
            logger.warning(f"Retrying task for document {document_id}: {exc}")


async def update_document_state(
    document_id: str, 
    state: DocumentState, 
    reason: Optional[str] = None
) -> bool:
    """Update document state asynchronously"""
    try:
        manager = SimpleDocumentStateManager()
        await manager.transition(
            document_id=document_id,
            to_state=state,
            reason=reason
        )
        return True
    except Exception as e:
        logger.error(f"Failed to update state for {document_id}: {e}")
        return False


@celery_app.task(bind=True, base=CallbackTask, max_retries=3)
def process_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
    """
    Process a document through the CocoIndex pipeline
    
    Args:
        document_id: UUID of the document to process
        **kwargs: Additional processing options
    
    Returns:
        Processing result with status and metadata
    """
    logger.info(f"Starting processing for document {document_id}")
    
    try:
        # Update state to PROCESSING
        asyncio.run(update_document_state(document_id, DocumentState.PROCESSING))
        
        # Use the dual-write flow to populate all Supabase tables
        import cocoindex
        from app.flows.document_ingestion_dual_write import run_dual_write_flow
        
        # Initialize CocoIndex if not already done
        cocoindex.init()
        
        # Process documents with dual-write pattern
        # This will populate all 8 Supabase tables + Qdrant + Neo4j
        result = run_dual_write_flow()
        
        if result and hasattr(result, 'success') and result.success:
            # Update state to PENDING_REVIEW
            asyncio.run(update_document_state(
                document_id, 
                DocumentState.PENDING_REVIEW,
                f"Processed successfully. {getattr(result, 'documents_processed', 1)} documents processed."
            ))
            
            return {
                "status": "success",
                "document_id": document_id,
                "documents_processed": getattr(result, 'documents_processed', 1),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            error_msg = getattr(result, 'error', 'Unknown error') if result else 'Flow returned no result'
            raise Exception(f"Flow processing failed: {error_msg}")
            
    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {exc}")
        
        # Retry with exponential backoff
        retry_count = self.request.retries
        if retry_count < self.max_retries:
            countdown = 2 ** retry_count * 10  # 10, 20, 40 seconds
            logger.info(f"Retrying in {countdown} seconds (attempt {retry_count + 1}/{self.max_retries})")
            raise self.retry(exc=exc, countdown=countdown)
        
        # Max retries exceeded, mark as failed
        asyncio.run(update_document_state(
            document_id,
            DocumentState.FAILED,
            f"Processing failed after {self.max_retries} attempts: {str(exc)}"
        ))
        
        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task
def check_processing_status(document_id: str) -> Dict[str, Any]:
    """Check the processing status of a document"""
    try:
        manager = SimpleDocumentStateManager()
        state_info = asyncio.run(manager.get_state(document_id))
        
        if state_info:
            return {
                "status": "found",
                "document_id": document_id,
                "current_state": state_info.current_state.value,
                "created_at": state_info.created_at.isoformat(),
                "updated_at": state_info.updated_at.isoformat(),
                "transition_count": len(state_info.transition_history),
                "time_in_state": state_info.get_time_in_state(),
                "metadata": state_info.metadata
            }
        else:
            return {
                "status": "not_found",
                "document_id": document_id,
                "error": "Document not found"
            }
            
    except Exception as exc:
        logger.error(f"Error checking status for {document_id}: {exc}")
        return {
            "status": "error",
            "document_id": document_id,
            "error": str(exc)
        }


# Health check task
@celery_app.task
def health_check() -> Dict[str, str]:
    """Simple health check task"""
    return {
        "status": "healthy",
        "worker": "cocoindex_worker",
        "timestamp": datetime.utcnow().isoformat()
    }


# Keep legacy test task for compatibility
@celery_app.task
def test_task(x, y):
    """Simple test task"""
    return x + y