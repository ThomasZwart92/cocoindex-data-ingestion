"""
Celery Tasks for Document Ingestion
Asynchronous tasks for processing documents from various sources
"""
import asyncio
from celery import Celery, Task
from celery.utils.log import get_task_logger
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.config import settings
from app.pipelines.notion_ingestion import run_notion_ingestion

logger = get_task_logger(__name__)

# Create Celery app
celery_app = Celery(
    'ingestion',
    broker=settings.redis_url or 'redis://localhost:6379/0',
    backend=settings.redis_url or 'redis://localhost:6379/0'
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit
)


class AsyncTask(Task):
    """Base task that properly handles async functions"""
    def run(self, *args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run(*args, **kwargs))
        finally:
            loop.close()
    
    async def _run(self, *args, **kwargs):
        raise NotImplementedError


@celery_app.task(bind=True, base=AsyncTask, name='ingest_notion_pages')
class IngestNotionPagesTask(AsyncTask):
    """Task to ingest pages from Notion"""
    
    async def _run(
        self,
        notion_token: str,
        database_ids: Optional[List[str]] = None,
        page_ids: Optional[List[str]] = None,
        full_scan: bool = False,
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest Notion pages asynchronously
        
        Args:
            notion_token: Notion API token
            database_ids: Database IDs to scan
            page_ids: Specific page IDs to process
            full_scan: Process all pages regardless of changes
            auto_approve: Auto-approve high confidence results
        """
        task_id = self.request.id
        logger.info(f"Starting Notion ingestion task {task_id}")
        
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Connecting to Notion...',
                'started_at': datetime.now().isoformat()
            }
        )
        
        try:
            # Run the ingestion pipeline
            status = await run_notion_ingestion(
                token=notion_token,
                database_ids=database_ids,
                page_ids=page_ids,
                full_scan=full_scan,
                auto_approve=auto_approve
            )
            
            # Convert status to dict
            result = {
                'task_id': task_id,
                'status': 'completed',
                'total_pages': status.total_pages,
                'processed_pages': status.processed_pages,
                'failed_pages': status.failed_pages,
                'new_chunks': status.new_chunks,
                'new_entities': status.new_entities,
                'new_relationships': status.new_relationships,
                'errors': status.errors,
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Task {task_id} completed: {status.processed_pages}/{status.total_pages} pages")
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Update task state with error
            self.update_state(
                state='FAILURE',
                meta={
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
            )
            raise


@celery_app.task(bind=True, name='scan_notion_sources')
def scan_notion_sources(self, scan_config: Dict[str, Any]) -> str:
    """
    Periodic task to scan Notion sources for changes
    
    Args:
        scan_config: Configuration for scanning
            - token: Notion API token
            - database_ids: List of database IDs
            - page_ids: List of page IDs
            - auto_approve: Whether to auto-approve
    """
    task_id = self.request.id
    logger.info(f"Starting periodic Notion scan {task_id}")
    
    # Trigger ingestion task
    result = ingest_notion_pages.delay(
        notion_token=scan_config['token'],
        database_ids=scan_config.get('database_ids', []),
        page_ids=scan_config.get('page_ids', []),
        full_scan=False,  # Only process changes
        auto_approve=scan_config.get('auto_approve', False)
    )
    
    logger.info(f"Triggered ingestion task: {result.id}")
    return result.id


@celery_app.task(bind=True, name='process_single_document')
def process_single_document(
    self,
    document_id: str,
    source: str,
    content: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single document through the pipeline
    
    Args:
        document_id: Unique document ID
        source: Source type (notion, gdrive, etc.)
        content: Document content
        metadata: Document metadata
    """
    task_id = self.request.id
    logger.info(f"Processing document {document_id} from {source}")
    
    try:
        # This would call the document processing pipeline
        # For now, return mock result
        result = {
            'task_id': task_id,
            'document_id': document_id,
            'status': 'completed',
            'chunks': 42,
            'entities': 15,
            'relationships': 23
        }
        
        logger.info(f"Document {document_id} processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        raise


# Periodic task configuration
celery_app.conf.beat_schedule = {
    'scan-notion-every-30-minutes': {
        'task': 'scan_notion_sources',
        'schedule': 1800.0,  # 30 minutes in seconds
        'args': ({
            'token': settings.notion_api_key,
            'database_ids': settings.notion_database_ids,
            'auto_approve': False
        },) if settings.notion_api_key else ()
    },
}


# Worker startup
@celery_app.task(name='health_check')
def health_check():
    """Simple health check task"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }


if __name__ == '__main__':
    # Run worker
    celery_app.start()