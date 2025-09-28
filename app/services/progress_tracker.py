"""
Progress Tracker Service for managing real-time processing progress updates
"""
import asyncio
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Singleton to manage progress update queues for documents being processed"""
    
    _instance = None
    _queues: Dict[str, asyncio.Queue] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressTracker, cls).__new__(cls)
            cls._instance._queues = {}
        return cls._instance
    
    def register_queue(self, document_id: str, queue: asyncio.Queue):
        """Register a queue for a document to receive progress updates"""
        self._queues[document_id] = queue
        logger.info(f"Registered progress queue for document {document_id}")
    
    def unregister_queue(self, document_id: str):
        """Remove a queue when connection closes"""
        if document_id in self._queues:
            del self._queues[document_id]
            logger.info(f"Unregistered progress queue for document {document_id}")
    
    async def send_progress(self, document_id: str, progress_data: dict):
        """Send progress update to the queue if one exists for this document"""
        if document_id in self._queues:
            try:
                await self._queues[document_id].put(progress_data)
                logger.debug(f"Sent progress update for document {document_id}: {progress_data}")
            except Exception as e:
                logger.error(f"Error sending progress update for document {document_id}: {e}")
    
    def has_listener(self, document_id: str) -> bool:
        """Check if there's a listener for this document"""
        return document_id in self._queues