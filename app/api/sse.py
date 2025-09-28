"""
Server-Sent Events (SSE) API for real-time updates
Provides live document processing status updates

Example event frames (one per line, prefixed by `data:`):

data: {"type": "processing_started", "document": {"id": "...", "status": "processing", "updated_at": "..."}, "timestamp": "..."}
data: {"type": "review_required", "document": {"id": "...", "status": "pending_review"}, "timestamp": "..."}
data: {"type": "processing_failed", "document": {"id": "...", "status": "failed"}, "timestamp": "..."}
data: {"type": "processing_complete", "document": {"id": "...", "status": "ingested"}, "timestamp": "..."}
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime
import uuid

from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sse", tags=["sse"])

# Store active SSE connections
active_connections = {}

class SSEManager:
    """Manages Server-Sent Events connections"""
    
    def __init__(self):
        self.connections = {}
        self.supabase_service = SupabaseService()
    
    async def connect(self, client_id: str):
        """Register a new SSE connection"""
        self.connections[client_id] = {
            "connected_at": datetime.now(),
            "last_ping": datetime.now()
        }
        logger.info(f"SSE client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove an SSE connection"""
        if client_id in self.connections:
            del self.connections[client_id]
            logger.info(f"SSE client {client_id} disconnected")
    
    async def send_event(self, client_id: str, event_type: str, data: dict):
        """Send an event to a specific client"""
        if client_id in self.connections:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            return f"data: {json.dumps(event)}\n\n"
        return None
    
    async def broadcast_event(self, event_type: str, data: dict):
        """Broadcast an event to all connected clients"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        message = f"data: {json.dumps(event)}\n\n"
        
        for client_id in list(self.connections.keys()):
            # Send to each client (in real implementation, would use queues)
            logger.debug(f"Broadcasting {event_type} to {client_id}")
        
        return message

# Create global SSE manager
sse_manager = SSEManager()

async def event_generator(request: Request, document_id: str = None) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for document processing updates
    """
    client_id = str(uuid.uuid4())
    await sse_manager.connect(client_id)
    
    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'client_id': client_id})}\n\n"
        
        # Track last known states
        last_states = {}
        
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            
            # Send heartbeat ping every 30 seconds
            yield f": ping\n\n"
            
            try:
                # Fetch current document states
                if document_id:
                    # Specific document updates
                    doc = sse_manager.supabase_service.get_document(document_id)
                    if doc:
                        documents = [doc.to_supabase_dict()]
                    else:
                        documents = []
                else:
                    # All documents updates
                    docs = sse_manager.supabase_service.list_documents(limit=100)
                    documents = [doc.to_supabase_dict() for doc in docs]
                
                # Check for state changes
                for doc in documents:
                    doc_id = doc.get('id')
                    current_state = doc.get('status', 'unknown')
                    
                    if doc_id not in last_states or last_states[doc_id] != current_state:
                        # State changed, send update
                        event_data = {
                            "id": doc_id,
                            "title": doc.get('title', ''),
                            "status": current_state,
                            "progress": doc.get('processing_progress', 0),
                            "source": doc.get('source_type', ''),
                            "updated_at": doc.get('updated_at', '')
                        }
                        
                        # Determine event type based on status
                        if current_state == 'processing':
                            event_type = 'processing_started'
                        elif current_state == 'complete':
                            event_type = 'processing_complete'
                        elif current_state == 'failed':
                            event_type = 'processing_failed'
                        elif current_state == 'pending_review':
                            event_type = 'review_required'
                        else:
                            event_type = 'status_update'
                        
                        yield f"data: {json.dumps({'type': event_type, 'document': event_data})}\n\n"
                        
                        last_states[doc_id] = current_state
                
            except Exception as e:
                logger.error(f"Error fetching document updates: {e}")
                # Send error event but continue
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to fetch updates'})}\n\n"
            
            # Wait before next check (adjust based on requirements)
            await asyncio.sleep(2)  # Check every 2 seconds
            
    except asyncio.CancelledError:
        logger.info(f"SSE connection {client_id} cancelled")
    except Exception as e:
        logger.error(f"SSE error for client {client_id}: {e}")
    finally:
        sse_manager.disconnect(client_id)

@router.get("/documents")
async def document_updates_stream(request: Request):
    """
    Stream real-time updates for all documents
    
    Returns Server-Sent Events with document status changes.
    """
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

@router.get("/documents/{document_id}")
async def specific_document_updates_stream(request: Request, document_id: str):
    """
    Stream real-time updates for a specific document
    
    Returns Server-Sent Events with status changes for the specified document.
    """
    return StreamingResponse(
        event_generator(request, document_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

# Helper function to send updates from other parts of the application
async def send_document_update(document_id: str, status: str, progress: int = 0, message: str = None):
    """
    Send a document update to all connected SSE clients
    
    This can be called from Celery tasks or other parts of the application
    to push real-time updates.
    """
    event_data = {
        "id": document_id,
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    # Determine event type
    if status == 'processing':
        event_type = 'processing_update'
    elif status == 'complete':
        event_type = 'processing_complete'
    elif status == 'failed':
        event_type = 'processing_failed'
    else:
        event_type = 'status_update'
    
    # In production, this would send to a message queue that SSE reads from
    logger.info(f"SSE update for document {document_id}: {status} ({progress}%)")
    
    # For now, just log it
    await sse_manager.broadcast_event(event_type, event_data)
