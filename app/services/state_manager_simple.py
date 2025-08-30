"""
Simplified Document State Manager Service
Works with existing Supabase schema without requiring new tables
Tracks state transitions in document metadata
"""
import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import os

from app.models.document_state import (
    DocumentState,
    DocumentStateInfo,
    StateTransition,
    StateValidationError
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SimpleDocumentStateManager:
    """Manages document state transitions using existing Supabase schema"""
    
    def __init__(self):
        """Initialize with Supabase client"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("SimpleDocumentStateManager initialized with Supabase")
    
    def _parse_transitions_from_metadata(self, metadata: dict) -> List[StateTransition]:
        """Parse transition history from metadata"""
        transitions = []
        transition_history = metadata.get("state_transitions", [])
        
        for record in transition_history:
            try:
                transitions.append(StateTransition(
                    from_state=DocumentState(record["from_state"]),
                    to_state=DocumentState(record["to_state"]),
                    timestamp=datetime.fromisoformat(record["timestamp"]),
                    user_id=record.get("user_id"),
                    reason=record.get("reason"),
                    metadata=record.get("metadata", {})
                ))
            except Exception as e:
                logger.warning(f"Failed to parse transition: {e}")
                
        return transitions
    
    def _save_transitions_to_metadata(self, transitions: List[StateTransition]) -> List[dict]:
        """Convert transitions to storable format"""
        return [
            {
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "timestamp": t.timestamp.isoformat(),
                "user_id": t.user_id,
                "reason": t.reason,
                "metadata": t.metadata
            }
            for t in transitions
        ]
    
    async def get_state(self, document_id: str) -> Optional[DocumentStateInfo]:
        """Get current state for document"""
        try:
            # Get document from Supabase
            result = self.client.table("documents").select("*").eq("id", document_id).execute()
            
            if not result.data:
                logger.warning(f"Document {document_id} not found")
                return None
            
            doc = result.data[0]
            metadata = doc.get("metadata", {})
            
            # Parse transitions from metadata
            transitions = self._parse_transitions_from_metadata(metadata)
            
            # Build DocumentStateInfo
            state_info = DocumentStateInfo(
                document_id=document_id,
                current_state=DocumentState(doc["status"]),
                created_at=datetime.fromisoformat(doc["created_at"]),
                updated_at=datetime.fromisoformat(doc["updated_at"]),
                transition_history=transitions,
                error_count=metadata.get("error_count", 0),
                retry_count=doc.get("retry_count", 0),
                metadata=metadata
            )
            
            return state_info
            
        except Exception as e:
            logger.error(f"Error getting state for document {document_id}: {e}")
            raise
    
    async def create_state(self, 
                          document_id: str,
                          initial_state: DocumentState = DocumentState.DISCOVERED,
                          metadata: Optional[Dict[str, Any]] = None) -> DocumentStateInfo:
        """Create initial state for document"""
        try:
            # Check if document already exists
            existing = await self.get_state(document_id)
            if existing:
                logger.warning(f"Document {document_id} already has state")
                return existing
            
            # Prepare metadata with state tracking
            full_metadata = metadata or {}
            full_metadata["state_transitions"] = []
            full_metadata["error_count"] = 0
            
            # Create document record
            now = datetime.utcnow()
            doc_data = {
                "id": document_id,
                "name": full_metadata.get("filename", f"document_{document_id[:8]}"),
                "source_type": full_metadata.get("source_type", "upload"),
                "file_type": full_metadata.get("file_type", "unknown"),
                "status": initial_state.value,
                "retry_count": 0,
                "metadata": full_metadata,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "version": 1,
                "tags": []
            }
            
            result = self.client.table("documents").insert(doc_data).execute()
            
            if not result.data:
                raise Exception("Failed to create document state")
            
            # Create initial state info
            state_info = DocumentStateInfo(
                document_id=document_id,
                current_state=initial_state,
                created_at=now,
                updated_at=now,
                transition_history=[],
                error_count=0,
                retry_count=0,
                metadata=full_metadata
            )
            
            logger.info(f"Created state for document {document_id}: {initial_state.value}")
            return state_info
            
        except Exception as e:
            logger.error(f"Error creating state for document {document_id}: {e}")
            raise
    
    async def transition(self,
                        document_id: str,
                        to_state: DocumentState,
                        user_id: Optional[str] = None,
                        reason: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> DocumentStateInfo:
        """Transition document to new state"""
        try:
            # Get current state
            state_info = await self.get_state(document_id)
            if not state_info:
                # Create if doesn't exist
                state_info = await self.create_state(document_id)
            
            # Validate transition
            if not state_info.current_state.can_transition_to(to_state):
                raise StateValidationError(
                    f"Cannot transition from {state_info.current_state.value} to {to_state.value}"
                )
            
            # Create transition record
            transition = state_info.add_transition(
                to_state=to_state,
                user_id=user_id,
                reason=reason,
                metadata=metadata
            )
            
            # Update metadata with transition history
            doc_metadata = state_info.metadata.copy()
            doc_metadata["state_transitions"] = self._save_transitions_to_metadata(
                state_info.transition_history
            )
            doc_metadata["error_count"] = state_info.error_count
            
            # Update document
            update_data = {
                "status": to_state.value,
                "updated_at": transition.timestamp.isoformat(),
                "retry_count": state_info.retry_count,
                "metadata": doc_metadata
            }
            
            # Add specific timestamp fields
            if to_state == DocumentState.PROCESSING:
                update_data["processed_at"] = transition.timestamp.isoformat()
            elif to_state == DocumentState.APPROVED:
                update_data["approved_at"] = transition.timestamp.isoformat()
                # approved_by expects UUID, so don't set if user_id is string
                # Store in metadata instead
                if user_id:
                    doc_metadata["approved_by_user"] = user_id
            
            self.client.table("documents")\
                .update(update_data)\
                .eq("id", document_id)\
                .execute()
            
            logger.info(
                f"Transitioned document {document_id} from "
                f"{transition.from_state.value} to {transition.to_state.value}"
            )
            
            return state_info
            
        except StateValidationError:
            raise
        except Exception as e:
            logger.error(f"Error transitioning document {document_id}: {e}")
            raise
    
    async def get_documents_by_state(self,
                                    state: DocumentState,
                                    limit: int = 100) -> List[DocumentStateInfo]:
        """Get all documents in a specific state"""
        try:
            result = self.client.table("documents")\
                .select("*")\
                .eq("status", state.value)\
                .limit(limit)\
                .execute()
            
            documents = []
            for doc in result.data:
                # Build state info for each document
                metadata = doc.get("metadata", {})
                transitions = self._parse_transitions_from_metadata(metadata)
                
                state_info = DocumentStateInfo(
                    document_id=doc["id"],
                    current_state=DocumentState(doc["status"]),
                    created_at=datetime.fromisoformat(doc["created_at"]),
                    updated_at=datetime.fromisoformat(doc["updated_at"]),
                    transition_history=transitions,
                    error_count=metadata.get("error_count", 0),
                    retry_count=doc.get("retry_count", 0),
                    metadata=metadata
                )
                documents.append(state_info)
            
            logger.info(f"Found {len(documents)} documents in state {state.value}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents by state {state.value}: {e}")
            raise
    
    async def get_documents_requiring_action(self,
                                            limit: int = 100) -> List[DocumentStateInfo]:
        """Get documents that require user action"""
        try:
            # States that require action
            action_states = [
                DocumentState.PENDING_REVIEW.value,
                DocumentState.FAILED.value,
                DocumentState.REJECTED.value
            ]
            
            result = self.client.table("documents")\
                .select("*")\
                .in_("status", action_states)\
                .limit(limit)\
                .execute()
            
            documents = []
            for doc in result.data:
                metadata = doc.get("metadata", {})
                transitions = self._parse_transitions_from_metadata(metadata)
                
                state_info = DocumentStateInfo(
                    document_id=doc["id"],
                    current_state=DocumentState(doc["status"]),
                    created_at=datetime.fromisoformat(doc["created_at"]),
                    updated_at=datetime.fromisoformat(doc["updated_at"]),
                    transition_history=transitions,
                    error_count=metadata.get("error_count", 0),
                    retry_count=doc.get("retry_count", 0),
                    metadata=metadata
                )
                documents.append(state_info)
            
            logger.info(f"Found {len(documents)} documents requiring action")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents requiring action: {e}")
            raise
    
    async def get_state_statistics(self) -> Dict[str, int]:
        """Get count of documents in each state"""
        try:
            # Get all documents and count by state
            result = self.client.table("documents").select("status").execute()
            
            stats = {state.value: 0 for state in DocumentState}
            
            for doc in result.data:
                status = doc.get("status")
                if status in stats:
                    stats[status] += 1
            
            logger.info(f"State statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting state statistics: {e}")
            raise
    
    async def reset_failed_document(self,
                                   document_id: str,
                                   reason: str = "Manual retry") -> DocumentStateInfo:
        """Reset a failed document back to DISCOVERED state for retry"""
        try:
            state_info = await self.get_state(document_id)
            if not state_info:
                raise ValueError(f"Document {document_id} not found")
            
            # Only reset if in error state
            if not state_info.current_state.is_error:
                raise StateValidationError(
                    f"Document is not in error state (current: {state_info.current_state.value})"
                )
            
            # Transition back to DISCOVERED
            return await self.transition(
                document_id=document_id,
                to_state=DocumentState.DISCOVERED,
                reason=reason,
                metadata={"reset": True, "previous_state": state_info.current_state.value}
            )
            
        except Exception as e:
            logger.error(f"Error resetting document {document_id}: {e}")
            raise
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for monitoring"""
        try:
            # Get state statistics
            stats = await self.get_state_statistics()
            
            # Get completed documents for timing metrics
            completed_docs = await self.get_documents_by_state(DocumentState.INGESTED, limit=1000)
            
            processing_times = []
            for doc in completed_docs:
                duration = doc.get_processing_duration()
                if duration:
                    processing_times.append(duration)
            
            total_docs = sum(stats.values())
            metrics = {
                "state_counts": stats,
                "total_documents": total_docs,
                "success_rate": stats.get(DocumentState.INGESTED.value, 0) / max(total_docs, 1),
                "failure_rate": stats.get(DocumentState.FAILED.value, 0) / max(total_docs, 1),
                "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "documents_pending_review": stats.get(DocumentState.PENDING_REVIEW.value, 0),
                "documents_failed": stats.get(DocumentState.FAILED.value, 0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting processing metrics: {e}")
            raise


# Singleton instance
state_manager = None

def get_state_manager() -> SimpleDocumentStateManager:
    """Get singleton state manager instance"""
    global state_manager
    if state_manager is None:
        state_manager = SimpleDocumentStateManager()
    return state_manager