"""
Custom CocoIndex sources for document ingestion
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import cocoindex
from cocoindex.sources import Source

from app.services.supabase_service import SupabaseService
from app.connectors.notion_connector import NotionConnector
from app.connectors.google_drive_connector import GoogleDriveConnector
from app.models.document import DocumentState
from app.config import settings

logger = logging.getLogger(__name__)


class SingleDocumentSource(Source):
    """
    CocoIndex source for fetching a single document from Supabase
    """
    
    def __init__(self, document_id: str):
        """
        Initialize source with document ID
        
        Args:
            document_id: UUID of the document to fetch
        """
        super().__init__()
        self.document_id = document_id
        self.supabase = SupabaseService()
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch the document from Supabase
        
        Returns:
            List with single document dict
        """
        logger.info(f"Fetching document {self.document_id} from Supabase")
        
        # Get document from Supabase
        response = self.supabase.client.table("documents").select("*").eq(
            "id", self.document_id
        ).single().execute()
        
        if not response.data:
            raise ValueError(f"Document {self.document_id} not found")
        
        doc = response.data
        
        # Ensure content exists
        if not doc.get("content"):
            logger.warning(f"Document {self.document_id} has no content, attempting to fetch")
            doc["content"] = self._fetch_content_from_source(doc)
        
        # Convert to format expected by CocoIndex
        result = {
            "id": doc["id"],
            "name": doc["name"],
            "source_type": doc["source_type"],
            "source_id": doc["source_id"],
            "source_url": doc.get("source_url"),
            "status": doc["status"],
            "content": doc["content"],
            "metadata": doc.get("metadata", {}),
            "created_at": doc["created_at"],
            "updated_at": doc["updated_at"],
            "processed_at": doc.get("processed_at")
        }
        
        return [result]
    
    def _fetch_content_from_source(self, doc: Dict[str, Any]) -> str:
        """
        Fetch content from the original source if not available
        
        Args:
            doc: Document dict from database
            
        Returns:
            Document content as string
        """
        source_type = doc.get("source_type", "")
        source_id = doc.get("source_id", "")
        
        logger.info(f"Fetching content for {doc['id']} from {source_type}")
        
        try:
            if source_type == "notion":
                # Use NotionConnector to fetch content
                if settings.notion_api_token:
                    connector = NotionConnector(settings.notion_api_token)
                    # Fetch page content
                    # Note: This is simplified - actual implementation would need proper page ID
                    content = f"# {doc['name']}\n\nContent from Notion page {source_id}"
                    logger.warning("Notion content fetching not fully implemented")
                else:
                    logger.error("Notion API token not configured")
                    content = f"# {doc['name']}\n\nNotion content unavailable - API token missing"
                    
            elif source_type == "google_drive":
                # Use GoogleDriveConnector to fetch content
                if settings.google_service_account_path:
                    connector = GoogleDriveConnector(settings.google_service_account_path)
                    # Fetch file content
                    # Note: This is simplified - actual implementation would download and parse
                    content = f"# {doc['name']}\n\nContent from Google Drive file {source_id}"
                    logger.warning("Google Drive content fetching not fully implemented")
                else:
                    logger.error("Google service account not configured")
                    content = f"# {doc['name']}\n\nGoogle Drive content unavailable - credentials missing"
                    
            else:
                # Default placeholder content
                content = f"# {doc['name']}\n\nPlaceholder content for testing. Source: {source_type}/{source_id}"
        
        except Exception as e:
            logger.error(f"Failed to fetch content from source: {e}")
            content = f"# {doc['name']}\n\nError fetching content: {str(e)}"
        
        # Update document with fetched content
        try:
            self.supabase.client.table("documents").update({
                "content": content,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", doc["id"]).execute()
            logger.info(f"Updated document {doc['id']} with fetched content")
        except Exception as e:
            logger.error(f"Failed to update document with content: {e}")
        
        return content


class SupabaseDocumentSource(Source):
    """
    CocoIndex source for fetching multiple documents from Supabase
    """
    
    def __init__(
        self,
        status: Optional[DocumentState] = None,
        source_type: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Initialize source with optional filters
        
        Args:
            status: Filter by document status
            source_type: Filter by source type (notion, google_drive, etc)
            limit: Maximum number of documents to fetch
        """
        super().__init__()
        self.status = status
        self.source_type = source_type
        self.limit = limit or 100
        self.supabase = SupabaseService()
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch documents from Supabase with filters
        
        Returns:
            List of document dicts
        """
        logger.info(f"Fetching documents from Supabase (status={self.status}, source_type={self.source_type}, limit={self.limit})")
        
        # Build query
        query = self.supabase.client.table("documents").select("*")
        
        # Apply filters
        if self.status:
            query = query.eq("status", self.status.value)
        
        if self.source_type:
            query = query.eq("source_type", self.source_type)
        
        # Apply limit and order
        query = query.limit(self.limit).order("created_at", desc=True)
        
        # Execute query
        response = query.execute()
        documents = response.data if response.data else []
        
        logger.info(f"Fetched {len(documents)} documents from Supabase")
        
        # Convert to CocoIndex format
        result = []
        for doc in documents:
            # Ensure content exists for each document
            if not doc.get("content"):
                logger.warning(f"Document {doc['id']} has no content")
                # For batch processing, we'll skip content fetching to avoid slowdown
                # Individual document processing will handle this
                doc["content"] = ""
            
            result.append({
                "id": doc["id"],
                "name": doc["name"],
                "source_type": doc["source_type"],
                "source_id": doc["source_id"],
                "source_url": doc.get("source_url"),
                "status": doc["status"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "created_at": doc["created_at"],
                "updated_at": doc["updated_at"],
                "processed_at": doc.get("processed_at")
            })
        
        return result


class IncrementalDocumentSource(SupabaseDocumentSource):
    """
    CocoIndex source that only fetches new or updated documents
    """
    
    def __init__(
        self,
        last_sync_time: Optional[datetime] = None,
        **kwargs
    ):
        """
        Initialize incremental source
        
        Args:
            last_sync_time: Only fetch documents updated after this time
            **kwargs: Other filters passed to parent class
        """
        super().__init__(**kwargs)
        self.last_sync_time = last_sync_time
    
    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch only new or updated documents
        
        Returns:
            List of document dicts that have been updated
        """
        if self.last_sync_time:
            logger.info(f"Fetching documents updated after {self.last_sync_time}")
            
            # Build query for updated documents
            query = self.supabase.client.table("documents").select("*")
            query = query.gt("updated_at", self.last_sync_time.isoformat())
            
            # Apply other filters
            if self.status:
                query = query.eq("status", self.status.value)
            
            if self.source_type:
                query = query.eq("source_type", self.source_type)
            
            # Apply limit and order
            query = query.limit(self.limit).order("updated_at", desc=True)
            
            # Execute query
            response = query.execute()
            documents = response.data if response.data else []
            
            logger.info(f"Fetched {len(documents)} updated documents")
            
            # Convert to CocoIndex format
            result = []
            for doc in documents:
                result.append({
                    "id": doc["id"],
                    "name": doc["name"],
                    "source_type": doc["source_type"],
                    "source_id": doc["source_id"],
                    "source_url": doc.get("source_url"),
                    "status": doc["status"],
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "created_at": doc["created_at"],
                    "updated_at": doc["updated_at"],
                    "processed_at": doc.get("processed_at")
                })
            
            return result
        else:
            # No last sync time, fetch all
            return super().fetch()