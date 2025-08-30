"""
Supabase client-based database service
Uses Supabase REST API instead of direct PostgreSQL connection
This simplifies authentication and avoids connection string issues
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create Supabase client (lazy initialization for test environments)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get or create Supabase client with lazy initialization"""
    global supabase
    if supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            # In test mode, allow missing credentials
            if os.getenv("TEST_MODE") == "true":
                logger.warning("Running in TEST_MODE without Supabase credentials")
                return None
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Connected to Supabase at {SUPABASE_URL}")
    
    return supabase


@dataclass
class Document:
    """Document model matching Supabase schema"""
    id: str
    name: str
    source_type: str
    file_type: str
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    source_url: Optional[str] = None
    source_id: Optional[str] = None
    file_size: Optional[int] = None
    processing_error: Optional[str] = None
    retry_count: Optional[int] = None
    parse_tier: Optional[str] = None
    parse_confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    processed_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    version: Optional[int] = None
    parent_document_id: Optional[str] = None
    # Legacy/additional fields
    security_level: Optional[str] = None
    access_level: Optional[int] = None
    content: Optional[str] = None
    url: Optional[str] = None
    storage_path: Optional[str] = None


class SupabaseDocumentService:
    """Service for managing documents in Supabase"""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.table = "documents"
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID"""
        try:
            result = self.client.table(self.table)\
                .select("*")\
                .eq("id", document_id)\
                .execute()
            
            if result.data:
                doc_data = result.data[0]
                return Document(**doc_data)
            return None
        except Exception as e:
            logger.error(f"Error fetching document {document_id}: {e}")
            return None
    
    def create_document(self, document: Document) -> Optional[Document]:
        """Create a new document"""
        try:
            doc_dict = {
                "id": document.id,
                "name": document.name,
                "source_type": document.source_type,
                "file_type": document.file_type,
                "status": document.status,
                "metadata": document.metadata or {},
                "security_level": document.security_level,
                "access_level": document.access_level,
                "content": document.content,
                "url": document.url,
                "storage_path": document.storage_path
            }
            
            # Remove None values
            doc_dict = {k: v for k, v in doc_dict.items() if v is not None}
            
            result = self.client.table(self.table).insert(doc_dict).execute()
            
            if result.data:
                return Document(**result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document"""
        try:
            # Add updated_at timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table(self.table)\
                .update(updates)\
                .eq("id", document_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    def update_document_status(self, document_id: str, status: str) -> bool:
        """Update document status"""
        return self.update_document(document_id, {"status": status})
    
    def list_documents(self, 
                      status: Optional[str] = None,
                      source_type: Optional[str] = None,
                      limit: int = 100) -> List[Document]:
        """List documents with optional filters"""
        try:
            query = self.client.table(self.table).select("*")
            
            if status:
                query = query.eq("status", status)
            if source_type:
                query = query.eq("source_type", source_type)
            
            query = query.limit(limit)
            result = query.execute()
            
            return [Document(**doc) for doc in result.data]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        try:
            result = self.client.table(self.table)\
                .delete()\
                .eq("id", document_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False


class SupabaseChunkService:
    """Service for managing document chunks in Supabase"""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.table = "chunks"  # Correct table name in Supabase
    
    def get_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        try:
            result = self.client.table(self.table)\
                .select("*")\
                .eq("document_id", document_id)\
                .order("chunk_index")\
                .execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []
    
    def create_chunk(self, chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new chunk"""
        try:
            # Map field names to match Supabase schema
            supabase_data = {
                "document_id": chunk_data.get("document_id"),
                "chunk_index": chunk_data.get("chunk_number", chunk_data.get("chunk_index", 0)),
                "chunk_text": chunk_data.get("chunk_text", chunk_data.get("text", "")),
                "chunk_size": chunk_data.get("chunk_size", len(chunk_data.get("chunk_text", ""))),
                "chunking_strategy": chunk_data.get("chunking_strategy", "recursive"),
                "chunk_overlap": chunk_data.get("chunk_overlap", 200),
                "metadata": chunk_data.get("metadata", {})
            }
            
            # Include id if provided
            if "id" in chunk_data:
                supabase_data["id"] = chunk_data.get("id")
            
            result = self.client.table(self.table).insert(supabase_data).execute()
            
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            return None
    
    def update_chunk(self, chunk_id: str, text: str) -> bool:
        """Update chunk text"""
        try:
            result = self.client.table(self.table)\
                .update({"chunk_text": text, "updated_at": datetime.utcnow().isoformat()})\
                .eq("id", chunk_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk"""
        try:
            result = self.client.table(self.table)\
                .delete()\
                .eq("id", chunk_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False


# Singleton instances
document_service = SupabaseDocumentService()
chunk_service = SupabaseChunkService()


def test_connection():
    """Test Supabase connection"""
    try:
        client = get_supabase_client()
        if client is None:
            logger.warning("No Supabase client available (test mode)")
            return False
        # Try to fetch one document
        result = client.table("documents").select("id").limit(1).execute()
        logger.info(f"✓ Connected to Supabase successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False