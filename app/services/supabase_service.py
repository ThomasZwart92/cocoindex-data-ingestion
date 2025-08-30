"""Supabase service for database operations"""
from typing import List, Optional, Dict, Any
from supabase import create_client, Client
from app.config import settings
from app.models.document import Document, DocumentState
from app.models.chunk import Chunk
from app.models.entity import Entity, EntityRelationship
from app.models.job import ProcessingJob

class SupabaseService:
    """Service for Supabase operations"""
    
    def __init__(self):
        self.client: Client = create_client(settings.supabase_url, settings.supabase_key)
    
    # Document operations
    def create_document(self, document: Document) -> Document:
        """Create a new document"""
        data = document.to_supabase_dict()
        result = self.client.table("documents").insert(data).execute()
        if result.data:
            return Document(**result.data[0])
        raise Exception("Failed to create document")
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        result = self.client.table("documents").select("*").eq("id", document_id).execute()
        if result.data:
            return Document(**result.data[0])
        return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> Document:
        """Update document"""
        # Handle enum conversions
        if 'status' in updates and isinstance(updates['status'], DocumentState):
            updates['status'] = updates['status'].value
        
        result = self.client.table("documents").update(updates).eq("id", document_id).execute()
        if result.data:
            return Document(**result.data[0])
        raise Exception("Failed to update document")
    
    def update_document_status(self, document_id: str, status: DocumentState, error: Optional[str] = None) -> Document:
        """Update document status"""
        updates = {"status": status.value}
        if error:
            updates["processing_error"] = error
        if status == DocumentState.PROCESSING:
            updates["processing_error"] = None
        return self.update_document(document_id, updates)
    
    def list_documents(self, status: Optional[DocumentState] = None, limit: int = 100, source_type: Optional[str] = None, source_id: Optional[str] = None) -> List[Document]:
        """List documents with optional filtering"""
        query = self.client.table("documents").select("*")
        
        if status:
            query = query.eq("status", status.value)
        
        if source_type:
            query = query.eq("source_type", source_type)
        
        if source_id:
            query = query.eq("source_id", source_id)
        
        result = query.limit(limit).order("created_at", desc=True).execute()
        return [Document(**doc) for doc in result.data]
    
    # Chunk operations
    def create_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Create multiple chunks"""
        data = [chunk.to_supabase_dict() for chunk in chunks]
        result = self.client.table("chunks").insert(data).execute()
        if result.data:
            return [Chunk(**chunk) for chunk in result.data]
        raise Exception("Failed to create chunks")
    
    def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document"""
        result = self.client.table("chunks").select("*").eq("document_id", document_id).order("chunk_index").execute()
        return [Chunk(**chunk) for chunk in result.data]
    
    def update_chunk(self, chunk_id: str, updates: Dict[str, Any]) -> Chunk:
        """Update a chunk"""
        result = self.client.table("chunks").update(updates).eq("id", chunk_id).execute()
        if result.data:
            return Chunk(**result.data[0])
        raise Exception("Failed to update chunk")
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        self.client.table("chunks").delete().eq("document_id", document_id).execute()
        return True
    
    # Entity operations
    def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple entities"""
        data = [entity.to_supabase_dict() for entity in entities]
        result = self.client.table("entities").insert(data).execute()
        if result.data:
            return [Entity(**entity) for entity in result.data]
        raise Exception("Failed to create entities")
    
    def get_document_entities(self, document_id: str) -> List[Entity]:
        """Get all entities for a document"""
        result = self.client.table("entities").select("*").eq("document_id", document_id).execute()
        return [Entity(**entity) for entity in result.data]
    
    def create_entity_relationships(self, relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Create entity relationships"""
        data = [rel.to_supabase_dict() for rel in relationships]
        result = self.client.table("entity_relationships").insert(data).execute()
        if result.data:
            return [EntityRelationship(**rel) for rel in result.data]
        raise Exception("Failed to create relationships")
    
    # Job operations
    def create_job(self, job: ProcessingJob) -> ProcessingJob:
        """Create a processing job"""
        data = job.to_supabase_dict()
        result = self.client.table("processing_jobs").insert(data).execute()
        if result.data:
            return ProcessingJob(**result.data[0])
        raise Exception("Failed to create job")
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        result = self.client.table("processing_jobs").select("*").eq("id", job_id).execute()
        if result.data:
            return ProcessingJob(**result.data[0])
        return None
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> ProcessingJob:
        """Update a job"""
        # Handle enum conversions
        if 'job_status' in updates:
            from app.models.job import JobStatus
            if isinstance(updates['job_status'], JobStatus):
                updates['job_status'] = updates['job_status'].value
        
        result = self.client.table("processing_jobs").update(updates).eq("id", job_id).execute()
        if result.data:
            return ProcessingJob(**result.data[0])
        raise Exception("Failed to update job")
    
    def get_document_jobs(self, document_id: str) -> List[ProcessingJob]:
        """Get all jobs for a document"""
        result = self.client.table("processing_jobs").select("*").eq("document_id", document_id).order("created_at", desc=True).execute()
        return [ProcessingJob(**job) for job in result.data]