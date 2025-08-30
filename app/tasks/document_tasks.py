"""Document processing tasks"""
import logging
from typing import List, Dict, Any
from celery import Task, chain, group
from app.celery_app import celery_app
from app.models.document import Document, DocumentState, ParseTier
from app.models.chunk import Chunk, ChunkingStrategy
from app.models.job import ProcessingJob, JobType, JobStatus
from app.services.supabase_service import SupabaseService
from app.processors.parser import DocumentParser
from app.processors.chunker import DocumentChunker
from app.processors.embedder import EmbeddingGenerator
from app.processors.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

class DocumentTask(Task):
    """Base task with database connections"""
    _supabase = None
    
    @property
    def supabase(self):
        if self._supabase is None:
            self._supabase = SupabaseService()
        return self._supabase

@celery_app.task(
    bind=True, 
    base=DocumentTask, 
    name="process_document",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 5},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True
)
def process_document(self, document_id: str, job_id: str) -> Dict[str, Any]:
    """
    Main document processing pipeline
    Orchestrates: parse -> chunk -> embed -> extract entities
    
    Retry policy:
    - Max 3 retries with exponential backoff
    - Initial delay: 5 seconds
    - Max delay: 600 seconds (10 minutes)
    - Adds jitter to prevent thundering herd
    """
    logger.info(f"Starting document processing for {document_id} (attempt {self.request.retries + 1})")
    
    try:
        # Get document and job
        document = self.supabase.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        job = self.supabase.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Update job status
        job.start()
        self.supabase.update_job(job_id, job.to_supabase_dict())
        
        # Update document status
        self.supabase.update_document_status(document_id, DocumentState.PROCESSING)
        
        # Create processing chain
        processing_chain = chain(
            parse_document.s(document_id, job_id),
            chunk_document.s(job_id),
            generate_embeddings.s(job_id),
            extract_entities.s(job_id)
        )
        
        # Execute chain
        result = processing_chain.apply_async()
        
        # Wait for completion
        final_result = result.get(timeout=3600)  # 1 hour timeout
        
        # Update document status to pending review
        self.supabase.update_document_status(document_id, DocumentState.PENDING_REVIEW)
        
        # Complete job
        job.complete()
        self.supabase.update_job(job_id, job.to_supabase_dict())
        
        logger.info(f"Document processing completed for {document_id}")
        return {
            "document_id": document_id,
            "job_id": job_id,
            "status": "completed",
            "result": final_result
        }
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {str(e)}")
        
        # Update document status
        self.supabase.update_document_status(
            document_id, 
            DocumentState.FAILED,
            error=str(e)
        )
        
        # Fail job
        if job:
            job.fail(str(e))
            self.supabase.update_job(job_id, job.to_supabase_dict())
        
        raise

@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="parse_document",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 10},
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def parse_document(self, document_id: str, job_id: str) -> Dict[str, Any]:
    """Parse document using LlamaParse with automatic retry"""
    logger.info(f"Parsing document {document_id} (attempt {self.request.retries + 1})")
    
    try:
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 10,
            "current_step": "Parsing document with LlamaParse"
        })
        
        # Get document
        document = self.supabase.get_document(document_id)
        
        # Parse document
        parser = DocumentParser()
        parse_result = parser.parse(
            document_path=document.source_url or "",
            document_name=document.name,
            parse_tier=ParseTier.BALANCED  # Start with balanced
        )
        
        # Update document with parse results
        self.supabase.update_document(document_id, {
            "parse_tier": ParseTier.BALANCED.value,
            "parse_confidence": parse_result.get("confidence", 0.8),
            "metadata": {
                **document.metadata,
                "parsed_text_length": len(parse_result["text"]),
                "parse_metadata": parse_result.get("metadata", {})
            }
        })
        
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 25,
            "current_step": "Document parsed successfully"
        })
        
        return {
            "document_id": document_id,
            "text": parse_result["text"],
            "metadata": parse_result.get("metadata", {}),
            "images": parse_result.get("images", [])
        }
        
    except Exception as e:
        logger.error(f"Document parsing failed: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="chunk_document",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 2, 'countdown': 5},
    retry_backoff=True
)
def chunk_document(self, parse_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Chunk parsed document text with automatic retry"""
    document_id = parse_result["document_id"]
    text = parse_result["text"]
    
    logger.info(f"Chunking document {document_id} (attempt {self.request.retries + 1})")
    
    try:
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 35,
            "current_step": "Chunking document"
        })
        
        # Delete existing chunks if any
        self.supabase.delete_document_chunks(document_id)
        
        # Create chunks using recursive strategy
        chunker = DocumentChunker()
        chunk_texts = chunker.chunk(
            text=text,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Create chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = Chunk(
                document_id=document_id,
                chunk_index=i,
                chunk_text=chunk_text,
                chunking_strategy=ChunkingStrategy.RECURSIVE,
                chunk_overlap=50
            )
            chunks.append(chunk)
        
        # Save chunks to database
        saved_chunks = self.supabase.create_chunks(chunks)
        
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 50,
            "current_step": f"Created {len(saved_chunks)} chunks"
        })
        
        logger.info(f"Created {len(saved_chunks)} chunks for document {document_id}")
        
        return {
            "document_id": document_id,
            "chunk_ids": [c.id for c in saved_chunks],
            "chunk_count": len(saved_chunks),
            "metadata": parse_result.get("metadata", {}),
            "images": parse_result.get("images", [])
        }
        
    except Exception as e:
        logger.error(f"Document chunking failed: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="generate_embeddings",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 10},
    retry_backoff=True,
    retry_backoff_max=300
)
def generate_embeddings(self, chunk_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Generate embeddings for chunks with automatic retry"""
    document_id = chunk_result["document_id"]
    chunk_ids = chunk_result["chunk_ids"]
    
    logger.info(f"Generating embeddings for {len(chunk_ids)} chunks (attempt {self.request.retries + 1})")
    
    try:
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 60,
            "current_step": "Generating embeddings"
        })
        
        # Get chunks
        chunks = self.supabase.get_document_chunks(document_id)
        
        # Generate embeddings
        embedder = EmbeddingGenerator()
        embedding_results = embedder.generate_embeddings(chunks)
        
        # Update chunks with embedding IDs
        for chunk, embedding_id in zip(chunks, embedding_results):
            self.supabase.update_chunk(chunk.id, {
                "embedding_id": embedding_id,
                "embedding_model": "text-embedding-3-small"
            })
        
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 75,
            "current_step": f"Generated {len(embedding_results)} embeddings"
        })
        
        logger.info(f"Generated {len(embedding_results)} embeddings")
        
        return {
            **chunk_result,
            "embeddings_generated": len(embedding_results)
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="extract_entities",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 15},
    retry_backoff=True,
    retry_backoff_max=300
)
def extract_entities(self, embedding_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Extract entities from document with automatic retry"""
    document_id = embedding_result["document_id"]
    
    logger.info(f"Extracting entities for document {document_id} (attempt {self.request.retries + 1})")
    
    try:
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 85,
            "current_step": "Extracting entities"
        })
        
        # Get chunks for entity extraction
        chunks = self.supabase.get_document_chunks(document_id)
        
        # Extract entities
        extractor = EntityExtractor()
        entities, relationships = extractor.extract(chunks, document_id)
        
        # Save entities
        if entities:
            saved_entities = self.supabase.create_entities(entities)
            logger.info(f"Saved {len(saved_entities)} entities")
        
        # Save relationships
        if relationships:
            saved_relationships = self.supabase.create_entity_relationships(relationships)
            logger.info(f"Saved {len(saved_relationships)} relationships")
        
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 95,
            "current_step": f"Extracted {len(entities)} entities and {len(relationships)} relationships"
        })
        
        return {
            **embedding_result,
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships)
        }
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        raise