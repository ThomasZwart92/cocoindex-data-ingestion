"""
Notion Ingestion Pipeline
End-to-end pipeline from Notion to vector/graph databases
"""
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from app.connectors.notion_connector import NotionConnector
from app.services.document_processor import DocumentProcessor
from app.services.llm_service import LLMService
from app.services.neo4j_service import Neo4jService
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from app.services.relationship_extractor import RelationshipExtractor
from app.models.document import Document, DocumentChunk, DocumentMetadata
from app.models.entities import Entity
from app.models.relationships import RelationshipModel
from app.database import get_db
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionStatus:
    """Track ingestion progress"""
    total_pages: int = 0
    processed_pages: int = 0
    failed_pages: int = 0
    new_chunks: int = 0
    new_entities: int = 0
    new_relationships: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class NotionIngestionPipeline:
    """Complete pipeline for ingesting Notion content"""
    
    def __init__(
        self,
        notion_token: str = None,
        security_level: str = "employee",
        database_ids: Optional[List[str]] = None,
        page_ids: Optional[List[str]] = None
    ):
        # Handle multiple security levels
        if notion_token:
            self.notion_token = notion_token
            self.security_level = self._infer_security_level(notion_token)
        else:
            # Use token for specified security level
            self.notion_token = settings.notion_tokens.get(security_level)
            self.security_level = security_level
            
        if not self.notion_token:
            raise ValueError(f"No Notion token available for security level: {security_level}")
        
        self.notion = NotionConnector(self.notion_token)
        self.processor = DocumentProcessor()
        self.llm = LLMService()
        self.neo4j = Neo4jService()
        self.qdrant = QdrantService()
        self.embedder = EmbeddingService()
        self.relationship_extractor = RelationshipExtractor()
        
        self.database_ids = database_ids or []
        self.page_ids = page_ids or []
        
        self.status = IngestionStatus()
        
        logger.info(f"Initialized Notion pipeline with security level: {self.security_level}")
    
    def _infer_security_level(self, token: str) -> str:
        """Infer security level from token"""
        for level, configured_token in settings.notion_tokens.items():
            if configured_token and configured_token == token:
                return level
        return "unknown"
    
    async def run(
        self,
        full_scan: bool = False,
        auto_approve: bool = False
    ) -> IngestionStatus:
        """
        Run the complete ingestion pipeline
        
        Args:
            full_scan: Process all pages regardless of changes
            auto_approve: Skip review queue and auto-approve high confidence
        """
        logger.info(f"Starting Notion ingestion pipeline (full_scan={full_scan})")
        
        try:
            # Connect to databases
            await self._connect_services()
            
            # Fetch pages from Notion
            pages = await self._fetch_pages(full_scan)
            self.status.total_pages = len(pages)
            
            logger.info(f"Found {len(pages)} pages to process")
            
            # Process each page
            for page in pages:
                try:
                    await self._process_page(page, auto_approve)
                    self.status.processed_pages += 1
                except Exception as e:
                    logger.error(f"Failed to process page {page.get('id')}: {e}")
                    self.status.failed_pages += 1
                    self.status.errors.append(f"Page {page.get('id')}: {str(e)}")
            
            logger.info(f"Ingestion complete: {self.status.processed_pages}/{self.status.total_pages} processed")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.status.errors.append(f"Pipeline error: {str(e)}")
        
        return self.status
    
    async def _connect_services(self):
        """Initialize service connections"""
        await self.neo4j.connect()
        await self.neo4j.ensure_constraints()
        
        # Ensure Qdrant collection exists
        try:
            await self.qdrant.create_collection("documents")
        except:
            pass  # Collection might already exist
    
    async def _fetch_pages(self, full_scan: bool) -> List[Dict]:
        """Fetch pages from Notion"""
        # Get last scan time if not full scan
        modified_since = None
        if not full_scan:
            # Get from database (simplified - you'd query your state DB)
            from datetime import timezone
            modified_since = datetime.now(timezone.utc) - timedelta(days=7)
        
        pages = await self.notion.get_workspace_pages(
            database_ids=self.database_ids,
            page_ids=self.page_ids,
            modified_since=modified_since
        )
        
        return pages
    
    async def _process_page(self, page: Dict, auto_approve: bool):
        """Process a single Notion page"""
        page_id = page["id"]
        logger.info(f"Processing page: {page.get('title', 'Untitled')} ({page_id})")
        
        # 1. Create document record
        document = await self._create_document(page)
        
        # 2. Extract and process chunks
        chunks = await self._process_chunks(document, page["content"])
        self.status.new_chunks += len(chunks)
        
        # 3. Extract entities
        entities = await self._extract_entities(document, page["content"])
        self.status.new_entities += len(entities)
        
        # 4. Extract relationships
        relationships = await self._extract_relationships(
            document, 
            page["content"], 
            entities
        )
        self.status.new_relationships += len(relationships)
        
        # 5. Store in vector database
        await self._store_vectors(document, chunks)
        
        # 6. Store in graph database
        await self._store_graph(document, entities, relationships)
        
        # 7. Update document status
        await self._update_document_status(document, "ingested" if auto_approve else "pending_review")
        
        logger.info(f"Successfully processed page {page_id}: {len(chunks)} chunks, {len(entities)} entities")
    
    async def _create_document(self, page: Dict) -> Document:
        """Create document record"""
        # Calculate content hash
        content_hash = hashlib.sha256(page["content"].encode()).hexdigest()
        
        # Extract metadata from page properties
        metadata = DocumentMetadata(
            title=page.get("title", "Untitled"),
            author=page.get("created_by", {}).get("name"),
            source="notion",
            source_id=page["id"],
            created_at=page.get("created_time"),
            updated_at=page.get("last_edited_time"),
            department=self._infer_department(page),
            tags=self._extract_tags(page),
            security_level=self.security_level,
            access_level=settings.security_levels.get(self.security_level, 0)
        )
        
        document = Document(
            id=f"notion_{page['id']}",
            content=page["content"],
            content_hash=content_hash,
            metadata=metadata,
            source_type="notion",
            source_id=page["id"],
            status="processing"
        )
        
        # Save to database
        async with get_db() as db:
            db.add(document)
            await db.commit()
        
        return document
    
    async def _process_chunks(self, document: Document, content: str) -> List[DocumentChunk]:
        """Process document into chunks with embeddings"""
        # Chunk the document
        chunks = await self.processor.chunk_document(
            content=content,
            method="recursive",
            chunk_size=1500,
            chunk_overlap=200,
            language="markdown"
        )
        
        # Generate embeddings for each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await self.embedder.embed_text(chunk["text"])
            
            doc_chunk = DocumentChunk(
                document_id=document.id,
                chunk_number=i,
                text=chunk["text"],
                embedding=embedding,
                metadata={
                    "start": chunk.get("start", 0),
                    "end": chunk.get("end", len(chunk["text"]))
                }
            )
            processed_chunks.append(doc_chunk)
        
        return processed_chunks
    
    async def _extract_entities(self, document: Document, content: str) -> List[Entity]:
        """Extract entities from content"""
        # Use LLM to extract entities
        entities = await self.llm.extract_entities(content)
        
        # Add document context to entities
        for entity in entities:
            entity.document_id = document.id
            entity.source = "notion"
        
        return entities
    
    async def _extract_relationships(
        self,
        document: Document,
        content: str,
        entities: List[Entity]
    ) -> List[RelationshipModel]:
        """Extract relationships between entities"""
        relationships = await self.relationship_extractor.extract_relationships(
            text=content,
            entities=entities,
            document_metadata=document.metadata
        )
        
        # Add document context
        for rel in relationships:
            rel.document_id = document.id
        
        return relationships
    
    async def _store_vectors(self, document: Document, chunks: List[DocumentChunk]):
        """Store chunks with embeddings in Qdrant"""
        points = []
        
        for chunk in chunks:
            point = {
                "id": f"{document.id}_chunk_{chunk.chunk_number}",
                "vector": chunk.embedding,
                "payload": {
                    "document_id": document.id,
                    "chunk_number": chunk.chunk_number,
                    "text": chunk.text,
                    "title": document.metadata.title,
                    "source": "notion",
                    "department": document.metadata.department,
                    "tags": document.metadata.tags,
                    "security_level": self.security_level,
                    "access_level": settings.security_levels.get(self.security_level, 0),
                    "created_at": document.metadata.created_at,
                    "location": f"chunk_{chunk.chunk_number}"
                }
            }
            points.append(point)
        
        # Batch insert into Qdrant
        await self.qdrant.upsert(
            collection_name="documents",
            points=points
        )
        
        logger.info(f"Stored {len(points)} vectors in Qdrant")
    
    async def _store_graph(
        self,
        document: Document,
        entities: List[Entity],
        relationships: List[RelationshipModel]
    ):
        """Store entities and relationships in Neo4j"""
        # Create document node
        await self.neo4j.create_document(document)
        
        # Create entity nodes
        if entities:
            await self.neo4j.create_entities(entities)
        
        # Create relationships
        if relationships:
            await self.neo4j.create_typed_relationships(relationships)
        
        logger.info(f"Stored {len(entities)} entities and {len(relationships)} relationships in Neo4j")
    
    async def _update_document_status(self, document: Document, status: str):
        """Update document processing status"""
        document.status = status
        document.ingested_at = datetime.now() if status == "ingested" else None
        
        async with get_db() as db:
            db.add(document)
            await db.commit()
    
    def _infer_department(self, page: Dict) -> Optional[str]:
        """Infer department from page properties or path"""
        # Check page properties
        props = page.get("properties", {})
        if "Department" in props:
            return props["Department"].get("select", {}).get("name")
        
        # Check parent database name
        parent = page.get("parent", {})
        if parent.get("type") == "database_id":
            db_title = parent.get("database_title", "").lower()
            for dept in ["engineering", "support", "sales", "marketing", 
                        "operations", "product", "hr", "finance"]:
                if dept in db_title:
                    return dept.title()
        
        return None
    
    def _extract_tags(self, page: Dict) -> List[str]:
        """Extract tags from page properties"""
        tags = []
        props = page.get("properties", {})
        
        # Check for Tags property
        if "Tags" in props:
            tag_data = props["Tags"].get("multi_select", [])
            tags.extend([tag["name"] for tag in tag_data])
        
        # Check for Category
        if "Category" in props:
            category = props["Category"].get("select", {}).get("name")
            if category:
                tags.append(category)
        
        return tags


async def run_notion_ingestion(
    token: str = None,
    security_level: str = "employee",
    database_ids: Optional[List[str]] = None,
    page_ids: Optional[List[str]] = None,
    full_scan: bool = False,
    auto_approve: bool = False
) -> IngestionStatus:
    """
    Run Notion ingestion pipeline
    
    Args:
        token: Notion API token (optional, will use security_level if not provided)
        security_level: Security level to use (public, employee, department, executive, admin)
        database_ids: List of database IDs to ingest
        page_ids: List of specific page IDs to ingest
        full_scan: Process all pages regardless of changes
        auto_approve: Auto-approve high confidence results
    """
    pipeline = NotionIngestionPipeline(
        notion_token=token,
        security_level=security_level,
        database_ids=database_ids,
        page_ids=page_ids
    )
    
    return await pipeline.run(full_scan=full_scan, auto_approve=auto_approve)


# Test function
async def test_notion_pipeline():
    """Test the Notion ingestion pipeline"""
    print("\n" + "="*60)
    print("NOTION INGESTION PIPELINE TEST")
    print("="*60)
    
    # You'll need to set these
    NOTION_TOKEN = settings.notion_api_key or input("Enter Notion API token: ")
    
    # Optional: specific databases or pages
    database_ids = []  # Add your database IDs
    page_ids = []  # Add specific page IDs
    
    if not database_ids and not page_ids:
        print("\n⚠️  No database or page IDs specified.")
        print("The pipeline will attempt to fetch all accessible pages.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\nStarting ingestion...")
    status = await run_notion_ingestion(
        token=NOTION_TOKEN,
        database_ids=database_ids,
        page_ids=page_ids,
        full_scan=True,  # Process all pages
        auto_approve=False  # Manual review
    )
    
    print("\n" + "="*60)
    print("INGESTION RESULTS")
    print("="*60)
    print(f"Total pages: {status.total_pages}")
    print(f"Processed: {status.processed_pages}")
    print(f"Failed: {status.failed_pages}")
    print(f"New chunks: {status.new_chunks}")
    print(f"New entities: {status.new_entities}")
    print(f"New relationships: {status.new_relationships}")
    
    if status.errors:
        print(f"\n❌ Errors:")
        for error in status.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    if status.processed_pages > 0:
        print(f"\n✅ Successfully processed {status.processed_pages} pages!")
        print("You can now search the ingested content using the search API.")
    else:
        print("\n⚠️  No pages were processed.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_notion_pipeline())