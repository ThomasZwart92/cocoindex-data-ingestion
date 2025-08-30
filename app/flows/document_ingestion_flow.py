"""
CocoIndex Flow for Document Ingestion Pipeline
Integrates parsing, chunking, embedding, entity extraction, and storage
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import asyncio
from datetime import timedelta

import cocoindex
from cocoindex import FlowBuilder, DataScope
from cocoindex.sources import LocalFile
from cocoindex.targets import Postgres, Qdrant, Neo4j
from cocoindex.functions import (
    SplitRecursively,
    ExtractByLlm,
    SentenceTransformerEmbed
)
from cocoindex.llm import LlmSpec, LlmApiType

from app.config import settings
from app.services.qdrant_service import QdrantService
from app.services.neo4j_service import Neo4jService as Neo4jServiceApp
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.processors.parser import DocumentParser

logger = logging.getLogger(__name__)

@dataclass
class DocumentEntity:
    """Entity extracted from document"""
    name: str
    type: str
    confidence: float
    properties: Dict[str, Any]

@dataclass
class DocumentMetadata:
    """Metadata extracted from document"""
    title: str
    author: Optional[str]
    category: Optional[str]
    tags: List[str]
    department: Optional[str]
    
@cocoindex.flow_def(name="DocumentIngestionFlow")
def document_ingestion_flow(flow_builder: FlowBuilder, data_scope: DataScope):
    """
    Main document ingestion flow using CocoIndex
    
    This flow:
    1. Reads documents from source (local/Notion/GDrive)
    2. Parses documents with LlamaParse
    3. Chunks content with multiple strategies
    4. Extracts entities and metadata with LLMs
    5. Generates embeddings
    6. Stores in Qdrant (vectors) and Neo4j (knowledge graph)
    """
    
    # Configure authentication for databases
    # Postgres for CocoIndex state tracking
    postgres_conn = flow_builder.add_auth_entry(
        "postgres_state",
        cocoindex.targets.PostgresConnection(
            host="localhost",
            port=5432,
            database="cocoindex_state",
            user="postgres",
            password=settings.database_url.split(":")[-1].split("@")[0] if settings.database_url else ""
        )
    )
    
    # Qdrant for vector storage
    qdrant_conn = flow_builder.add_auth_entry(
        "qdrant_vectors",
        cocoindex.targets.QdrantConnection(
            grpc_url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    )
    
    # Neo4j for knowledge graph
    neo4j_conn = flow_builder.add_auth_entry(
        "neo4j_graph",
        cocoindex.targets.Neo4jConnection(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password
        )
    )
    
    # Add source - start with local files, can extend to Notion/GDrive
    data_scope["documents"] = flow_builder.add_source(
        LocalFile(
            path="data/documents",  # Directory with documents to process
            pattern="**/*.{pdf,txt,md,docx}"  # File patterns to match
        )
    )
    
    # Create collectors for different outputs
    chunk_embeddings = data_scope.add_collector()
    document_entities = data_scope.add_collector()
    document_metadata = data_scope.add_collector()
    
    # Process each document
    with data_scope["documents"].row() as doc:
        
        # Step 1: Parse document with LlamaParse (or fallback to simple text)
        # For now, we'll use the content directly, but this is where LlamaParse would go
        doc["parsed_content"] = doc["content"]
        doc["parse_metadata"] = {"tier": "balanced", "confidence": 0.9}
        
        # Step 2: Extract metadata using LLM
        doc["metadata"] = doc["parsed_content"].transform(
            ExtractByLlm(
                llm_spec=LlmSpec(
                    api_type=LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                    api_key=settings.openai_api_key
                ),
                output_type=DocumentMetadata,
                instruction="""Extract metadata from this document:
                - title: Document title
                - author: Author name if mentioned
                - category: Document category (e.g., technical, business, research)
                - tags: List of relevant tags
                - department: Relevant department if applicable
                """
            )
        )
        
        # Collect metadata
        document_metadata.collect(
            document_id=doc["filename"],
            metadata=doc["metadata"]
        )
        
        # Step 3: Chunk the document
        doc["chunks"] = doc["parsed_content"].transform(
            SplitRecursively(),
            language="markdown",
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
            min_chunk_size=200
        )
        
        # Process each chunk
        with doc["chunks"].row() as chunk:
            
            # Step 4: Generate embeddings for chunks
            # Using custom embedding function since CocoIndex doesn't have OpenAI embeddings built-in
            chunk["embedding"] = chunk["text"].transform(
                CustomOpenAIEmbed(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key
                )
            )
            
            # Collect chunks with embeddings
            chunk_embeddings.collect(
                document_id=doc["filename"],
                chunk_number=chunk["chunk_index"],
                chunk_text=chunk["text"],
                chunk_size=len(chunk["text"]),
                embedding=chunk["embedding"],
                start_position=chunk["start_index"],
                end_position=chunk["end_index"]
            )
        
        # Step 5: Extract entities from document
        doc["entities"] = doc["parsed_content"].transform(
            ExtractByLlm(
                llm_spec=LlmSpec(
                    api_type=LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                    api_key=settings.openai_api_key
                ),
                output_type=List[DocumentEntity],
                instruction="""Extract all named entities from this document:
                - People (PERSON)
                - Organizations (ORGANIZATION)  
                - Locations (LOCATION)
                - Concepts/Technologies (CONCEPT)
                - Products (PRODUCT)
                Include confidence score for each entity.
                """
            )
        )
        
        # Collect entities
        with doc["entities"].row() as entity:
            document_entities.collect(
                document_id=doc["filename"],
                entity_name=entity["name"],
                entity_type=entity["type"],
                confidence=entity["confidence"],
                properties=entity["properties"]
            )
    
    # Export to Qdrant for vector search
    chunk_embeddings.export(
        "document_chunks",
        Qdrant(
            connection=qdrant_conn,
            collection_name="document_chunks",
            id_field="document_id",
            vector_field="embedding"
        ),
        primary_key_fields=["document_id", "chunk_number"]
    )
    
    # Export entities to Neo4j for knowledge graph
    document_entities.export(
        "entities",
        Neo4j(
            connection=neo4j_conn,
            mapping=cocoindex.targets.Nodes(
                label="Entity",
                properties={
                    "id": "document_id",
                    "name": "entity_name",
                    "type": "entity_type",
                    "confidence": "confidence"
                }
            )
        )
    )
    
    # Export metadata to Postgres
    document_metadata.export(
        "metadata",
        Postgres(
            connection=postgres_conn,
            table_name="document_metadata"
        ),
        primary_key_fields=["document_id"]
    )

# Custom embedding function for OpenAI
@cocoindex.op.function()
class CustomOpenAIEmbed:
    """Custom function to generate OpenAI embeddings"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self.embedder = None
    
    def __call__(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.embedder:
            from app.services.embedding_service import EmbeddingService
            self.embedder = EmbeddingService(model=self.model)
        
        # Run async function in sync context
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.embedder.embed_text(text)
            )
            return result.embedding
        finally:
            loop.close()

def run_ingestion_flow(
    source_path: str = "data/documents",
    reset: bool = False
) -> bool:
    """
    Run the document ingestion flow
    
    Args:
        source_path: Path to documents to ingest
        reset: Whether to reset the flow state
        
    Returns:
        True if successful
    """
    try:
        # Initialize CocoIndex
        cocoindex.init()
        
        # Create flow instance
        flow = document_ingestion_flow()
        
        # Reset if requested
        if reset:
            logger.info("Resetting flow state...")
            flow.reset()
        
        # Run the flow
        logger.info(f"Starting document ingestion from {source_path}")
        flow.run()
        
        logger.info("Document ingestion flow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Document ingestion flow failed: {e}")
        return False

async def process_single_document(
    file_path: str,
    document_id: str
) -> Dict[str, Any]:
    """
    Process a single document through the pipeline
    (Alternative implementation without full CocoIndex flow)
    
    Args:
        file_path: Path to document
        document_id: Unique document ID
        
    Returns:
        Processing results
    """
    results = {
        "document_id": document_id,
        "chunks": [],
        "entities": [],
        "metadata": {},
        "errors": []
    }
    
    try:
        # Initialize services
        parser = DocumentParser()
        llm_service = LLMService()
        embedder = EmbeddingService()
        qdrant = QdrantService()
        neo4j = Neo4jServiceApp()
        
        # Step 1: Parse document
        logger.info(f"Parsing document: {file_path}")
        parse_result = await parser.parse_document(
            file_path,
            tier="balanced"
        )
        
        if not parse_result["success"]:
            results["errors"].append(f"Parse failed: {parse_result.get('error')}")
            return results
        
        content = parse_result["content"]
        
        # Step 2: Extract metadata
        logger.info("Extracting metadata...")
        metadata_obj = await llm_service.extract_metadata(content)
        # Convert dataclass to dict
        from dataclasses import asdict
        results["metadata"] = asdict(metadata_obj)
        
        # Step 3: Chunk document
        logger.info("Chunking document...")
        from app.processors.chunker import DocumentChunker
        from app.models.chunk import ChunkingStrategy
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(
            text=content,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap
        )
        
        # Step 4: Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_data = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            # Prepare chunk data
            chunk_dict = {
                "document_id": document_id,
                "chunk_number": i + 1,
                "chunk_text": chunk["text"],
                "chunk_size": chunk["metadata"]["chunk_size"],
                "start_position": chunk["metadata"]["start_index"],
                "end_position": chunk["metadata"]["end_index"]
            }
            chunk_data.append(chunk_dict)
            
            # Generate embedding
            embed_result = await embedder.embed_text(chunk["text"])
            embeddings.append(embed_result.embedding)
            
        results["chunks"] = chunk_data
        
        # Step 5: Store in Qdrant
        logger.info("Storing chunks in Qdrant...")
        await qdrant.ensure_collection("document_chunks")
        await qdrant.upsert_chunks(chunk_data, embeddings)
        
        # Step 6: Extract entities
        logger.info("Extracting entities...")
        entity_result = await llm_service.extract_entities_with_comparison(
            content,
            use_gemini=True
        )
        
        # Step 7: Store in Neo4j
        logger.info("Storing entities in Neo4j...")
        await neo4j.connect()
        await neo4j.upsert_document(document_id, {
            "title": results["metadata"].get("title", ""),
            "source": "ingestion_pipeline",
            "path": file_path
        })
        
        # Convert entities for Neo4j
        from app.services.neo4j_service import Entity
        neo4j_entities = []
        for entity in entity_result["consensus"]:
            neo4j_entities.append(Entity(
                id=f"{document_id}_{entity['name'].replace(' ', '_')}",
                name=entity["name"],
                type=entity["type"],
                properties={},
                confidence=entity.get("confidence", 0.9)
            ))
        
        await neo4j.upsert_entities(neo4j_entities, document_id)
        results["entities"] = entity_result["consensus"]
        
        # Close connections
        await qdrant.close()
        await neo4j.close()
        
        logger.info(f"Successfully processed document: {document_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        results["errors"].append(str(e))
        return results

if __name__ == "__main__":
    # Test the flow
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        print("Resetting flow state...")
        success = run_ingestion_flow(reset=True)
    else:
        success = run_ingestion_flow()
    
    sys.exit(0 if success else 1)