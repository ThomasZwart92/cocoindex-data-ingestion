"""
CocoIndex Flow for Document Processing with Hybrid Approach
Integrates with existing API structure while leveraging CocoIndex benefits
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

import cocoindex
from cocoindex import FlowBuilder, DataScope

from app.services.supabase_service import SupabaseService
from app.services.qdrant_service import QdrantService
from app.services.neo4j_service import Neo4jService
from app.models.document import DocumentState
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity extracted from document"""
    name: str
    type: str
    confidence: float
    properties: Dict[str, Any] = None


@dataclass  
class Relationship:
    """Relationship between entities"""
    source: str
    target: str
    type: str
    confidence: float


@cocoindex.op.function()
def fetch_document_from_supabase(document_id: str) -> Dict[str, Any]:
    """
    Custom function to fetch a single document from Supabase
    
    Args:
        document_id: UUID of the document to fetch
        
    Returns:
        Document dict with content
    """
    logger.info(f"Fetching document {document_id} from Supabase")
    
    supabase = SupabaseService()
    
    # Get document from Supabase
    response = supabase.client.table("documents").select("*").eq(
        "id", document_id
    ).single().execute()
    
    if not response.data:
        raise ValueError(f"Document {document_id} not found")
    
    doc = response.data
    
    # Ensure content exists
    if not doc.get("content"):
        logger.warning(f"Document {document_id} has no content, using placeholder")
        doc["content"] = f"# {doc.get('name', 'Untitled')}\n\nPlaceholder content for testing"
        
        # Update document with placeholder content
        supabase.client.table("documents").update({
            "content": doc["content"],
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", document_id).execute()
    
    return doc


@cocoindex.op.function()
def parse_with_llamaparse(content: str, document_type: str = "pdf") -> str:
    """
    Parse document content using LlamaParse if needed
    
    Args:
        content: Raw document content
        document_type: Type of document (pdf, docx, etc)
        
    Returns:
        Parsed content as markdown
    """
    # For now, just return the content as-is
    # In production, this would use LlamaParse for PDFs and other formats
    logger.info(f"Parsing document of type {document_type}")
    return content


@cocoindex.op.function()
def filter_entities(entities: List[Entity]) -> List[Entity]:
    """
    Filter out low-quality entities
    
    Args:
        entities: List of extracted entities
        
    Returns:
        Filtered list of high-quality entities
    """
    # Common words to filter out (stop words and generic terms)
    stop_words = {
        'the', 'this', 'that', 'these', 'those', 'how', 'when', 'where', 'why', 'what',
        'who', 'which', 'can', 'could', 'would', 'should', 'may', 'might', 'must',
        'will', 'shall', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'and', 'or', 'but',
        'if', 'then', 'else', 'for', 'to', 'from', 'with', 'without', 'by', 'at',
        'in', 'on', 'up', 'down', 'out', 'off', 'over', 'under', 'between',
        'through', 'during', 'before', 'after', 'above', 'below', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also',
        'user', 'users', 'system', 'systems', 'data', 'information', 'document',
        'file', 'files', 'item', 'items', 'thing', 'things', 'way', 'ways'
    }
    
    filtered = []
    for entity in entities:
        # Skip if name is too short
        if len(entity.name) < 2:
            continue
        
        # Skip if it's a stop word
        if entity.name.lower() in stop_words:
            continue
        
        # Skip if confidence is too low
        if entity.confidence < 0.5:
            continue
        
        # Skip single letters or numbers
        if len(entity.name) == 1 and (entity.name.isalpha() or entity.name.isdigit()):
            continue
        
        filtered.append(entity)
    
    # Sort by confidence and return top 20
    filtered.sort(key=lambda x: x.confidence, reverse=True)
    return filtered[:20]


@cocoindex.op.function()
def save_chunks_to_supabase(chunks: List[Dict[str, Any]], document_id: str) -> int:
    """
    Save chunks to Supabase
    
    Args:
        chunks: List of chunk dictionaries
        document_id: Document ID the chunks belong to
        
    Returns:
        Number of chunks saved
    """
    logger.info(f"Saving {len(chunks)} chunks for document {document_id}")
    
    supabase = SupabaseService()
    
    # Delete existing chunks for this document
    supabase.client.table("chunks").delete().eq("document_id", document_id).execute()
    
    # Prepare chunks for insertion
    chunk_records = []
    for i, chunk in enumerate(chunks):
        chunk_records.append({
            "document_id": document_id,
            "chunk_index": i,
            "content": chunk.get("text", ""),
            "metadata": {
                "start_position": chunk.get("start", 0),
                "end_position": chunk.get("end", 0),
                "chunk_size": len(chunk.get("text", ""))
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
    
    # Insert chunks in batches
    if chunk_records:
        supabase.client.table("chunks").insert(chunk_records).execute()
    
    return len(chunk_records)


@cocoindex.op.function()
def save_entities_to_supabase(entities: List[Entity], document_id: str) -> int:
    """
    Save entities to Supabase
    
    Args:
        entities: List of Entity objects
        document_id: Document ID the entities belong to
        
    Returns:
        Number of entities saved
    """
    logger.info(f"Saving {len(entities)} entities for document {document_id}")
    
    supabase = SupabaseService()
    
    # Delete existing entities for this document
    supabase.client.table("entities").delete().eq("document_id", document_id).execute()
    
    # Prepare entities for insertion
    entity_records = []
    for entity in entities:
        entity_records.append({
            "document_id": document_id,
            "name": entity.name,
            "type": entity.type,
            "confidence": entity.confidence,
            "metadata": entity.properties or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
    
    # Insert entities
    if entity_records:
        supabase.client.table("entities").insert(entity_records).execute()
    
    return len(entity_records)


@cocoindex.flow_def(name="DocumentProcessor")
def process_document_flow(flow_builder: FlowBuilder, data_scope: DataScope, document_id: str):
    """
    Main document processing flow using CocoIndex
    
    This flow:
    1. Fetches a document from Supabase
    2. Parses content if needed
    3. Chunks the content using CocoIndex's built-in function
    4. Extracts entities using LLM
    5. Generates embeddings
    6. Stores everything in appropriate databases
    
    Args:
        flow_builder: CocoIndex flow builder
        data_scope: CocoIndex data scope
        document_id: ID of document to process
    """
    logger.info(f"Starting CocoIndex flow for document {document_id}")
    
    # Add collectors for different outputs
    chunks_collector = data_scope.add_collector()
    entities_collector = data_scope.add_collector()
    vectors_collector = data_scope.add_collector()
    
    # Fetch document from Supabase
    doc_data = fetch_document_from_supabase(document_id)
    
    # Create a data slice from the document
    data_scope["doc"] = flow_builder.add_data([doc_data])
    
    # Process the document
    with data_scope["doc"].row() as doc:
        # Parse content if it's a binary format
        if doc["source_type"] in ["pdf", "docx"]:
            doc["parsed_content"] = doc["content"].transform(
                parse_with_llamaparse,
                document_type=doc["source_type"]
            )
        else:
            doc["parsed_content"] = doc["content"]
        
        # Chunk the document
        doc["chunks"] = doc["parsed_content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=1500,
            chunk_overlap=200,
            min_chunk_size=100
        )
        
        # Process each chunk
        with doc["chunks"].row() as chunk:
            # Generate embedding for the chunk
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(
                    model="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
            
            # Collect chunk data
            chunks_collector.collect(
                document_id=doc["id"],
                chunk_index=chunk["index"],
                text=chunk["text"],
                start=chunk.get("start", 0),
                end=chunk.get("end", 0)
            )
            
            # Collect vector data
            vectors_collector.collect(
                document_id=doc["id"],
                chunk_index=chunk["index"],
                text=chunk["text"],
                embedding=chunk["embedding"]
            )
        
        # Extract entities from the full content
        try:
            doc["entities"] = doc["parsed_content"].transform(
                cocoindex.functions.ExtractByLlm(
                    llm_spec=cocoindex.LlmSpec(
                        api_type=cocoindex.LlmApiType.OPENAI,
                        model="gpt-4o-mini",
                        api_key=settings.openai_api_key
                    ),
                    output_type=List[Entity],
                    instruction="""You are an expert entity extraction system for a knowledge management platform.
                    Extract ONLY meaningful named entities that would be valuable for knowledge retrieval and search.
                    
                    STRICT REQUIREMENTS:
                    1. Only extract proper nouns and specific terms that represent:
                       - People (full names, not just "user" or "person")
                       - Organizations/Companies (actual names, not generic terms)
                       - Locations (specific places, cities, countries)
                       - Products/Technologies (specific products, tools, frameworks)
                       - Important concepts/technical terms (domain-specific terminology)
                       - Dates/Events (specific dates or named events)
                       
                    2. DO NOT extract:
                       - Common words (the, this, that, how, when, where, why, etc.)
                       - Generic terms (system, user, data, information, etc.)
                       - Pronouns or generic references
                       - Common verbs or adjectives
                       - Single letters or very short non-specific words
                       
                    3. Each entity must be:
                       - At least 2 characters long
                       - A proper noun or domain-specific term
                       - Something that would be useful as a search term
                    
                    For each entity, provide:
                    - name: The exact entity name as it appears (preserve capitalization)
                    - type: One of [person, organization, location, product, technology, concept, event]
                    - confidence: Your confidence (0.0-1.0) based on:
                      * 0.9-1.0: Clearly identifiable proper nouns
                      * 0.7-0.9: Likely entities with strong context
                      * 0.5-0.7: Possible entities needing verification
                      * Below 0.5: Do not include
                    - properties: Any additional relevant properties
                    
                    Maximum 20 most important entities. Quality over quantity.
                    """
                )
            )
            
            # Filter out low-quality entities
            doc["filtered_entities"] = doc["entities"].transform(filter_entities)
            
            # Collect entity data
            with doc["filtered_entities"].row() as entity:
                entities_collector.collect(
                    document_id=doc["id"],
                    name=entity["name"],
                    type=entity["type"],
                    confidence=entity["confidence"],
                    properties=entity.get("properties", {})
                )
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            # Create a default entity on error
            entities_collector.collect(
                document_id=doc["id"],
                name=doc["name"],
                type="document",
                confidence=1.0,
                properties={"error": str(e)}
            )
    
    # Export to Postgres/Supabase (metadata and chunks)
    chunks_collector.export(
        "chunks",
        cocoindex.targets.Postgres(
            connection=cocoindex.targets.PostgresConnection(
                host=settings.supabase_host,
                port=settings.supabase_port,
                database=settings.supabase_database,
                user=settings.supabase_user,
                password=settings.supabase_password
            )
        ),
        table_name="chunks",
        primary_key_fields=["document_id", "chunk_index"]
    )
    
    entities_collector.export(
        "entities",
        cocoindex.targets.Postgres(
            connection=cocoindex.targets.PostgresConnection(
                host=settings.supabase_host,
                port=settings.supabase_port,
                database=settings.supabase_database,
                user=settings.supabase_user,
                password=settings.supabase_password
            )
        ),
        table_name="entities",
        primary_key_fields=["document_id", "name"]
    )
    
    # Export to Qdrant (vector storage) if configured
    if settings.qdrant_url:
        try:
            vectors_collector.export(
                "vectors",
                cocoindex.targets.Qdrant(
                    connection=cocoindex.targets.QdrantConnection(
                        grpc_url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key
                    )
                ),
                collection_name="documents",
                vector_field="embedding"
            )
            logger.info("Exported vectors to Qdrant")
        except Exception as e:
            logger.error(f"Failed to export to Qdrant: {e}")
    
    # Export to Neo4j (knowledge graph) if configured
    if settings.neo4j_uri:
        try:
            entities_collector.export(
                "graph",
                cocoindex.targets.Neo4j(
                    connection=cocoindex.targets.Neo4jConnection(
                        uri=settings.neo4j_uri,
                        username=settings.neo4j_username,
                        password=settings.neo4j_password
                    )
                ),
                mapping=cocoindex.targets.Nodes(
                    label="Entity",
                    key_property="name"
                )
            )
            logger.info("Exported entities to Neo4j")
        except Exception as e:
            logger.error(f"Failed to export to Neo4j: {e}")
    
    logger.info(f"Completed CocoIndex flow for document {document_id}")


async def process_document_simple(document_id: str) -> Dict[str, Any]:
    """
    Simple document processing without full CocoIndex flow
    Directly processes the document using the three-tier chunker
    """
    logger.info(f"Processing document {document_id} with simple pipeline")
    
    try:
        # Get document from Supabase
        supabase = SupabaseService()
        doc_response = supabase.client.table("documents").select("*").eq("id", document_id).single().execute()
        
        if not doc_response.data:
            raise ValueError(f"Document {document_id} not found")
        
        document = doc_response.data
        content = document.get("content", "")
        
        if not content:
            logger.warning(f"Document {document_id} has no content")
            return {"status": "failed", "error": "No content to process"}
        
        # Import the three-tier chunker
        from app.processors.three_tier_chunker import ThreeTierChunker
        
        # Create chunks using three-tier strategy
        chunker = ThreeTierChunker()
        chunks_data = await chunker.process_document(
            document_id=document_id,
            content=content,
            metadata=document.get("metadata", {})
        )
        
        # Delete existing chunks
        supabase.client.table("chunks").delete().eq("document_id", document_id).execute()
        logger.info(f"Deleted existing chunks for document {document_id}")
        
        # Save new chunks to Supabase
        chunks_to_insert = []
        for i, chunk in enumerate(chunks_data):  # chunks_data is already a list
            # Add hierarchy level to metadata
            metadata = chunk.metadata or {}
            metadata["hierarchy_level"] = chunk.chunk_level
            metadata["semantic_density"] = chunk.metadata.get("semantic_density", 0) if chunk.metadata else 0
            metadata["semantic_focus"] = chunk.semantic_focus
            
            chunk_record = {
                "id": chunk.id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "chunk_size": chunk.chunk_size,
                "chunking_strategy": chunk.chunking_strategy,
                "chunk_level": chunk.chunk_level,  # Add chunk level
                "contextual_summary": chunk.contextual_summary,  # Add contextual summary
                "contextualized_text": chunk.contextualized_text,  # Add contextualized text
                "bm25_tokens": chunk.bm25_tokens,  # Add BM25 tokens
                "metadata": metadata,
                "parent_chunk_id": chunk.parent_chunk_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            chunks_to_insert.append(chunk_record)
        
        if chunks_to_insert:
            supabase.client.table("chunks").insert(chunks_to_insert).execute()
            logger.info(f"Inserted {len(chunks_to_insert)} chunks for document {document_id}")
        
        # Extract entities using improved extraction
        from app.processors.entity_extractor import EntityExtractor
        from app.models.chunk import Chunk
        
        # Convert chunk data to Chunk objects for entity extraction
        chunk_objects = []
        for i, chunk in enumerate(chunks_data):  # chunks_data is already a list
            chunk_obj = Chunk(
                id=chunk.id,
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                chunk_size=chunk.chunk_size,
                chunking_strategy=chunk.chunking_strategy,
                metadata=chunk.metadata or {}
            )
            chunk_objects.append(chunk_obj)
        
        # Extract entities
        extractor = EntityExtractor()
        entities, relationships = extractor.extract(chunk_objects, document_id)
        
        # Delete existing entities
        supabase.client.table("entities").delete().eq("document_id", document_id).execute()
        logger.info(f"Deleted existing entities for document {document_id}")
        
        # Save entities to Supabase
        if entities:
            entities_to_insert = []
            for entity in entities:
                entity_record = {
                    "id": entity.id,
                    "document_id": document_id,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                    "confidence": entity.confidence,
                    "metadata": entity.metadata,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                entities_to_insert.append(entity_record)
            
            supabase.client.table("entities").insert(entities_to_insert).execute()
            logger.info(f"Inserted {len(entities_to_insert)} entities for document {document_id}")
        
        # Update document status
        supabase.client.table("documents").update({
            "status": "ingested",
            "chunk_count": len(chunks_to_insert),
            "entity_count": len(entities) if entities else 0,
            "processed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", document_id).execute()
        
        return {
            "status": "success",
            "chunks_created": len(chunks_to_insert),
            "entities_extracted": len(entities) if entities else 0
        }
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
        
        # Update document status to failed
        try:
            supabase = SupabaseService()
            supabase.client.table("documents").update({
                "status": "failed",
                "updated_at": datetime.utcnow().isoformat(),
                "metadata": {"processing_error": str(e)}
            }).eq("id", document_id).execute()
        except:
            pass
        
        return {"status": "failed", "error": str(e)}


async def create_and_run_flow(document_id: str) -> Dict[str, Any]:
    """
    Create and run a CocoIndex flow for a single document
    
    Args:
        document_id: ID of document to process
        
    Returns:
        Result dictionary with processing status
    """
    try:
        logger.info(f"Creating CocoIndex flow for document {document_id}")
        
        # Initialize CocoIndex if not already done
        cocoindex.init()
        
        # Create a simple flow for single document processing
        # For now, use a simple manual processing approach
        return await process_document_simple(document_id)
        
        # Update document status in Supabase
        supabase = SupabaseService()
        supabase.client.table("documents").update({
            "status": DocumentState.PENDING_REVIEW.value,
            "processed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", document_id).execute()
        
        # Get counts for response
        chunks_response = supabase.client.table("chunks").select("id").eq("document_id", document_id).execute()
        entities_response = supabase.client.table("entities").select("id").eq("document_id", document_id).execute()
        
        chunk_count = len(chunks_response.data) if chunks_response.data else 0
        entity_count = len(entities_response.data) if entities_response.data else 0
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_created": chunk_count,
            "entities_extracted": entity_count,
            "message": f"Document processed successfully with {chunk_count} chunks and {entity_count} entities"
        }
        
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}", exc_info=True)
        
        # Update document status to failed
        try:
            supabase = SupabaseService()
            supabase.client.table("documents").update({
                "status": DocumentState.FAILED.value,
                "updated_at": datetime.utcnow().isoformat(),
                "metadata": {"error": str(e)}
            }).eq("id", document_id).execute()
        except:
            pass
        
        return {
            "status": "error",
            "document_id": document_id,
            "error": str(e),
            "message": f"Failed to process document: {str(e)}"
        }