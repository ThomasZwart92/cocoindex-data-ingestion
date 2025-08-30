"""
CocoIndex Flow for Document Ingestion Pipeline - PROPER IMPLEMENTATION
This follows the declarative dataflow pattern as specified in CLAUDE.md
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

import cocoindex
from cocoindex import FlowBuilder, DataScope
from cocoindex.sources import LocalFile, GoogleDrive
from cocoindex.targets import Postgres, Qdrant, Neo4j
from cocoindex.functions import (
    SplitRecursively,
    ExtractByLlm,
    EmbedText
)
from cocoindex.llm import LlmSpec, LlmApiType

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Data classes for structured extraction
@dataclass
class DocumentEntity:
    """Entity extracted from document"""
    name: str
    type: str
    confidence: float
    properties: Dict[str, Any] = None

@dataclass
class DocumentMetadata:
    """Metadata extracted from document"""
    title: str
    author: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None
    department: Optional[str] = None
    security_level: int = 1  # Default to public

@dataclass
class EntityRelationship:
    """Relationship between entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float

# Custom functions for our specific needs
@cocoindex.op.function()
def parse_document_with_llamaparse(content: bytes, tier: str = "balanced") -> str:
    """Parse document using LlamaParse with quality tiers"""
    try:
        from llama_parse import LlamaParse
        
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_PARSE_API_KEY"),
            result_type="markdown",
            parsing_mode=tier,  # balanced, agentic, agentic_plus
            verbose=False
        )
        
        # Parse the document
        result = parser.parse_raw(content)
        
        # Track parsing cost
        # Cost tracking should be handled by external monitoring
        # Not hardcoded in the application
        cost = None  # Will be tracked via LlamaParse API usage dashboard
        
        logger.info(f"Parsed document with tier '{tier}'")
        
        return result.text
        
    except Exception as e:
        logger.error(f"LlamaParse failed: {e}, falling back to basic extraction")
        # Fallback to basic text extraction
        return content.decode('utf-8', errors='ignore')

@cocoindex.op.function()
def extract_entities_with_hybrid(text: str) -> List[DocumentEntity]:
    """Extract entities using hybrid approach (rules + LLM)"""
    entities = []
    
    # Rule-based extraction for common patterns
    import re
    
    # Extract email addresses
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities.append(DocumentEntity(
            name=email,
            type="Email",
            confidence=1.0,
            properties={"source": "rule-based"}
        ))
    
    # Extract product codes (pattern: NC####)
    product_codes = re.findall(r'\bNC\d{4}\b', text)
    for code in product_codes:
        entities.append(DocumentEntity(
            name=code,
            type="ProductCode",
            confidence=1.0,
            properties={"source": "rule-based"}
        ))
    
    # Extract error codes (pattern: E###)
    error_codes = re.findall(r'\bE\d{3}\b', text)
    for code in error_codes:
        entities.append(DocumentEntity(
            name=code,
            type="ErrorCode",
            confidence=1.0,
            properties={"source": "rule-based"}
        ))
    
    # LLM extraction will be added via ExtractByLlm in the flow
    return entities

@cocoindex.op.function()
def extract_relationships(text: str, entities: List[DocumentEntity]) -> List[EntityRelationship]:
    """Extract relationships between entities"""
    relationships = []
    
    # Define relationship patterns
    patterns = {
        "TROUBLESHOOTS": [
            r"({entity1}).*(?:fixes?|solves?|resolves?|troubleshoots?).*({entity2})",
            r"({entity2}).*(?:fixed by|solved by|resolved by).*({entity1})"
        ],
        "DEPENDS_ON": [
            r"({entity1}).*(?:depends? on|requires?|needs?).*({entity2})",
            r"({entity2}).*(?:required by|needed by).*({entity1})"
        ],
        "COMPONENT_OF": [
            r"({entity1}).*(?:part of|component of|included in).*({entity2})",
            r"({entity2}).*(?:contains?|includes?).*({entity1})"
        ],
        "REPLACES": [
            r"({entity1}).*(?:replaces?|supersedes?).*({entity2})",
            r"({entity2}).*(?:replaced by|superseded by).*({entity1})"
        ]
    }
    
    # Extract relationships based on patterns
    import re
    entity_names = [e.name for e in entities]
    
    for rel_type, rel_patterns in patterns.items():
        for pattern in rel_patterns:
            # Check for relationships between each pair of entities
            for i, entity1 in enumerate(entity_names):
                for entity2 in entity_names[i+1:]:
                    # Create pattern with actual entity names
                    actual_pattern = pattern.replace("{entity1}", re.escape(entity1))
                    actual_pattern = actual_pattern.replace("{entity2}", re.escape(entity2))
                    
                    if re.search(actual_pattern, text, re.IGNORECASE):
                        relationships.append(EntityRelationship(
                            source_entity=entity1,
                            target_entity=entity2,
                            relationship_type=rel_type,
                            confidence=0.8
                        ))
    
    return relationships

@cocoindex.flow_def(name="DocumentIngestionFlow")
def document_ingestion_flow(flow_builder: FlowBuilder, data_scope: DataScope):
    """
    Main document ingestion flow using PROPER CocoIndex declarative pattern
    
    This flow:
    1. Reads documents from sources (Local/Notion/GDrive)
    2. Parses documents with LlamaParse
    3. Chunks content with CocoIndex functions
    4. Extracts entities and metadata with LLMs
    5. Generates embeddings
    6. Stores in Supabase, Qdrant, and Neo4j
    """
    
    # Configure database connections
    # Note: Using Supabase PostgreSQL for both app data and CocoIndex state
    
    # CocoIndex state tracking (uses separate schema in Supabase)
    # CRITICAL: No fallbacks - must have proper environment variables
    supabase_host = os.getenv("SUPABASE_HOST")
    supabase_password = os.getenv("SUPABASE_DB_PASSWORD")
    if not supabase_host or not supabase_password:
        raise ValueError("SUPABASE_HOST and SUPABASE_DB_PASSWORD must be set in environment")
    
    postgres_conn = flow_builder.add_auth_entry(
        "cocoindex_state",
        cocoindex.targets.PostgresConnection(
            host=supabase_host,
            port=5432,
            database="postgres",  # Supabase default
            user="postgres",
            password=supabase_password,
            schema="cocoindex"  # Separate schema for CocoIndex state
        )
    )
    
    # Qdrant for vector storage
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL must be set in environment")
    
    qdrant_conn = flow_builder.add_auth_entry(
        "qdrant_vectors",
        cocoindex.targets.QdrantConnection(
            grpc_url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY")  # Optional
        )
    )
    
    # Neo4j for knowledge graph
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_uri or not neo4j_username or not neo4j_password:
        raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in environment")
    
    neo4j_conn = flow_builder.add_auth_entry(
        "neo4j_graph",
        cocoindex.targets.Neo4jConnection(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
    )
    
    # Configure source based on environment
    # CRITICAL: Must use real sources, no mock data
    source_type = os.getenv("DOCUMENT_SOURCE_TYPE", "google_drive").lower()
    
    if source_type == "google_drive":
        # Google Drive source - primary production source
        service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
        drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        
        if not service_account_path or not drive_folder_id:
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_PATH and GOOGLE_DRIVE_FOLDER_ID must be set for Google Drive source")
        
        data_scope["documents"] = flow_builder.add_source(
            GoogleDrive(
                service_account_credential_path=service_account_path,
                root_folder_ids=[drive_folder_id],
                recent_changes_poll_interval=timedelta(minutes=30)
            )
        )
        logger.info(f"Using Google Drive source with folder: {drive_folder_id}")
        
    elif source_type == "local":
        # Local file source - only for development with real documents
        local_path = os.getenv("LOCAL_DOCUMENTS_PATH")
        if not local_path:
            raise ValueError("LOCAL_DOCUMENTS_PATH must be set for local file source")
        
        # Verify the path exists and contains documents
        from pathlib import Path
        doc_path = Path(local_path)
        if not doc_path.exists():
            raise ValueError(f"Local documents path does not exist: {local_path}")
        
        data_scope["documents"] = flow_builder.add_source(
            LocalFile(
                path=local_path,
                pattern="**/*.{pdf,txt,md,docx}"  # Real document patterns
            )
        )
        logger.info(f"Using local file source: {local_path}")
        
    else:
        raise ValueError(f"Unknown source type: {source_type}. Use 'google_drive' or 'local'")
    
    # Add collectors for different outputs
    chunks_output = data_scope.add_collector()
    entities_output = data_scope.add_collector()
    relationships_output = data_scope.add_collector()
    
    # DECLARATIVE TRANSFORMATIONS - This is the key pattern!
    # Each transformation creates a new field, never mutates existing ones
    
    with data_scope["documents"].row() as doc:
        # Step 1: Parse document (PDF → Markdown)
        doc["parsed_content"] = doc["content"].transform(
            parse_document_with_llamaparse,
            tier="balanced"  # Can upgrade to "agentic" for complex docs
        )
        
        # Step 2: Extract metadata using LLM
        doc["metadata"] = doc["parsed_content"].transform(
            ExtractByLlm(
                llm_spec=LlmSpec(
                    api_type=LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                output_type=DocumentMetadata,
                instruction="""
                Extract document metadata including:
                - Title
                - Author (if mentioned)
                - Category (technical, business, support, etc.)
                - Tags (relevant keywords)
                - Department (engineering, sales, support, etc.)
                - Security level (1-5 based on content sensitivity)
                """
            )
        )
        
        # Step 3: Chunk the document
        doc["chunks"] = doc["parsed_content"].transform(
            SplitRecursively(),
            language="markdown",
            chunk_size=1500,
            chunk_overlap=200,
            min_chunk_size=100
        )
        
        # Step 4: Extract entities (hybrid approach)
        doc["rule_entities"] = doc["parsed_content"].transform(
            extract_entities_with_hybrid
        )
        
        doc["llm_entities"] = doc["parsed_content"].transform(
            ExtractByLlm(
                llm_spec=LlmSpec(
                    api_type=LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                output_type=List[DocumentEntity],
                instruction="""
                Extract all important entities from this document:
                - People (names, roles)
                - Products (product names, models)
                - Companies/Organizations
                - Technical terms
                - Error codes
                - Departments
                Include confidence score for each entity.
                """
            )
        )
        
        # Step 5: Extract relationships
        doc["relationships"] = doc["parsed_content"].transform(
            extract_relationships,
            entities=doc["llm_entities"]
        )
        
        # Step 6: Process each chunk
        with doc["chunks"].row() as chunk:
            # Generate embeddings for each chunk
            chunk["embedding"] = chunk["text"].transform(
                EmbedText(
                    model="text-embedding-3-small",
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            )
            
            # Collect chunks for vector storage
            chunks_output.collect(
                document_id=doc["id"],
                chunk_text=chunk["text"],
                chunk_embedding=chunk["embedding"],
                chunk_position=chunk["position"],
                metadata=doc["metadata"],
                security_level=doc["metadata"]["security_level"]
            )
        
        # Step 7: Collect entities
        for entity in doc["llm_entities"]:
            entities_output.collect(
                document_id=doc["id"],
                entity_name=entity["name"],
                entity_type=entity["type"],
                confidence=entity["confidence"],
                properties=entity["properties"]
            )
        
        # Step 8: Collect relationships
        for rel in doc["relationships"]:
            relationships_output.collect(
                document_id=doc["id"],
                source_entity=rel["source_entity"],
                target_entity=rel["target_entity"],
                relationship_type=rel["relationship_type"],
                confidence=rel["confidence"]
            )
    
    # EXPORT TO TARGETS - Multi-database architecture
    
    # Export chunks to Qdrant for vector search
    chunks_output.export(
        "document_chunks",
        Qdrant(
            connection=qdrant_conn,
            collection_name="document_chunks",
            vector_field="chunk_embedding",
            payload_fields=["document_id", "chunk_text", "metadata", "security_level"]
        )
    )
    
    # Export entities to Neo4j
    entities_output.export(
        "entities",
        Neo4j(
            connection=neo4j_conn,
            mapping=cocoindex.targets.Nodes(
                label="Entity",
                property_mapping={
                    "name": "entity_name",
                    "type": "entity_type",
                    "confidence": "confidence"
                }
            )
        )
    )
    
    # Export relationships to Neo4j
    relationships_output.export(
        "relationships",
        Neo4j(
            connection=neo4j_conn,
            mapping=cocoindex.targets.Relationships(
                type_field="relationship_type",
                source_node_label="Entity",
                source_node_property="name",
                source_field="source_entity",
                target_node_label="Entity",
                target_node_property="name",
                target_field="target_entity",
                property_mapping={"confidence": "confidence"}
            )
        )
    )
    
    # Export metadata to Supabase (application database)
    chunks_output.export(
        "app_metadata",
        Postgres(
            connection=postgres_conn,
            table_name="document_chunks",
            schema="public"  # Main app schema
        )
    )

# Utility function to run the flow
def run_ingestion_flow():
    """Run the document ingestion flow"""
    try:
        # Initialize CocoIndex
        cocoindex.init()
        
        # Create flow builder and data scope
        flow_builder = cocoindex.FlowBuilder()
        data_scope = cocoindex.DataScope()
        
        # Build the flow
        document_ingestion_flow(flow_builder, data_scope)
        
        # Get the built flow
        flow = flow_builder.build()
        
        logger.info("Starting document ingestion flow...")
        result = flow.run()
        
        if hasattr(result, 'success'):
            if result.success:
                logger.info(f"Flow completed successfully. Processed {getattr(result, 'documents_processed', 'N/A')} documents.")
            else:
                logger.error(f"Flow failed: {getattr(result, 'error', 'Unknown error')}")
        else:
            logger.info("Flow completed")
            
        return result
        
    except Exception as e:
        logger.error(f"Error running flow: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the flow
    result = run_ingestion_flow()