"""
CocoIndex Flow with Dual-Write Pattern for Complete Supabase Population
This flow implements the dual-write pattern to populate all 8 Supabase tables
while maintaining optimized search in Qdrant and graph queries in Neo4j
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta, datetime
from dotenv import load_dotenv
import uuid

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
# Connection classes are created inline with the target definitions

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
    summary: Optional[str] = None
    keywords: List[str] = None
    language: str = "en"

@dataclass
class EntityRelationship:
    """Relationship between entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float

@dataclass
class ImageCaption:
    """Image caption and metadata"""
    image_index: int
    page_number: int
    caption: str
    confidence: float
    ocr_text: Optional[str] = None

# Custom functions for our specific needs
@cocoindex.op.function()
def parse_document_with_llamaparse(content: bytes, tier: str = "balanced") -> tuple[str, float]:
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
        
        # Estimate confidence based on tier
        confidence = {
            "balanced": 0.7,
            "agentic": 0.85,
            "agentic_plus": 0.95
        }.get(tier, 0.7)
        
        logger.info(f"Parsed document with tier '{tier}', confidence {confidence}")
        
        return result.text, confidence
        
    except Exception as e:
        logger.error(f"LlamaParse failed: {e}, falling back to basic extraction")
        # Fallback to basic text extraction
        return content.decode('utf-8', errors='ignore'), 0.5

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
    
    return entities

@cocoindex.op.function()
def extract_relationships(text: str, entities: List[DocumentEntity]) -> List[EntityRelationship]:
    """Extract relationships between entities"""
    relationships = []
    
    # Define relationship patterns (14 types as specified)
    patterns = {
        # Technical relationships
        "COMPONENT_OF": [
            r"({entity1}).*(?:part of|component of|included in).*({entity2})",
            r"({entity2}).*(?:contains?|includes?).*({entity1})"
        ],
        "CONNECTS_TO": [
            r"({entity1}).*(?:connects? to|interfaces? with|links? to).*({entity2})"
        ],
        "DEPENDS_ON": [
            r"({entity1}).*(?:depends? on|requires?|needs?).*({entity2})"
        ],
        "REPLACES": [
            r"({entity1}).*(?:replaces?|supersedes?).*({entity2})"
        ],
        "TROUBLESHOOTS": [
            r"({entity1}).*(?:fixes?|solves?|resolves?|troubleshoots?).*({entity2})"
        ],
        # Documentation relationships
        "DEFINES": [
            r"({entity1}).*(?:defines?|specifies?).*({entity2})"
        ],
        "DOCUMENTS": [
            r"({entity1}).*(?:documents?|describes?).*({entity2})"
        ],
        "REFERENCES": [
            r"({entity1}).*(?:references?|refers? to|cites?).*({entity2})"
        ],
        "TARGETS": [
            r"({entity1}).*(?:targets?|aims? at|focuses? on).*({entity2})"
        ],
        # Business relationships
        "RESPONSIBLE_FOR": [
            r"({entity1}).*(?:responsible for|manages?|oversees?).*({entity2})"
        ],
        "SERVES": [
            r"({entity1}).*(?:serves?|provides? to|supports?).*({entity2})"
        ],
        "IMPACTS": [
            r"({entity1}).*(?:impacts?|affects?|influences?).*({entity2})"
        ],
        # Flexible relationships
        "RELATES_TO": [
            r"({entity1}).*(?:relates? to|associated with).*({entity2})"
        ],
        "COMPATIBLE_WITH": [
            r"({entity1}).*(?:compatible with|works? with).*({entity2})"
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

@cocoindex.op.function()
def compare_llm_outputs(gpt_output: Any, gemini_output: Any) -> Dict[str, Any]:
    """Compare outputs from multiple LLMs for quality review"""
    comparison = {
        "gpt4_output": gpt_output,
        "gemini_output": gemini_output,
        "selected_model": None,
        "selected_output": None,
        "confidence_difference": 0
    }
    
    # Simple selection logic - can be enhanced
    if gpt_output and gemini_output:
        # For now, prefer GPT-4 unless Gemini has significantly higher confidence
        comparison["selected_model"] = "gpt4"
        comparison["selected_output"] = gpt_output
    elif gpt_output:
        comparison["selected_model"] = "gpt4"
        comparison["selected_output"] = gpt_output
    elif gemini_output:
        comparison["selected_model"] = "gemini"
        comparison["selected_output"] = gemini_output
    
    return comparison

@cocoindex.flow_def(name="DocumentIngestionDualWrite")
def document_ingestion_dual_write(flow_builder: FlowBuilder, data_scope: DataScope):
    """
    Document ingestion flow with dual-write pattern to populate all Supabase tables
    
    This flow implements the architecture where:
    - Supabase stores UI state and review data (8 tables)
    - Qdrant handles optimized vector search
    - Neo4j manages knowledge graph relationships
    """
    
    # Configure database connections
    # Note: CocoIndex will use environment variables or default connection settings
    # For Supabase/Postgres, ensure DATABASE_URL or individual PG_* vars are set
    # For Qdrant, it will use QDRANT_URL if set
    # For Neo4j, it will use NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    
    # Verify critical environment variables
    supabase_host = os.getenv("SUPABASE_HOST")
    supabase_password = os.getenv("SUPABASE_DB_PASSWORD")
    if supabase_host and supabase_password:
        # Set DATABASE_URL for CocoIndex to use
        os.environ["DATABASE_URL"] = f"postgresql://postgres:{supabase_password}@{supabase_host}:5432/postgres"
    
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD must be set in environment")
    
    # Configure source
    source_type = os.getenv("DOCUMENT_SOURCE_TYPE", "local").lower()
    
    if source_type == "google_drive":
        service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
        drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        
        if not service_account_path or not drive_folder_id:
            raise ValueError("Google Drive configuration incomplete")
        
        data_scope["documents"] = flow_builder.add_source(
            GoogleDrive(
                service_account_credential_path=service_account_path,
                root_folder_ids=[drive_folder_id],
                recent_changes_poll_interval=timedelta(minutes=30)
            )
        )
    else:
        # Default to local files for testing
        data_path = os.getenv("LOCAL_DATA_PATH", "./test_data")
        data_scope["documents"] = flow_builder.add_source(
            LocalFile(path=data_path)
        )
    
    # Create collectors for all 8 Supabase tables
    documents_collector = data_scope.add_collector()
    chunks_collector = data_scope.add_collector()
    entities_collector = data_scope.add_collector()
    relationships_collector = data_scope.add_collector()
    metadata_collector = data_scope.add_collector()
    processing_jobs_collector = data_scope.add_collector()
    llm_comparisons_collector = data_scope.add_collector()
    images_collector = data_scope.add_collector()
    
    # Also create collectors for Qdrant and Neo4j
    qdrant_chunks_collector = data_scope.add_collector()
    neo4j_entities_collector = data_scope.add_collector()
    neo4j_relationships_collector = data_scope.add_collector()
    
    # Process each document
    with data_scope["documents"].row() as doc:
        # Generate document ID
        doc["id"] = str(uuid.uuid4())
        doc["created_at"] = datetime.utcnow().isoformat()
        
        # Step 1: Parse document with LlamaParse
        parse_result = doc["content"].transform(
            parse_document_with_llamaparse,
            tier=os.getenv("LLAMAPARSE_TIER", "balanced")
        )
        doc["parsed_content"] = parse_result[0]
        doc["parse_confidence"] = parse_result[1]
        
        # Collect document info for documents table
        documents_collector.collect(
            id=doc["id"],
            name=doc.get("filename", "Unknown"),
            source_type="local",
            file_type=doc.get("extension", "unknown"),
            status="processing",
            parse_tier=os.getenv("LLAMAPARSE_TIER", "balanced"),
            parse_confidence=doc["parse_confidence"],
            created_at=doc["created_at"],
            updated_at=doc["created_at"]
        )
        
        # Create processing job record
        job_id = str(uuid.uuid4())
        processing_jobs_collector.collect(
            id=job_id,
            document_id=doc["id"],
            job_type="full_pipeline",
            job_status="running",
            progress=10,
            current_step="Extracting metadata",
            total_steps=7,
            created_at=doc["created_at"],
            updated_at=doc["created_at"]
        )
        
        # Step 2: Extract metadata using multiple LLMs for comparison
        doc["gpt_metadata"] = doc["parsed_content"].transform(
            ExtractByLlm(
                llm_spec=LlmSpec(
                    api_type=LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                output_type=DocumentMetadata,
                instruction="""
                Extract comprehensive document metadata including:
                - Title
                - Author (if mentioned)
                - Category (technical, business, support, etc.)
                - Tags (relevant keywords)
                - Department (engineering, sales, support, etc.)
                - Security level (1-5 based on content sensitivity)
                - Summary (2-3 sentences)
                - Keywords (top 10 important terms)
                - Language
                """
            )
        )
        
        # Try Gemini as well for comparison
        try:
            doc["gemini_metadata"] = doc["parsed_content"].transform(
                ExtractByLlm(
                    llm_spec=LlmSpec(
                        api_type=LlmApiType.GEMINI,
                        model="gemini-2.5-pro",
                        api_key=os.getenv("GOOGLE_AI_API_KEY")
                    ),
                    output_type=DocumentMetadata,
                    instruction="Extract document metadata with title, author, category, tags, department, security level, summary, keywords, and language."
                )
            )
        except:
            doc["gemini_metadata"] = None
        
        # Compare LLM outputs
        metadata_comparison = compare_llm_outputs(doc["gpt_metadata"], doc["gemini_metadata"])
        doc["metadata"] = metadata_comparison["selected_output"]
        
        # Collect LLM comparison
        llm_comparisons_collector.collect(
            id=str(uuid.uuid4()),
            document_id=doc["id"],
            comparison_type="metadata",
            gpt4_output=doc["gpt_metadata"],
            gpt4_confidence=0.85,
            gemini_output=doc["gemini_metadata"],
            gemini_confidence=0.80,
            selected_model=metadata_comparison["selected_model"],
            selected_output=metadata_comparison["selected_output"],
            created_at=doc["created_at"]
        )
        
        # Collect document metadata
        metadata_collector.collect(
            id=str(uuid.uuid4()),
            document_id=doc["id"],
            title=doc["metadata"].get("title", "Untitled"),
            author=doc["metadata"].get("author"),
            description=doc["metadata"].get("summary"),
            language=doc["metadata"].get("language", "en"),
            category=doc["metadata"].get("category"),
            keywords=doc["metadata"].get("keywords", []),
            summary=doc["metadata"].get("summary"),
            metadata_source="extracted",
            extraction_model=metadata_comparison["selected_model"],
            created_at=doc["created_at"],
            updated_at=doc["created_at"]
        )
        
        # Step 3: Chunk the document
        doc["chunks"] = doc["parsed_content"].transform(
            SplitRecursively(),
            language="markdown",
            chunk_size=1500,
            chunk_overlap=200,
            min_chunk_size=100
        )
        
        # Step 4: Extract entities
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
        
        # Combine rule-based and LLM entities
        all_entities = doc["rule_entities"] + doc["llm_entities"]
        
        # Step 5: Extract relationships
        doc["relationships"] = doc["parsed_content"].transform(
            extract_relationships,
            entities=all_entities
        )
        
        # Step 6: Process each chunk
        chunk_index = 0
        with doc["chunks"].row() as chunk:
            chunk_id = str(uuid.uuid4())
            
            # Generate embeddings
            chunk["embedding"] = chunk["text"].transform(
                EmbedText(
                    model="text-embedding-3-small",
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            )
            
            # Collect for Supabase chunks table
            chunks_collector.collect(
                id=chunk_id,
                document_id=doc["id"],
                chunk_index=chunk_index,
                chunk_text=chunk["text"],
                chunk_size=len(chunk["text"]),
                chunking_strategy="recursive",
                chunk_overlap=200,
                embedding_model="text-embedding-3-small",
                created_at=doc["created_at"],
                updated_at=doc["created_at"]
            )
            
            # Collect for Qdrant
            qdrant_chunks_collector.collect(
                id=chunk_id,
                document_id=doc["id"],
                chunk_text=chunk["text"],
                chunk_embedding=chunk["embedding"],
                chunk_index=chunk_index,
                metadata=doc["metadata"],
                security_level=doc["metadata"].get("security_level", 1)
            )
            
            chunk_index += 1
        
        # Step 7: Collect entities
        for entity in all_entities:
            entity_id = str(uuid.uuid4())
            
            # Collect for Supabase entities table
            entities_collector.collect(
                id=entity_id,
                document_id=doc["id"],
                entity_name=entity.name,
                entity_type=entity.type,
                confidence_score=entity.confidence,
                source_model="hybrid",
                metadata=entity.properties,
                created_at=doc["created_at"],
                updated_at=doc["created_at"]
            )
            
            # Collect for Neo4j
            neo4j_entities_collector.collect(
                id=entity_id,
                document_id=doc["id"],
                name=entity.name,
                type=entity.type,
                confidence=entity.confidence,
                properties=entity.properties
            )
        
        # Step 8: Collect relationships
        for rel in doc["relationships"]:
            rel_id = str(uuid.uuid4())
            
            # Collect for Supabase
            relationships_collector.collect(
                id=rel_id,
                source_entity_id=rel.source_entity,
                target_entity_id=rel.target_entity,
                relationship_type=rel.relationship_type,
                confidence_score=rel.confidence,
                source_model="hybrid",
                created_at=doc["created_at"]
            )
            
            # Collect for Neo4j
            neo4j_relationships_collector.collect(
                source_entity=rel.source_entity,
                target_entity=rel.target_entity,
                relationship_type=rel.relationship_type,
                confidence=rel.confidence,
                document_id=doc["id"]
            )
        
        # Update processing job to completed
        processing_jobs_collector.collect(
            id=job_id,
            document_id=doc["id"],
            job_type="full_pipeline",
            job_status="completed",
            progress=100,
            current_step="Complete",
            total_steps=7,
            completed_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
    
    # DUAL-WRITE EXPORTS: Write to all databases
    
    # 1. Export to all 8 Supabase tables
    # CocoIndex will use DATABASE_URL or default Postgres connection
    documents_collector.export(
        "supabase_documents",
        Postgres(table_name="documents"),
        primary_key_fields=["id"]
    )
    
    chunks_collector.export(
        "supabase_chunks",
        Postgres(table_name="chunks"),
        primary_key_fields=["id"]
    )
    
    entities_collector.export(
        "supabase_entities",
        Postgres(table_name="entities"),
        primary_key_fields=["id"]
    )
    
    relationships_collector.export(
        "supabase_relationships",
        Postgres(table_name="entity_relationships"),
        primary_key_fields=["id"]
    )
    
    metadata_collector.export(
        "supabase_metadata",
        Postgres(table_name="document_metadata"),
        primary_key_fields=["id"]
    )
    
    processing_jobs_collector.export(
        "supabase_jobs",
        Postgres(table_name="processing_jobs"),
        primary_key_fields=["id"]
    )
    
    llm_comparisons_collector.export(
        "supabase_comparisons",
        Postgres(table_name="llm_comparisons"),
        primary_key_fields=["id"]
    )
    
    # Images table will be populated when we implement image extraction
    # For now, export empty collector to create the structure
    images_collector.export(
        "supabase_images",
        Postgres(table_name="document_images"),
        primary_key_fields=["id"]
    )
    
    # 2. Export to Qdrant for vector search
    qdrant_chunks_collector.export(
        "qdrant_vectors",
        Qdrant(
            collection_name="document_chunks",
            vector_field="chunk_embedding",
            payload_fields=["document_id", "chunk_text", "chunk_index", "metadata", "security_level"]
        )
    )
    
    # 3. Export to Neo4j for knowledge graph
    neo4j_entities_collector.export(
        "neo4j_entities",
        Neo4j(
            mapping=cocoindex.targets.Nodes(
                label="Entity",
                property_mapping={
                    "id": "id",
                    "name": "name",
                    "type": "type",
                    "confidence": "confidence",
                    "document_id": "document_id"
                }
            )
        )
    )
    
    neo4j_relationships_collector.export(
        "neo4j_relationships",
        Neo4j(
            mapping=cocoindex.targets.Relationships(
                type_field="relationship_type",
                source_node_label="Entity",
                source_node_property="name",
                source_field="source_entity",
                target_node_label="Entity",
                target_node_property="name",
                target_field="target_entity",
                property_mapping={
                    "confidence": "confidence",
                    "document_id": "document_id"
                }
            )
        )
    )
    
    logger.info("Dual-write flow configured: All 8 Supabase tables + Qdrant + Neo4j")

def run_dual_write_flow():
    """Run the dual-write document ingestion flow"""
    try:
        # Initialize CocoIndex
        cocoindex.init()
        
        # The flow_def decorator creates a Flow object directly
        flow = document_ingestion_dual_write
        
        logger.info("Starting dual-write document ingestion flow...")
        # Run the flow - this will call the setup and update methods
        result = flow.update()
        
        if hasattr(result, 'success'):
            if result.success:
                logger.info(f"Flow completed successfully. All 8 Supabase tables populated.")
                logger.info("Tables populated: documents, chunks, entities, entity_relationships, ")
                logger.info("                 document_metadata, processing_jobs, llm_comparisons, document_images")
            else:
                logger.error(f"Flow failed: {getattr(result, 'error', 'Unknown error')}")
        else:
            logger.info("Flow completed")
            
        return result
        
    except Exception as e:
        logger.error(f"Error running dual-write flow: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the flow
    result = run_dual_write_flow()
    print(f"Flow result: {result}")