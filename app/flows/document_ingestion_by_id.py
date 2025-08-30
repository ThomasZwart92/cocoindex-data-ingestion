"""
CocoIndex Flow for processing specific documents by ID
This version fetches documents from Supabase and processes them individually
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta, datetime
from dotenv import load_dotenv
import json

import cocoindex
from cocoindex import FlowBuilder, DataScope
from cocoindex.sources import LocalFile
from cocoindex.targets import Postgres, Qdrant, Neo4j
from cocoindex.functions import (
    SplitRecursively,
    ExtractByLlm,
    EmbedText
)
from cocoindex.llm import LlmSpec, LlmApiType
from supabase import create_client, Client

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Import data classes from main flow
from app.flows.document_ingestion_flow_v2 import (
    DocumentEntity, 
    DocumentMetadata,
    EntityRelationship,
    parse_document_with_llamaparse,
    extract_entities_with_hybrid,
    extract_relationships
)

@cocoindex.op.function()
def fetch_document_from_supabase(document_id: str) -> Dict[str, Any]:
    """Fetch a specific document from Supabase by ID"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        client: Client = create_client(supabase_url, supabase_key)
        
        # Fetch document metadata
        result = client.table("documents").select("*").eq("id", document_id).execute()
        
        if not result.data:
            raise ValueError(f"Document {document_id} not found in Supabase")
        
        doc_meta = result.data[0]
        
        # Fetch actual document content from storage
        # This depends on where the document is stored
        if doc_meta.get("storage_path"):
            # Document is in Supabase storage
            storage_path = doc_meta["storage_path"]
            bucket_name = storage_path.split("/")[0]
            file_path = "/".join(storage_path.split("/")[1:])
            
            # Download from Supabase storage
            file_data = client.storage.from_(bucket_name).download(file_path)
            
            return {
                "id": document_id,
                "name": doc_meta.get("name", "Unknown"),
                "content": file_data,
                "metadata": doc_meta.get("metadata", {}),
                "source_type": doc_meta.get("source_type", "upload"),
                "file_type": doc_meta.get("file_type", "pdf")
            }
        
        elif doc_meta.get("source_url"):
            # Document needs to be fetched from source
            source_url = doc_meta["source_url"]
            source_type = doc_meta.get("source_type", "unknown")
            
            if source_type == "google_drive":
                # Fetch from Google Drive
                from googleapiclient.discovery import build
                from google.oauth2 import service_account
                
                creds_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
                if not creds_path:
                    raise ValueError("GOOGLE_SERVICE_ACCOUNT_PATH not set")
                
                credentials = service_account.Credentials.from_service_account_file(
                    creds_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                
                service = build('drive', 'v3', credentials=credentials)
                
                # Extract file ID from URL
                import re
                file_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', source_url)
                if not file_id_match:
                    raise ValueError(f"Could not extract file ID from URL: {source_url}")
                
                file_id = file_id_match.group(1)
                
                # Download file content
                request = service.files().get_media(fileId=file_id)
                content = request.execute()
                
                return {
                    "id": document_id,
                    "name": doc_meta.get("name", "Unknown"),
                    "content": content,
                    "metadata": doc_meta.get("metadata", {}),
                    "source_type": "google_drive",
                    "file_type": doc_meta.get("file_type", "pdf")
                }
            
            elif source_type == "notion":
                # Fetch from Notion
                from notion_client import Client as NotionClient
                
                notion_token = os.getenv(f"NOTION_API_KEY_{doc_meta.get('security_level', 'PUBLIC').upper()}_ACCESS")
                if not notion_token:
                    notion_token = os.getenv("NOTION_API_KEY")
                
                if not notion_token:
                    raise ValueError("NOTION_API_KEY not set")
                
                notion = NotionClient(auth=notion_token)
                
                # Extract page ID from URL
                import re
                page_id_match = re.search(r'([a-f0-9]{32})', source_url.replace("-", ""))
                if not page_id_match:
                    raise ValueError(f"Could not extract page ID from URL: {source_url}")
                
                page_id = page_id_match.group(1)
                
                # Fetch page content
                page = notion.pages.retrieve(page_id=page_id)
                blocks = notion.blocks.children.list(page_id=page_id)
                
                # Convert to markdown
                content = convert_notion_to_markdown(page, blocks)
                
                return {
                    "id": document_id,
                    "name": doc_meta.get("name", page.get("properties", {}).get("title", {}).get("title", [{}])[0].get("text", {}).get("content", "Unknown")),
                    "content": content.encode('utf-8'),
                    "metadata": doc_meta.get("metadata", {}),
                    "source_type": "notion",
                    "file_type": "markdown"
                }
            
            else:
                raise ValueError(f"Unknown source type: {source_type}")
        
        else:
            # Document content might be inline
            if doc_meta.get("content"):
                return {
                    "id": document_id,
                    "name": doc_meta.get("name", "Unknown"),
                    "content": doc_meta["content"].encode('utf-8'),
                    "metadata": doc_meta.get("metadata", {}),
                    "source_type": doc_meta.get("source_type", "text"),
                    "file_type": doc_meta.get("file_type", "txt")
                }
            else:
                raise ValueError(f"Document {document_id} has no content or source information")
        
    except Exception as e:
        logger.error(f"Error fetching document {document_id}: {e}")
        raise


def convert_notion_to_markdown(page: dict, blocks: dict) -> str:
    """Convert Notion page and blocks to markdown"""
    # Simple conversion - can be enhanced
    lines = []
    
    # Add title
    title = page.get("properties", {}).get("title", {}).get("title", [{}])[0].get("text", {}).get("content", "")
    if title:
        lines.append(f"# {title}\n")
    
    # Process blocks
    for block in blocks.get("results", []):
        block_type = block.get("type")
        
        if block_type == "paragraph":
            text = block.get("paragraph", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            if text:
                lines.append(f"{text}\n")
        
        elif block_type == "heading_1":
            text = block.get("heading_1", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            if text:
                lines.append(f"# {text}\n")
        
        elif block_type == "heading_2":
            text = block.get("heading_2", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            if text:
                lines.append(f"## {text}\n")
        
        elif block_type == "heading_3":
            text = block.get("heading_3", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            if text:
                lines.append(f"### {text}\n")
        
        elif block_type == "bulleted_list_item":
            text = block.get("bulleted_list_item", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            if text:
                lines.append(f"- {text}\n")
        
        elif block_type == "numbered_list_item":
            text = block.get("numbered_list_item", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            if text:
                lines.append(f"1. {text}\n")
        
        elif block_type == "code":
            code = block.get("code", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            language = block.get("code", {}).get("language", "")
            if code:
                lines.append(f"```{language}\n{code}\n```\n")
    
    return "\n".join(lines)


@cocoindex.flow_def(name="DocumentIngestionByID")
def document_ingestion_by_id_flow(flow_builder: FlowBuilder, data_scope: DataScope, document_id: str):
    """
    Process a specific document by ID from Supabase
    
    Args:
        document_id: UUID of the document to process
    """
    
    # Configure database connections (same as main flow)
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
            database="postgres",
            user="postgres",
            password=supabase_password,
            schema="cocoindex"
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
            api_key=os.getenv("QDRANT_API_KEY")
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
    
    # Fetch the specific document from Supabase
    logger.info(f"Fetching document {document_id} from Supabase")
    doc_data = fetch_document_from_supabase(document_id)
    
    # Create a custom source for this single document
    # We'll process it as a single item
    data_scope["document"] = doc_data
    
    # Add collectors
    chunks_output = data_scope.add_collector()
    entities_output = data_scope.add_collector()
    relationships_output = data_scope.add_collector()
    
    # Process the document
    with data_scope["document"] as doc:
        # Parse document based on file type
        if doc["file_type"] in ["pdf", "docx"]:
            doc["parsed_content"] = doc["content"].transform(
                parse_document_with_llamaparse,
                tier="balanced"
            )
        else:
            # Text/markdown files don't need parsing
            doc["parsed_content"] = doc["content"].decode('utf-8', errors='ignore')
        
        # Extract metadata
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
        
        # Chunk the document
        doc["chunks"] = doc["parsed_content"].transform(
            SplitRecursively(),
            language="markdown",
            chunk_size=1500,
            chunk_overlap=200,
            min_chunk_size=100
        )
        
        # Extract entities
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
        
        # Extract relationships
        doc["relationships"] = doc["parsed_content"].transform(
            extract_relationships,
            entities=doc["llm_entities"]
        )
        
        # Process chunks
        with doc["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                EmbedText(
                    model="text-embedding-3-small",
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            )
            
            chunks_output.collect(
                document_id=doc["id"],
                chunk_text=chunk["text"],
                chunk_embedding=chunk["embedding"],
                chunk_position=chunk["position"],
                metadata=doc["metadata"],
                security_level=doc["metadata"]["security_level"]
            )
        
        # Collect entities
        for entity in doc["llm_entities"]:
            entities_output.collect(
                document_id=doc["id"],
                entity_name=entity["name"],
                entity_type=entity["type"],
                confidence=entity["confidence"],
                properties=entity["properties"]
            )
        
        # Collect relationships
        for rel in doc["relationships"]:
            relationships_output.collect(
                document_id=doc["id"],
                source_entity=rel["source_entity"],
                target_entity=rel["target_entity"],
                relationship_type=rel["relationship_type"],
                confidence=rel["confidence"]
            )
    
    # Export to targets (same as main flow)
    chunks_output.export(
        "document_chunks",
        Qdrant(
            connection=qdrant_conn,
            collection_name="document_chunks",
            vector_field="chunk_embedding",
            payload_fields=["document_id", "chunk_text", "metadata", "security_level"]
        )
    )
    
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
    
    # Update document metadata in Supabase
    chunks_output.export(
        "app_metadata",
        Postgres(
            connection=postgres_conn,
            table_name="document_chunks",
            schema="public"
        )
    )
    
    logger.info(f"Successfully processed document {document_id}")


def run_document_by_id(document_id: str):
    """Run the flow for a specific document"""
    try:
        # Initialize CocoIndex
        cocoindex.init()
        
        # Create flow builder and data scope
        flow_builder = cocoindex.FlowBuilder()
        data_scope = cocoindex.DataScope()
        
        # Build the flow with document ID
        document_ingestion_by_id_flow(flow_builder, data_scope, document_id)
        
        # Get the built flow
        flow = flow_builder.build()
        
        logger.info(f"Starting processing for document {document_id}...")
        result = flow.run()
        
        if hasattr(result, 'success'):
            if result.success:
                logger.info(f"Successfully processed document {document_id}")
            else:
                logger.error(f"Failed to process document {document_id}: {getattr(result, 'error', 'Unknown error')}")
        else:
            logger.info(f"Flow completed for document {document_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python document_ingestion_by_id.py <document_id>")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    result = run_document_by_id(doc_id)