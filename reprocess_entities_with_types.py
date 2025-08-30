"""Re-process existing entities with new enhanced entity types"""
import asyncio
import os
from dotenv import load_dotenv
from app.services.supabase_service import SupabaseService
from app.processors.entity_extractor import EntityExtractor
from app.models.chunk import Chunk
from app.services.llm_service import LLMService, LLMProvider
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def reprocess_entities():
    """Re-process NC2050 document entities with new types"""
    
    # Document ID for NC2050
    document_id = "e86cbf9f-5fb7-4f5d-8ff9-44633a432a2d"
    
    # Initialize services
    supabase = SupabaseService()
    entity_extractor = EntityExtractor()
    llm_service = LLMService()
    
    logger.info(f"Re-processing entities for document {document_id}")
    
    try:
        # Get chunks for the document directly via HTTP
        import httpx
        
        headers = {
            'apikey': os.getenv('SUPABASE_KEY'),
            'Authorization': f'Bearer {os.getenv("SUPABASE_KEY")}'
        }
        
        response = httpx.get(
            f'{os.getenv("SUPABASE_URL")}/rest/v1/chunks?document_id=eq.{document_id}&order=chunk_index.asc',
            headers=headers
        )
        chunks_data = response.json()
        logger.info(f"Found {len(chunks_data)} chunks")
        
        # Convert to Chunk objects - only process parent chunks to avoid overlap
        parent_chunks = [c for c in chunks_data if c.get('parent_chunk_id') is None]
        logger.info(f"Found {len(parent_chunks)} parent chunks (no content overlap)")
        
        chunks = []
        for chunk_data in parent_chunks:  # Process all parent chunks
            chunk = Chunk(
                id=chunk_data['id'],
                document_id=document_id,
                chunk_text=chunk_data['chunk_text'],
                chunk_index=chunk_data['chunk_index'],
                chunking_strategy=chunk_data.get('chunking_strategy', 'semantic'),
                hierarchy_level=chunk_data.get('hierarchy_level', 1),
                parent_chunk_id=chunk_data.get('parent_chunk_id')
            )
            chunks.append(chunk)
        
        # Extract entities with new types
        logger.info("Extracting entities with enhanced types...")
        entities, relationships = await entity_extractor._extract_async(chunks, document_id)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        
        # Clear existing entities for this document
        logger.info("Clearing old entities...")
        # Direct HTTP delete since SupabaseService doesn't have this method
        delete_response = httpx.delete(
            f'{os.getenv("SUPABASE_URL")}/rest/v1/entities?document_id=eq.{document_id}',
            headers=headers
        )
        logger.info(f"Deleted existing entities: {delete_response.status_code}")
        
        # Save new entities
        logger.info("Saving new entities with enhanced metadata...")
        entity_dicts = []
        for entity in entities:
            entity_dict = entity.to_supabase_dict()
            # Ensure entity_name and entity_type are set for database
            entity_dict['entity_name'] = entity.name
            entity_dict['entity_type'] = entity.type.lower() if hasattr(entity.type, 'lower') else str(entity.type)
            entity_dict['confidence_score'] = entity.confidence
            # Remove fields that don't exist in database
            if 'confidence' in entity_dict:
                del entity_dict['confidence']
            if 'name' in entity_dict:
                del entity_dict['name']
            if 'type' in entity_dict:
                del entity_dict['type']
            entity_dicts.append(entity_dict)
        
        # Use direct HTTP POST to create entities
        for entity_dict in entity_dicts:
            post_response = httpx.post(
                f'{os.getenv("SUPABASE_URL")}/rest/v1/entities',
                headers=headers,
                json=entity_dict
            )
            if post_response.status_code not in [200, 201]:
                logger.error(f"Failed to create entity: {post_response.text}")
        
        # Save relationships
        logger.info("Saving relationships...")
        if relationships:
            for rel in relationships:
                rel_dict = rel.to_supabase_dict()
                post_response = httpx.post(
                    f'{os.getenv("SUPABASE_URL")}/rest/v1/entity_relationships',
                    headers=headers,
                    json=rel_dict
                )
                if post_response.status_code not in [200, 201]:
                    logger.error(f"Failed to create relationship: {post_response.text}")
        
        # Display summary
        logger.info("\n=== Entity Types Summary ===")
        entity_types = {}
        for entity in entities:
            entity_type = entity.type if isinstance(entity.type, str) else entity.type.value
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity.name)
        
        for entity_type, names in sorted(entity_types.items()):
            logger.info(f"\n{entity_type.upper()} ({len(names)} entities):")
            for name in names[:5]:  # Show first 5 of each type
                metadata = entity.metadata if hasattr(entity, 'metadata') else {}
                logger.info(f"  - {name}")
                if metadata.get('description'):
                    logger.info(f"    Description: {metadata['description']}")
                if metadata.get('category'):
                    logger.info(f"    Category: {metadata['category']}")
                if metadata.get('attributes'):
                    logger.info(f"    Attributes: {metadata['attributes']}")
        
        logger.info("\n✅ Successfully re-processed entities with enhanced types!")
        
    except Exception as e:
        logger.error(f"Error re-processing entities: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(reprocess_entities())