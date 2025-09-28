#!/usr/bin/env python
"""Manually clean duplicate entities from NC2056 in database"""

import os
from dotenv import load_dotenv
from app.services.supabase_service import SupabaseService
from app.utils.entity_deduplication import EntityDeduplicator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    # Initialize Supabase
    supabase = SupabaseService()
    
    document_id = "b8fab07d-9a19-47f3-a05c-a2ba246e62ce"  # NC2056
    
    # Get all entities for this document
    response = supabase.client.table("entities").select("*").eq("document_id", document_id).execute()
    entities = response.data
    
    logger.info(f"Found {len(entities)} entities for NC2056")
    
    # Convert to format expected by deduplicator
    entity_dicts = []
    for entity in entities:
        entity_dict = {
            'id': entity['id'],
            'name': entity.get('entity_name', entity.get('name', '')),
            'type': entity.get('entity_type', entity.get('type', '')),
            'confidence': entity.get('confidence', 0.5),
            'metadata': entity.get('metadata', {})
        }
        entity_dicts.append(entity_dict)
    
    # Deduplicate
    deduplicator = EntityDeduplicator()
    deduplicated, review_groups = deduplicator.deduplicate_entities(
        entity_dicts,
        auto_merge_threshold=0.90,
        review_threshold=0.80,
        cross_type=True
    )
    
    logger.info(f"After deduplication: {len(deduplicated)} entities (from {len(entities)})")
    logger.info(f"Will delete {len(entities) - len(deduplicated)} duplicate entities")
    
    # Get IDs to keep
    ids_to_keep = {e['id'] for e in deduplicated}
    
    # Delete entities not in deduplicated list
    ids_to_delete = []
    for entity in entities:
        if entity['id'] not in ids_to_keep:
            ids_to_delete.append(entity['id'])
    
    if ids_to_delete:
        logger.info(f"Deleting {len(ids_to_delete)} duplicate entities...")
        
        # Delete in batches
        batch_size = 50
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i+batch_size]
            supabase.client.table("entities").delete().in_("id", batch).execute()
            logger.info(f"Deleted batch {i//batch_size + 1}")
        
        # Update entity count in document metadata
        supabase.client.table("documents").update({
            "entity_count": len(deduplicated),
            "metadata": {
                "entity_count": len(deduplicated),
                "chunk_count": 25,  # Keep existing chunk count
                "deduplication_applied": True
            }
        }).eq("id", document_id).execute()
        
        logger.info(f"Successfully cleaned duplicates. Final count: {len(deduplicated)} entities")
    else:
        logger.info("No duplicates to delete")
    
    return len(deduplicated)

if __name__ == "__main__":
    final_count = main()
    print(f"\nFinal entity count: {final_count}")