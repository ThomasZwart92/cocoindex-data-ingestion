#!/usr/bin/env python
"""Deduplicate the actual NC2056 entities and show improvements"""

import json
from app.utils.entity_deduplication import EntityDeduplicator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load the actual NC2056 entities
    with open('nc2056_entities.json', 'r') as f:
        data = json.load(f)
    
    entities = data['all_entities']
    logger.info(f"Loaded {len(entities)} entities from NC2056")
    
    # Analyze duplicates
    deduplicator = EntityDeduplicator()
    
    # Find duplicate groups
    duplicate_groups = deduplicator.find_duplicates(
        entities, 
        threshold=0.75,  # Lower threshold to catch more variations
        cross_type=True  # Allow matching across types
    )
    
    logger.info(f"\n=== DUPLICATE ANALYSIS ===")
    logger.info(f"Found {len(duplicate_groups)} groups of duplicates")
    
    # Show duplicate groups
    for i, group in enumerate(duplicate_groups, 1):
        names = [f"{e['name']} ({e['type']})" for e in group]
        logger.info(f"\nGroup {i}: {names}")
    
    # Perform deduplication
    deduplicated, review_groups = deduplicator.deduplicate_entities(
        entities,
        auto_merge_threshold=0.90,  # High threshold for auto-merge
        review_threshold=0.75,       # Lower threshold for review
        cross_type=True
    )
    
    logger.info(f"\n=== DEDUPLICATION RESULTS ===")
    logger.info(f"Original entities: {len(entities)}")
    logger.info(f"After deduplication: {len(deduplicated)}")
    logger.info(f"Reduction: {len(entities) - len(deduplicated)} entities removed")
    logger.info(f"Groups needing review: {len(review_groups)}")
    
    # Show what would be merged
    logger.info(f"\n=== ENTITIES THAT WOULD BE MERGED ===")
    
    # Count by type
    type_counts_before = {}
    type_counts_after = {}
    
    for entity in entities:
        entity_type = entity.get('type', 'unknown')
        type_counts_before[entity_type] = type_counts_before.get(entity_type, 0) + 1
    
    for entity in deduplicated:
        entity_type = entity.get('type', 'unknown')
        type_counts_after[entity_type] = type_counts_after.get(entity_type, 0) + 1
    
    logger.info(f"\nEntity counts by type:")
    for entity_type in sorted(set(list(type_counts_before.keys()) + list(type_counts_after.keys()))):
        before = type_counts_before.get(entity_type, 0)
        after = type_counts_after.get(entity_type, 0)
        reduction = before - after
        if reduction > 0:
            logger.info(f"  {entity_type}: {before} -> {after} (-{reduction})")
        else:
            logger.info(f"  {entity_type}: {before} -> {after}")
    
    # Save deduplicated results
    output_data = {
        "document_id": data.get("document_id"),
        "original_count": len(entities),
        "deduplicated_count": len(deduplicated),
        "entities": deduplicated
    }
    
    with open('nc2056_entities_deduplicated.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nSaved deduplicated entities to nc2056_entities_deduplicated.json")
    
    return deduplicated

if __name__ == "__main__":
    deduplicated = main()