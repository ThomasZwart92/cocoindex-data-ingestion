"""
Migration script to standardize existing relationship types to the fixed vocabulary.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.supabase_service import SupabaseService
from app.utils.relationship_types import canonicalize_relationship_type, RELATIONSHIP_TYPES_CANONICAL
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_relationship_types():
    """Standardize all relationship types to the canonical vocabulary."""

    supabase = SupabaseService()

    logger.info("Fetching all canonical relationships...")
    all_relationships_res = supabase.client.table("canonical_relationships") \
        .select("*") \
        .execute()

    relationships = all_relationships_res.data or []
    logger.info(f"Found {len(relationships)} canonical relationships")

    # Track statistics
    stats = {
        "total": len(relationships),
        "already_canonical": 0,
        "migrated": 0,
        "failed": 0,
    }

    # Track non-standard types found
    non_standard_types = {}

    for rel in relationships:
        rel_id = rel["id"]
        current_type = rel.get("relationship_type", "")

        # Canonicalize the type using the fixed function
        canonical_type = canonicalize_relationship_type(current_type)

        # Check if it's already canonical
        if current_type == canonical_type:
            stats["already_canonical"] += 1
            continue

        # Track non-standard type
        if current_type not in non_standard_types:
            non_standard_types[current_type] = {
                "canonical": canonical_type,
                "count": 0
            }
        non_standard_types[current_type]["count"] += 1

        # Update to canonical type
        try:
            supabase.client.table("canonical_relationships") \
                .update({"relationship_type": canonical_type}) \
                .eq("id", rel_id) \
                .execute()
            stats["migrated"] += 1
            logger.debug(f"Migrated '{current_type}' -> '{canonical_type}'")
        except Exception as e:
            logger.error(f"Failed to update relationship {rel_id}: {e}")
            stats["failed"] += 1

    # Report statistics
    logger.info("\n" + "="*50)
    logger.info("MIGRATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Total relationships: {stats['total']}")
    logger.info(f"Already canonical: {stats['already_canonical']}")
    logger.info(f"Successfully migrated: {stats['migrated']}")
    logger.info(f"Failed migrations: {stats['failed']}")

    if non_standard_types:
        logger.info("\nNon-standard types found and migrated:")
        for orig_type, info in sorted(non_standard_types.items(), key=lambda x: x[1]['count'], reverse=True):
            logger.info(f"  '{orig_type}' -> '{info['canonical']}' ({info['count']} occurrences)")

    # Verify final state
    logger.info("\nVerifying final state...")
    final_res = supabase.client.table("canonical_relationships") \
        .select("relationship_type") \
        .execute()

    final_types = set()
    for rel in (final_res.data or []):
        final_types.add(rel.get("relationship_type"))

    logger.info(f"Unique relationship types after migration: {len(final_types)}")
    logger.info(f"Expected canonical types: {len(RELATIONSHIP_TYPES_CANONICAL)}")

    # Check for any remaining non-canonical types
    non_canonical = final_types - RELATIONSHIP_TYPES_CANONICAL
    if non_canonical:
        logger.warning(f"WARNING: Found {len(non_canonical)} non-canonical types still present:")
        for nt in sorted(non_canonical):
            logger.warning(f"  - {nt}")
    else:
        logger.info("SUCCESS: All relationship types are now canonical!")


if __name__ == "__main__":
    migrate_relationship_types()