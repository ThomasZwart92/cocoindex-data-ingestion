"""
Cleanup script to remove duplicate relationships that accumulated from document reprocessing.
This identifies and removes duplicate relationships while preserving manually created ones.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.supabase_service import SupabaseService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_duplicate_relationships():
    """Remove duplicate relationships while preserving manually created ones."""

    supabase = SupabaseService()

    logger.info("Fetching all canonical relationships...")

    try:
        all_relationships = supabase.client.table("canonical_relationships") \
            .select("*") \
            .execute()
    except Exception as e:
        logger.error(f"Failed to fetch relationships: {e}")
        return

    relationships = all_relationships.data or []
    logger.info(f"Found {len(relationships)} total relationships")

    # Group relationships by key (source, target, type)
    grouped = defaultdict(list)
    for rel in relationships:
        key = (
            rel.get("source_entity_id"),
            rel.get("target_entity_id"),
            rel.get("relationship_type")
        )
        grouped[key].append(rel)

    # Find duplicates
    duplicates_to_delete = []
    groups_with_duplicates = 0

    for key, rels in grouped.items():
        if len(rels) > 1:
            groups_with_duplicates += 1

            # Sort by priority:
            # 1. Keep manually created (is_verified=True or metadata.manual=True)
            # 2. Keep most recent
            # 3. Keep highest confidence

            def get_priority(rel):
                is_manual = (
                    rel.get("is_verified") or
                    (rel.get("metadata") or {}).get("manual") or
                    not (rel.get("metadata") or {}).get("document_id")  # No document_id likely means manual
                )
                created_at = rel.get("created_at", "")
                confidence = rel.get("confidence_score", 0)

                # Return tuple for sorting (higher priority first)
                return (is_manual, created_at, confidence)

            # Sort relationships by priority (keep the best one)
            sorted_rels = sorted(rels, key=get_priority, reverse=True)

            # Keep the first (best) one, delete the rest
            to_keep = sorted_rels[0]
            to_delete = sorted_rels[1:]

            logger.info(
                f"Found {len(rels)} duplicates for {key[0][:8]}... -> {key[2]} -> {key[1][:8]}..."
            )
            logger.info(f"  Keeping: {to_keep['id'][:8]}... (verified={to_keep.get('is_verified')}, "
                       f"manual={(to_keep.get('metadata') or {}).get('manual')})")

            for rel in to_delete:
                duplicates_to_delete.append(rel["id"])
                logger.info(f"  Deleting: {rel['id'][:8]}... (verified={rel.get('is_verified')}, "
                          f"manual={(rel.get('metadata') or {}).get('manual')})")

    logger.info(f"\nFound {groups_with_duplicates} groups with duplicates")
    logger.info(f"Total relationships to delete: {len(duplicates_to_delete)}")

    if duplicates_to_delete:
        user_input = input("\nProceed with deletion? (y/n): ")
        if user_input.lower() == 'y':
            # Delete in batches
            batch_size = 50
            deleted_count = 0

            for i in range(0, len(duplicates_to_delete), batch_size):
                batch = duplicates_to_delete[i:i + batch_size]
                try:
                    supabase.client.table("canonical_relationships") \
                        .delete() \
                        .in_("id", batch) \
                        .execute()
                    deleted_count += len(batch)
                    logger.info(f"Deleted batch {i//batch_size + 1}: {len(batch)} relationships")
                except Exception as e:
                    logger.error(f"Failed to delete batch: {e}")

            logger.info(f"\nSuccessfully deleted {deleted_count} duplicate relationships")
        else:
            logger.info("Deletion cancelled")
    else:
        logger.info("No duplicates found!")

    # Verify final state
    logger.info("\nVerifying final state...")
    try:
        final_relationships = supabase.client.table("canonical_relationships") \
            .select("id", count="exact") \
            .execute()

        final_count = getattr(final_relationships, 'count', 0) or 0
        logger.info(f"Final relationship count: {final_count}")
        logger.info(f"Removed: {len(relationships) - final_count} relationships")
    except Exception as e:
        logger.error(f"Failed to verify final state: {e}")


if __name__ == "__main__":
    cleanup_duplicate_relationships()