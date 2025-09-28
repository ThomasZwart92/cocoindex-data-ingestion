"""
Backfill script to fix canonical_entity_id assignments for existing mentions.
This addresses the case mismatch bug that prevented proper canonicalization.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.supabase_service import SupabaseService
from app.models.entity_v2 import CanonicalEntity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backfill_canonical_ids():
    """Fix canonical_entity_id for mentions that failed due to type case mismatch."""

    supabase = SupabaseService()

    # Get all mentions without canonical_entity_id
    logger.info("Fetching mentions without canonical_entity_id...")
    unassigned_res = supabase.client.table("entity_mentions") \
        .select("*") \
        .is_("canonical_entity_id", "null") \
        .execute()

    unassigned_mentions = unassigned_res.data or []
    logger.info(f"Found {len(unassigned_mentions)} mentions without canonical assignment")

    if not unassigned_mentions:
        logger.info("No unassigned mentions found. Exiting.")
        return

    # Group by unique (name, type) combinations
    unique_entities = {}
    for mention in unassigned_mentions:
        name = mention.get("text", "").strip()
        raw_type = mention.get("type", "concept")

        if not name:
            continue

        # Normalize the type (matching the fix in document_tasks.py)
        normalized_type = raw_type.strip().upper() if raw_type else "CONCEPT"
        key = (name.lower(), normalized_type)

        if key not in unique_entities:
            unique_entities[key] = {
                "name": name,
                "type": normalized_type,
                "raw_types": set(),
                "mention_ids": []
            }

        unique_entities[key]["raw_types"].add(raw_type)
        unique_entities[key]["mention_ids"].append(mention["id"])

    logger.info(f"Found {len(unique_entities)} unique entities to process")

    # Create CanonicalEntity objects
    canonical_entities = []
    for (name_lower, norm_type), data in unique_entities.items():
        ce = CanonicalEntity(
            name=data["name"],
            type=norm_type,
            aliases=list(data["raw_types"] - {norm_type})  # Store variant types as aliases
        )
        canonical_entities.append(ce)

    # Upsert and get mapping
    logger.info("Upserting canonical entities...")
    canonical_map = supabase.upsert_canonical_entities_map(canonical_entities)
    logger.info(f"Created/retrieved {len(canonical_map)} canonical entity mappings")

    # Update mentions with canonical_entity_id
    updates_made = 0
    failures = 0

    for (name_lower, norm_type), data in unique_entities.items():
        # Try different key combinations (matching the fix)
        key_upper = (name_lower, norm_type)
        key_lower = (name_lower, norm_type.lower())

        canonical_id = canonical_map.get(key_upper) or canonical_map.get(key_lower)

        if canonical_id:
            # Update all mentions with this canonical_id
            for mention_id in data["mention_ids"]:
                try:
                    supabase.client.table("entity_mentions") \
                        .update({"canonical_entity_id": canonical_id}) \
                        .eq("id", mention_id) \
                        .execute()
                    updates_made += 1
                except Exception as e:
                    logger.error(f"Failed to update mention {mention_id}: {e}")
                    failures += 1

            if updates_made % 100 == 0:
                logger.info(f"Updated {updates_made} mentions so far...")
        else:
            logger.warning(f"No canonical ID found for: {data['name']} ({norm_type})")

    logger.info(f"\nBackfill complete!")
    logger.info(f"Successfully updated: {updates_made} mentions")
    logger.info(f"Failed updates: {failures}")

    # Verify the fix
    logger.info("\nVerifying results...")
    remaining_res = supabase.client.table("entity_mentions") \
        .select("id", count="exact") \
        .is_("canonical_entity_id", "null") \
        .execute()

    remaining_count = getattr(remaining_res, 'count', 0) or 0
    logger.info(f"Mentions still without canonical_entity_id: {remaining_count}")

    # Trigger description generation for newly canonicalized entities
    if updates_made > 0:
        logger.info("\nTriggering description generation for updated canonical entities...")
        from app.services.canonical_description_service import CanonicalEntityDescriptionService

        desc_service = CanonicalEntityDescriptionService()

        # Get unique canonical IDs that were assigned
        assigned_canonical_ids = set()
        for (name_lower, norm_type), data in unique_entities.items():
            key_upper = (name_lower, norm_type)
            key_lower = (name_lower, norm_type.lower())
            canonical_id = canonical_map.get(key_upper) or canonical_map.get(key_lower)
            if canonical_id:
                assigned_canonical_ids.add(canonical_id)

        logger.info(f"Refreshing descriptions for {len(assigned_canonical_ids)} canonical entities...")

        for canonical_id in assigned_canonical_ids:
            try:
                # Get mentions for evidence
                mentions_res = supabase.client.table("entity_mentions") \
                    .select("*") \
                    .eq("canonical_entity_id", canonical_id) \
                    .limit(10) \
                    .execute()

                evidence_list = []
                for mention in (mentions_res.data or []):
                    evidence_list.append({
                        "mention": mention.get("text", ""),
                        "context": mention.get("attributes", {}).get("context", ""),
                        "summary": mention.get("attributes", {}).get("summary", "")
                    })

                if evidence_list:
                    desc_service.refresh_canonical_entity_description(
                        canonical_entity_id=canonical_id,
                        evidence_list=evidence_list
                    )
                    logger.debug(f"Updated description for canonical entity {canonical_id}")
            except Exception as e:
                logger.error(f"Failed to refresh description for {canonical_id}: {e}")

        logger.info("Description generation complete!")


if __name__ == "__main__":
    backfill_canonical_ids()