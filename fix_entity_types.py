"""Script to fix entity types in the database to use only prescribed types"""
import asyncio
import logging
from supabase import create_client, Client
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type mapping from incorrect types to correct prescribed types
TYPE_MAPPING = {
    # Problem-related mappings
    "IssueType": "problem",
    "Issue": "problem", 
    "Error": "problem",
    "Symptom": "problem",
    "Defect": "problem",
    "Failure": "problem",
    
    # Component-related mappings
    "HardwareConnection": "component",
    "Connection": "component",
    "Connector": "component",
    "Hardware": "component",
    
    # Chemical-related mappings
    "CleaningAgent": "chemical",
    "Cleaner": "chemical",
    "Substance": "chemical",
    "Solvent": "chemical",
    
    # Procedure-related mappings
    "Method": "procedure",
    "Process": "procedure",
    "Technique": "procedure",
    "Step": "procedure",
    
    # Tool-related mappings
    "Equipment": "tool",
    "Instrument": "tool",
    "Device": "tool",
    
    # Material-related mappings
    "Supply": "material",
    "Consumable": "material",
    "Resource": "material",
    
    # Condition-related mappings
    "Status": "condition",
    "Quality": "condition",
    "Degradation": "condition",
}

# Valid entity types (lowercase)
VALID_TYPES = {
    "person", "organization", "location", "date", "product",
    "component", "technology", "chemical", "procedure", 
    "specification", "system", "measurement", "problem",
    "condition", "state", "tool", "material", "concept", 
    "event", "other"
}

async def fix_entity_types():
    """Fix entity types in the database"""
    # Initialize Supabase client
    supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
    
    try:
        # Get all entities from entity_mentions table
        logger.info("Fetching all entity mentions...")
        response = supabase.table("entity_mentions").select("*").execute()
        entities = response.data
        logger.info(f"Found {len(entities)} entity mentions")
        
        # Track statistics
        stats = {
            "total": len(entities),
            "fixed": 0,
            "already_valid": 0,
            "type_counts": {}
        }
        
        # Process each entity
        for entity in entities:
            entity_id = entity.get("id")
            # Check both possible column names: entity_type and type
            current_type = entity.get("entity_type") or entity.get("type") or ""
            current_type = current_type.strip() if current_type else ""
            
            # Track type counts
            if current_type:
                stats["type_counts"][current_type] = stats["type_counts"].get(current_type, 0) + 1
            
            # Check if type needs fixing
            if current_type and current_type.lower() not in VALID_TYPES:
                # Try to map to correct type
                new_type = None
                
                # Check direct mapping (case-insensitive)
                for incorrect, correct in TYPE_MAPPING.items():
                    if current_type.lower() == incorrect.lower():
                        new_type = correct
                        break
                
                # If no mapping found, default to "other"
                if not new_type:
                    logger.warning(f"Unknown type '{current_type}' for entity {entity_id}, mapping to 'other'")
                    new_type = "other"
                
                # Update the entity - try both column names
                logger.info(f"Updating entity {entity_id}: '{current_type}' -> '{new_type}'")
                update_data = {}
                # Check which columns exist
                if "entity_type" in entity:
                    update_data["entity_type"] = new_type
                if "type" in entity:
                    update_data["type"] = new_type
                    
                if update_data:
                    supabase.table("entity_mentions").update(update_data).eq("id", entity_id).execute()
                
                stats["fixed"] += 1
            elif current_type.lower() in VALID_TYPES:
                # Type is valid but might need case normalization
                if current_type != current_type.lower():
                    logger.info(f"Normalizing case for entity {entity_id}: '{current_type}' -> '{current_type.lower()}'")
                    update_data = {}
                    # Check which columns exist
                    if "entity_type" in entity:
                        update_data["entity_type"] = current_type.lower()
                    if "type" in entity:
                        update_data["type"] = current_type.lower()
                    
                    if update_data:
                        supabase.table("entity_mentions").update(update_data).eq("id", entity_id).execute()
                    stats["fixed"] += 1
                else:
                    stats["already_valid"] += 1
            else:
                # No type set, default to "other"
                logger.info(f"Setting default type 'other' for entity {entity_id}")
                update_data = {}
                # Check which columns exist
                if "entity_type" in entity:
                    update_data["entity_type"] = "other"
                if "type" in entity:
                    update_data["type"] = "other"
                    
                if update_data:
                    supabase.table("entity_mentions").update(update_data).eq("id", entity_id).execute()
                stats["fixed"] += 1
        
        # Print statistics
        logger.info("\n=== Entity Type Fix Results ===")
        logger.info(f"Total entities: {stats['total']}")
        logger.info(f"Fixed: {stats['fixed']}")
        logger.info(f"Already valid: {stats['already_valid']}")
        logger.info("\nType distribution:")
        for entity_type, count in sorted(stats["type_counts"].items()):
            logger.info(f"  {entity_type}: {count}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fixing entity types: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(fix_entity_types())