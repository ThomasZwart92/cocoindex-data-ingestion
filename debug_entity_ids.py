"""Debug entity ID issues in relationships."""

from app.services.supabase_service import SupabaseService

supabase = SupabaseService()

# Get document
doc_res = supabase.client.table('documents').select('id').eq('name', 'NC2056 - Black Screen & Config File corruption').execute()
doc_id = doc_res.data[0]['id']

# Get relationships
rel_res = supabase.client.table('canonical_relationships').select('*').eq('metadata->>document_id', doc_id).execute()
relationships = rel_res.data or []

# Collect all entity IDs from relationships
entity_ids = set()
for r in relationships:
    entity_ids.add(r['source_entity_id'])
    entity_ids.add(r['target_entity_id'])

print(f"Found {len(entity_ids)} unique entity IDs in relationships")

# Check which are valid canonical entities
print(f"Checking which IDs exist in canonical_entities table...")
valid_res = supabase.client.table('canonical_entities').select('id').in_('id', list(entity_ids)).execute()
valid_ids = {e['id'] for e in (valid_res.data or [])}
print(f"Found {len(valid_ids)} valid canonical entity IDs")

invalid_ids = entity_ids - valid_ids
print(f"Found {len(invalid_ids)} invalid IDs")

if invalid_ids:
    print(f"\n{len(invalid_ids)} IDs are NOT canonical entities!")
    print("Checking if they are entity mention IDs...")

    for invalid_id in list(invalid_ids)[:10]:
        # Check entity_mentions table
        mention_res = supabase.client.table('entity_mentions').select('text, type, canonical_entity_id').eq('id', invalid_id).execute()
        if mention_res.data:
            m = mention_res.data[0]
            canonical = m.get('canonical_entity_id')
            print(f"\n{invalid_id[:8]}...")
            print(f"  IS A MENTION: '{m['text']}' (type: {m.get('type')})")
            print(f"  Canonical ID: {canonical[:8] + '...' if canonical else 'NONE!'}")
        else:
            print(f"\n{invalid_id[:8]}... NOT FOUND in entity_mentions either!")

    print("\n\nThis confirms the bug: relationships are using entity mention IDs")
    print("instead of canonical entity IDs when canonical_entity_id is None!")