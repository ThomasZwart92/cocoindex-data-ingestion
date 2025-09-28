"""Debug why entity names aren't found."""

from app.services.supabase_service import SupabaseService

supabase = SupabaseService()

# These are the problematic entity IDs
entity_ids = [
    'c45ec630-10ea-44e7-8934-4cd483f22441',
    'a4eb36fb-311e-468a-8bd6-58cc7d42cc97',
    '4ce3564d-62ca-44c3-8cc6-a8a8215d8711',
    '8eb40575-2014-4d01-97c7-95aaeead6b6c'
]

print("Checking if these entity IDs exist in canonical_entities...")
existing = supabase.client.table("canonical_entities").select("id, name, type").in_("id", entity_ids).execute()

if existing.data:
    print(f"\nFound {len(existing.data)} entities:")
    for e in existing.data:
        print(f"  {e['id'][:8]}... -> {e['name']} ({e['type']})")
else:
    print("\nNONE of these IDs exist in canonical_entities!")

    # Check if they're entity mentions instead
    print("\nChecking if they're entity mention IDs...")
    for eid in entity_ids[:2]:
        mention_res = supabase.client.table("entity_mentions").select("text, type, canonical_entity_id").eq("id", eid).execute()
        if mention_res.data:
            m = mention_res.data[0]
            print(f"  {eid[:8]}... IS a mention: '{m['text']}' -> canonical: {m.get('canonical_entity_id', 'NONE')}")