#!/usr/bin/env python3
"""Check canonical entities and their mapping"""

from app.services.supabase_service import get_supabase_client

def check_canonical():
    client = get_supabase_client()
    
    print("Checking canonical entities...")
    canonical = client.table("canonical_entities").select("*").limit(20).execute()
    print(f"Found {len(canonical.data)} canonical entities:")
    
    for ce in canonical.data:
        print(f"  ID: {ce['id']}")
        print(f"  Name: {ce.get('name', 'N/A')}")
        print(f"  Type: {ce.get('type', 'N/A')}")
        print("  ---")
    
    # Check if any entity mentions actually have canonical_entity_id
    print("\nChecking entity mentions with canonical_entity_id...")
    mentions_with_canonical = client.table("entity_mentions").select("*").not_.is_("canonical_entity_id", "null").execute()
    print(f"Found {len(mentions_with_canonical.data)} mentions with canonical_entity_id")
    
    for mention in mentions_with_canonical.data[:5]:
        print(f"  Text: {mention.get('text')} -> Canonical ID: {mention.get('canonical_entity_id')}")

if __name__ == "__main__":
    check_canonical()