#!/usr/bin/env python3
"""Find a document with entity mentions that have canonical_entity_id"""

from app.services.supabase_service import get_supabase_client

def find_good_doc():
    client = get_supabase_client()
    
    print("Finding documents with entity mentions that have canonical_entity_id...")
    
    # Get entity mentions with canonical_entity_id
    mentions_with_canonical = client.table("entity_mentions").select("document_id,canonical_entity_id").not_.is_("canonical_entity_id", "null").limit(10).execute()
    
    if mentions_with_canonical.data:
        doc_ids = set()
        for mention in mentions_with_canonical.data:
            doc_ids.add(mention['document_id'])
        
        print(f"Found {len(doc_ids)} documents with properly canonicalized entities:")
        for doc_id in list(doc_ids)[:3]:  # Show first 3
            print(f"  Document ID: {doc_id}")
            
            # Count mentions for this document
            all_mentions = client.table("entity_mentions").select("*").eq("document_id", doc_id).execute()
            canonical_mentions = client.table("entity_mentions").select("*").eq("document_id", doc_id).not_.is_("canonical_entity_id", "null").execute()
            
            print(f"    Total mentions: {len(all_mentions.data)}")
            print(f"    Canonical mentions: {len(canonical_mentions.data)}")
            
            if canonical_mentions.data:
                print("    Sample canonical mentions:")
                for mention in canonical_mentions.data[:3]:
                    print(f"      - {mention.get('text')} -> {mention.get('canonical_entity_id')}")
            print()

if __name__ == "__main__":
    find_good_doc()