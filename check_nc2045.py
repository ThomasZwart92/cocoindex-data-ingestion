#!/usr/bin/env python3
"""Check NC2045 document and its entities"""

from app.services.supabase_service import get_supabase_client

def check_nc2045():
    client = get_supabase_client()
    doc_id = "f286b4fa-06f1-4a09-b6d5-a1780b9079d1"
    
    print(f"Checking document {doc_id} (NC2045)...")
    
    # Get document
    doc = client.table("documents").select("*").eq("id", doc_id).execute()
    if doc.data:
        doc_data = doc.data[0]
        print(f"  Title: {doc_data.get('title', 'N/A')}")
        print(f"  Status: {doc_data.get('status', 'N/A')}")
        print(f"  Created: {doc_data.get('created_at', 'N/A')}")
        print(f"  Updated: {doc_data.get('updated_at', 'N/A')}")
        print()
        
        # Check chunks
        chunks = client.table("chunks").select("*").eq("document_id", doc_id).execute()
        print(f"  Total chunks: {len(chunks.data)}")
        
        # Check entity mentions
        mentions = client.table("entity_mentions").select("*").eq("document_id", doc_id).execute()
        print(f"  Total entity mentions: {len(mentions.data)}")
        
        # Check canonical entity mentions
        canonical_mentions = client.table("entity_mentions").select("*").eq("document_id", doc_id).not_.is_("canonical_entity_id", "null").execute()
        print(f"  Canonical entity mentions: {len(canonical_mentions.data)}")
        
        if canonical_mentions.data:
            print("\n  Sample canonicalized entities:")
            for mention in canonical_mentions.data[:10]:
                print(f"    - {mention.get('text')} (type: {mention.get('type')})")
                
            # Get unique canonical entity IDs
            canonical_ids = set()
            for mention in canonical_mentions.data:
                if mention.get('canonical_entity_id'):
                    canonical_ids.add(mention['canonical_entity_id'])
            
            print(f"\n  Unique canonical entities: {len(canonical_ids)}")
            
            # Fetch canonical entity details
            if canonical_ids:
                canonical_entities = client.table("canonical_entities").select("*").in_("id", list(canonical_ids)).execute()
                print(f"\n  Canonical entities in database:")
                for ce in canonical_entities.data[:10]:
                    print(f"    - {ce.get('name')} (type: {ce.get('type')})")
    else:
        print("Document not found!")

if __name__ == "__main__":
    check_nc2045()