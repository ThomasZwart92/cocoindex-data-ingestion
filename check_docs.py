#!/usr/bin/env python3
"""Check what documents actually exist"""

from app.services.supabase_service import get_supabase_client

def check_docs():
    client = get_supabase_client()
    
    print("Checking existing documents...")
    docs = client.table("documents").select("*").limit(10).execute()
    print(f"Found {len(docs.data)} documents:")
    
    for doc in docs.data:
        print(f"  ID: {doc['id']}")
        print(f"  Title: {doc.get('title', 'N/A')}")
        print(f"  Status: {doc.get('status', 'N/A')}")
        print(f"  Created: {doc.get('created_at', 'N/A')}")
        print("  ---")
    
    if docs.data:
        # Use the first document to test entity mentions
        test_doc_id = docs.data[0]['id']
        print(f"\nChecking entity_mentions for document: {test_doc_id}")
        
        mentions = client.table("entity_mentions").select("*").eq("document_id", test_doc_id).execute()
        print(f"Found {len(mentions.data)} entity mentions")
        
        if mentions.data:
            print("Sample mentions:")
            for mention in mentions.data[:3]:
                print(f"  - {mention.get('text')} (canonical_id: {mention.get('canonical_entity_id')})")

if __name__ == "__main__":
    check_docs()