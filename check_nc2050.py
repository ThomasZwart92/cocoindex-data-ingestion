#!/usr/bin/env python3
"""Check for NC2050 document in the fresh database"""

from app.services.supabase_service import get_supabase_client
import json

def check_nc2050():
    client = get_supabase_client()
    
    print("Looking for NC2050 document...")
    
    # Search for documents with NC2050 in title or content
    all_docs = client.table("documents").select("*").execute()
    
    nc2050_docs = []
    for doc in all_docs.data:
        title = doc.get('title', '')
        source_id = doc.get('source_id', '')
        metadata = doc.get('metadata', {})
        
        # Check if NC2050 is mentioned
        if 'NC2050' in str(title).upper() or 'NC2050' in str(source_id).upper() or 'NC2050' in str(metadata).upper():
            nc2050_docs.append(doc)
    
    if nc2050_docs:
        print(f"Found {len(nc2050_docs)} documents mentioning NC2050:")
        for doc in nc2050_docs:
            print(f"\n  ID: {doc['id']}")
            print(f"  Title: {doc.get('title', 'N/A')}")
            print(f"  Source: {doc.get('source_type', 'N/A')}")
            print(f"  Status: {doc.get('status', 'N/A')}")
            print(f"  Created: {doc.get('created_at', 'N/A')}")
            
            # Check entity mentions
            doc_id = doc['id']
            mentions = client.table("entity_mentions").select("*").eq("document_id", doc_id).execute()
            print(f"  Entity mentions: {len(mentions.data)}")
            
            canonical_mentions = client.table("entity_mentions").select("*").eq("document_id", doc_id).not_.is_("canonical_entity_id", "null").execute()
            print(f"  Canonical mentions: {len(canonical_mentions.data)}")
            
            if canonical_mentions.data:
                print("  Sample canonical entities:")
                for mention in canonical_mentions.data[:3]:
                    print(f"    - {mention.get('text')}")
    else:
        print("No documents found with NC2050. Listing all available documents:")
        print(f"Total documents in database: {len(all_docs.data)}")
        
        for doc in all_docs.data[:10]:  # Show first 10
            print(f"\n  ID: {doc['id']}")
            print(f"  Title: {doc.get('title', 'N/A')}")
            print(f"  Source: {doc.get('source_type', 'N/A')}")
            print(f"  Status: {doc.get('status', 'N/A')}")

if __name__ == "__main__":
    check_nc2050()