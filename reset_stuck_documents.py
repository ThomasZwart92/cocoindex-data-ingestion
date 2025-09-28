"""
Script to reset documents stuck in processing state back to discovered
"""
import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env file
load_dotenv()

# Get Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_KEY must be set in environment")
    exit(1)

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Find documents stuck in processing state
response = supabase.table("documents").select("*").eq("status", "processing").execute()
stuck_docs = response.data

print(f"Found {len(stuck_docs)} documents stuck in processing state")

if stuck_docs:
    for doc in stuck_docs:
        title = doc.get('title', doc.get('name', 'Untitled'))
        print(f"- {doc['id']}: {title}")
    
    # Reset to discovered state
    confirm = input("\nReset these documents to 'discovered' state? (y/n): ")
    if confirm.lower() == 'y':
        for doc in stuck_docs:
            title = doc.get('title', doc.get('name', 'Untitled'))
            result = supabase.table("documents").update({"status": "discovered"}).eq("id", doc['id']).execute()
            if result.data:
                print(f"[OK] Reset {title} to discovered")
            else:
                print(f"[FAIL] Failed to reset {title}")
        print("\nDone! Documents can now be reprocessed.")
    else:
        print("Cancelled.")
else:
    print("No stuck documents found.")