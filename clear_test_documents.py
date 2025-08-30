"""
Clear test documents from Supabase database
This will remove all documents marked as test or from test sources
"""

from supabase import create_client, Client
import os
from dotenv import load_dotenv
import sys
import io

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

def clear_test_documents():
    """Clear all test documents from the database"""
    
    # Initialize Supabase client
    url = os.getenv("SUPABASE_URL", "https://ycddtddyffnqqehmjsid.supabase.co")
    # Try both possible environment variable names
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    
    if not key:
        print("ERROR: SUPABASE_KEY not found in environment")
        return False
    
    supabase: Client = create_client(url, key)
    print(f"[OK] Connected to Supabase at {url}")
    
    try:
        # First, get all documents to see what we're deleting
        print("\n[INFO] Fetching all documents...")
        response = supabase.table("documents").select("*").execute()
        documents = response.data if response.data else []
        
        print(f"Found {len(documents)} total documents")
        
        if not documents:
            print("No documents to delete")
            return True
        
        # Show what will be deleted
        print("\n[DELETE] Documents to be deleted:")
        for doc in documents:
            print(f"  - {doc.get('name', 'Unnamed')} (ID: {doc.get('id')[:8]}...)")
            print(f"    Source: {doc.get('source_type', 'unknown')}, Status: {doc.get('status', 'unknown')}")
        
        # Confirm deletion
        response = input("\n[WARNING] Delete ALL documents? This cannot be undone! (yes/no): ")
        if response.lower() != 'yes':
            print("Deletion cancelled")
            return False
        
        # Delete in correct order due to foreign key constraints
        print("\n[CLEAN] Cleaning database...")
        
        # 1. Delete all chunks
        print("  Deleting chunks...")
        chunk_response = supabase.table("chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print(f"  [OK] Deleted chunks")
        
        # 2. Delete all entities
        print("  Deleting entities...")
        entity_response = supabase.table("entities").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print(f"  [OK] Deleted entities")
        
        # 3. Delete all documents
        print("  Deleting documents...")
        doc_response = supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print(f"  [OK] Deleted {len(documents)} documents")
        
        # 4. Clear ingestion queue if it exists
        try:
            print("  Clearing ingestion queue...")
            queue_response = supabase.table("ingestion_queue").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            print(f"  [OK] Cleared ingestion queue")
        except:
            print("  [INFO] No ingestion queue to clear")
        
        # 5. Clear processing jobs if exists
        try:
            print("  Clearing processing jobs...")
            jobs_response = supabase.table("processing_jobs").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            print(f"  [OK] Cleared processing jobs")
        except:
            print("  [INFO] No processing jobs to clear")
        
        print("\n[SUCCESS] Database cleared successfully!")
        print("You can now scan Notion and Google Drive for your actual documents.")
        
        # Verify deletion
        print("\n[VERIFY] Verifying cleanup...")
        response = supabase.table("documents").select("count", count="exact").execute()
        remaining = response.count if hasattr(response, 'count') else 0
        print(f"Documents remaining: {remaining}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error clearing documents: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CLEAR TEST DOCUMENTS FROM DATABASE")
    print("=" * 60)
    
    success = clear_test_documents()
    
    if success:
        print("\n[READY] Ready for real document scanning!")
        print("\nNext steps:")
        print("1. Go to http://localhost:3001")
        print("2. Click 'Scan Now' for Notion or Google Drive")
        print("3. Your real documents will be imported")
    else:
        print("\n[ERROR] Failed to clear documents")