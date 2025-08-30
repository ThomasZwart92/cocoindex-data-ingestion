"""Setup Supabase tables for the document ingestion portal"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import requests

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def setup_tables():
    """Execute SQL to create all required tables"""
    
    print("Setting up Supabase tables...")
    print(f"URL: {SUPABASE_URL}")
    
    # Read the SQL file
    with open("supabase_schema.sql", "r") as f:
        sql_script = f.read()
    
    # Unfortunately, Supabase Python client doesn't support direct SQL execution
    # We need to use the REST API directly
    
    print("\n" + "=" * 60)
    print("MANUAL SETUP REQUIRED")
    print("=" * 60)
    print("\nThe Supabase Python client doesn't support direct SQL execution.")
    print("Please follow these steps to create the tables:\n")
    print("1. Go to your Supabase dashboard: https://app.supabase.com")
    print("2. Select your project")
    print("3. Click on 'SQL Editor' in the left sidebar")
    print("4. Click 'New query'")
    print("5. Copy and paste the contents of 'supabase_schema.sql'")
    print("6. Click 'Run' to execute the SQL\n")
    print("The SQL file creates the following tables:")
    print("  - documents (main document registry)")
    print("  - chunks (document chunks)")
    print("  - entities (extracted entities)")
    print("  - entity_relationships (entity connections)")
    print("  - processing_jobs (async job tracking)")
    print("  - llm_comparisons (multi-model outputs)")
    print("  - document_metadata (structured metadata)")
    print("  - document_images (extracted images)")
    print("\nAll tables include:")
    print("  - Proper indexes for performance")
    print("  - Updated_at triggers")
    print("  - Foreign key relationships")
    print("  - Check constraints for data integrity")
    
    # Test if tables exist
    print("\n" + "=" * 60)
    print("Testing current table status...")
    print("=" * 60)
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        tables_to_check = [
            "documents",
            "chunks", 
            "entities",
            "entity_relationships",
            "processing_jobs",
            "llm_comparisons",
            "document_metadata",
            "document_images"
        ]
        
        for table_name in tables_to_check:
            try:
                # Try to query the table
                result = supabase.table(table_name).select("*").limit(1).execute()
                print(f"✅ Table '{table_name}' exists")
            except Exception as e:
                if "does not exist" in str(e):
                    print(f"❌ Table '{table_name}' does not exist - needs to be created")
                else:
                    print(f"⚠️  Table '{table_name}' - error checking: {str(e)[:50]}")
    
    except Exception as e:
        print(f"\nError connecting to Supabase: {e}")
    
    print("\n" + "=" * 60)
    print("After creating the tables, run this script again to verify.")
    print("=" * 60)
    
    # Also save a direct link
    if SUPABASE_URL:
        project_ref = SUPABASE_URL.split('.')[0].replace('https://', '')
        sql_editor_url = f"https://app.supabase.com/project/{project_ref}/sql/new"
        print(f"\nDirect link to SQL editor:\n{sql_editor_url}")
    
    return True

def test_table_operations():
    """Test basic operations on the tables after they're created"""
    
    print("\n" + "=" * 60)
    print("Testing table operations...")
    print("=" * 60)
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test inserting a document
        test_doc = {
            "name": "test_setup.pdf",
            "source_type": "upload",
            "file_type": "pdf",
            "status": "discovered"
        }
        
        result = supabase.table("documents").insert(test_doc).execute()
        
        if result.data:
            doc_id = result.data[0]['id']
            print(f"✅ Successfully inserted test document with ID: {doc_id}")
            
            # Test updating the document
            update_result = supabase.table("documents").update({
                "status": "processing"
            }).eq("id", doc_id).execute()
            
            if update_result.data:
                print("✅ Successfully updated document status")
            
            # Test querying
            query_result = supabase.table("documents").select("*").eq("id", doc_id).execute()
            
            if query_result.data:
                print(f"✅ Successfully queried document: {query_result.data[0]['name']}")
            
            # Clean up test data
            delete_result = supabase.table("documents").delete().eq("id", doc_id).execute()
            print("✅ Cleaned up test data")
            
            print("\n🎉 All table operations working correctly!")
            return True
            
    except Exception as e:
        print(f"\n❌ Table operations failed: {e}")
        print("\nPlease create the tables first using the SQL script.")
        return False

if __name__ == "__main__":
    setup_tables()
    
    # Ask if user has created the tables
    print("\n" + "=" * 60)
    response = input("Have you created the tables in Supabase? (y/n): ")
    
    if response.lower() == 'y':
        test_table_operations()
    else:
        print("\nPlease create the tables first, then run this script again to test.")