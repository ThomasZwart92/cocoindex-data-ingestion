"""Check the schema of the documents table"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def check_document_schema():
    """Check the schema of the documents table"""
    
    print("Checking documents table schema...")
    print("=" * 60)
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to get one document to see the columns
        result = supabase.table("documents").select("*").limit(1).execute()
        
        if result.data and len(result.data) > 0:
            print("Document columns found:")
            doc = result.data[0]
            for key in doc.keys():
                value_type = type(doc[key]).__name__
                print(f"  - {key}: {value_type}")
        else:
            # Try to insert a dummy document to see what columns are required
            print("No documents found. Attempting to check structure...")
            try:
                test_doc = {
                    "name": "test",
                    "status": "discovered"
                }
                result = supabase.table("documents").insert(test_doc).execute()
                if result.data:
                    doc = result.data[0]
                    print("Document columns (from insert):")
                    for key in doc.keys():
                        value_type = type(doc[key]).__name__
                        print(f"  - {key}: {value_type}")
                    # Delete the test document
                    supabase.table("documents").delete().eq("id", doc["id"]).execute()
            except Exception as e:
                print(f"Insert test failed: {e}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_document_schema()