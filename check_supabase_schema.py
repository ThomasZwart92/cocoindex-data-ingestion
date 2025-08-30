"""Check Supabase table schemas"""
import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"Checking Supabase table schemas...")

try:
    # Create client
    supabase = create_client(url, key)
    
    # Check documents table
    print("\n1. Documents table:")
    docs = supabase.table("documents").select("*").limit(1).execute()
    if docs.data and len(docs.data) > 0:
        print("   Columns found:")
        for col in docs.data[0].keys():
            print(f"   - {col}")
    else:
        print("   Table is empty, inserting test row to check schema...")
        test_doc = {
            "name": "Schema Test",
            "source_type": "upload",
            "status": "discovered"
        }
        result = supabase.table("documents").insert(test_doc).execute()
        if result.data:
            print("   Columns from inserted row:")
            for col in result.data[0].keys():
                print(f"   - {col}")
            # Clean up
            supabase.table("documents").delete().eq("id", result.data[0]["id"]).execute()
    
    # Check chunks table
    print("\n2. Chunks table:")
    chunks = supabase.table("chunks").select("*").limit(1).execute()
    if chunks.data and len(chunks.data) > 0:
        print("   Columns found:")
        for col in chunks.data[0].keys():
            print(f"   - {col}")
    else:
        print("   Table is empty")
    
    # Check entities table
    print("\n3. Entities table:")
    entities = supabase.table("entities").select("*").limit(1).execute()
    if entities.data and len(entities.data) > 0:
        print("   Columns found:")
        for col in entities.data[0].keys():
            print(f"   - {col}")
    else:
        print("   Table is empty")
    
    print("\nSUCCESS: Schema check complete")
    
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e).__name__}")