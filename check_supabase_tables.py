"""Check existing Supabase tables"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import sys

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def check_tables():
    """Check which tables exist in Supabase"""
    
    print("Checking Supabase tables...")
    print(f"URL: {SUPABASE_URL}")
    print("=" * 60)
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        tables_to_check = [
            "documents",
            "document_chunks",
            "chunks", 
            "entities",
            "entity_relationships",
            "processing_jobs",
            "llm_comparisons",
            "document_metadata",
            "document_images",
            "ingestion_queue",
            "source_configs"
        ]
        
        existing_tables = []
        missing_tables = []
        
        for table_name in tables_to_check:
            try:
                # Try to query the table
                result = supabase.table(table_name).select("*").limit(1).execute()
                print(f"[OK] Table '{table_name}' exists")
                existing_tables.append(table_name)
            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg or "relation" in error_msg:
                    print(f"[MISSING] Table '{table_name}' does not exist")
                    missing_tables.append(table_name)
                else:
                    print(f"[WARNING] Table '{table_name}' - error: {error_msg[:100]}")
        
        print("\n" + "=" * 60)
        print(f"Summary: {len(existing_tables)} tables exist, {len(missing_tables)} missing")
        
        if existing_tables:
            print("\nExisting tables:")
            for table in existing_tables:
                print(f"  - {table}")
        
        if missing_tables:
            print("\nMissing tables:")
            for table in missing_tables:
                print(f"  - {table}")
                
        return existing_tables, missing_tables
        
    except Exception as e:
        print(f"\nError connecting to Supabase: {e}")
        return [], []

if __name__ == "__main__":
    existing, missing = check_tables()
    sys.exit(0 if not missing else 1)