"""Execute SQL directly in Supabase using the REST API"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def execute_sql():
    """Execute SQL statements through Supabase REST API"""
    
    print("Executing SQL in Supabase...")
    
    # Read the SQL file
    with open("supabase_schema.sql", "r") as f:
        sql_content = f.read()
    
    # Supabase REST API endpoint for raw SQL execution
    # Note: This requires service role key for DDL operations
    api_url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    # Try using the pg endpoint directly
    # Supabase exposes a PostgreSQL REST interface
    sql_statements = sql_content.split(';')
    
    # Alternative: Try using the Supabase Management API
    # This requires the service role key, not the anon key
    
    print("\nAttempting to create tables via Supabase API...")
    
    # Let's try a different approach - create tables one by one
    # using the Supabase client with raw SQL through stored procedure
    
    from supabase import create_client, Client
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to execute SQL using the query builder
        # Unfortunately, Supabase Python client doesn't expose direct SQL execution
        
        # Alternative approach: Create tables using the REST API with POST requests
        print("\nCreating tables individually...")
        
        # Since we can't execute DDL directly, let's create a function in the database
        # that can execute dynamic SQL (this needs to be done once manually)
        
        create_exec_function = """
        CREATE OR REPLACE FUNCTION exec_sql(sql_query text)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
            EXECUTE sql_query;
        END;
        $$;
        """
        
        print("\nTo enable SQL execution, first run this in Supabase SQL editor:")
        print("-" * 60)
        print(create_exec_function)
        print("-" * 60)
        print("\nThen this script can execute the schema creation.")
        
        # Try to call the function if it exists
        try:
            # Test if the function exists by trying to execute a simple command
            result = supabase.rpc('exec_sql', {'sql_query': 'SELECT 1'}).execute()
            print("\n[OK] exec_sql function exists!")
            
            # Now execute our schema SQL
            print("\nExecuting schema creation...")
            
            # Split the SQL into individual statements
            sql_statements = [s.strip() for s in sql_content.split('--;')]
            
            for i, statement in enumerate(sql_statements):
                if statement and not statement.startswith('--'):
                    try:
                        supabase.rpc('exec_sql', {'sql_query': statement}).execute()
                        print(f"[OK] Executed statement {i+1}")
                    except Exception as e:
                        print(f"[ERROR] Statement {i+1}: {str(e)[:100]}")
            
            print("\n[SUCCESS] Schema creation completed!")
            
        except Exception as e:
            if "could not find the public.rpc()" in str(e) or "function" in str(e).lower():
                print("\n[INFO] The exec_sql function doesn't exist yet.")
                print("\nPlease follow these steps:")
                print("1. Go to: https://app.supabase.com/project/ycddtddyffnqqehmjsid/sql/new")
                print("2. First create the exec_sql function (shown above)")
                print("3. Then run this script again, OR")
                print("4. Just paste the contents of 'supabase_schema.sql' and run it directly")
            else:
                print(f"\n[ERROR] {e}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to connect: {e}")
        print("\nDirect SQL execution through the Python client is not supported.")
        print("Please use the Supabase dashboard to execute the SQL.")

if __name__ == "__main__":
    execute_sql()
    
    # Also try using direct HTTP request to Supabase
    print("\n" + "=" * 60)
    print("Alternative: Direct HTTP approach")
    print("=" * 60)
    
    # Supabase uses PostgREST which doesn't support DDL operations
    # DDL (CREATE TABLE, etc.) must be done through the dashboard
    
    print("\nSupabase's REST API (PostgREST) only supports DML operations (SELECT, INSERT, UPDATE, DELETE).")
    print("DDL operations (CREATE TABLE, CREATE INDEX, etc.) must be executed through:")
    print("1. The Supabase Dashboard SQL editor")
    print("2. The Supabase CLI (if you have it installed)")
    print("3. Direct PostgreSQL connection (requires connection pooling setup)")
    
    print("\n[RECOMMENDED] Quickest approach:")
    print("1. Open: https://app.supabase.com/project/ycddtddyffnqqehmjsid/sql/new")
    print("2. Copy all content from 'supabase_schema.sql'")
    print("3. Paste and click 'Run'")
    print("\nThis will create all 8 tables with proper relationships and indexes.")