"""Create the document_state_transitions table in Supabase"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def create_transitions_table():
    """Check if transitions table exists and provide SQL to create it"""
    
    print("Checking for document_state_transitions table...")
    print("=" * 60)
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Check if table exists
        try:
            result = supabase.table("document_state_transitions").select("*").limit(1).execute()
            print("[OK] Table 'document_state_transitions' already exists")
            return True
        except Exception as e:
            if "does not exist" in str(e) or "not found" in str(e):
                print("[MISSING] Table 'document_state_transitions' does not exist")
                print("\nPlease create it using the following SQL in Supabase SQL Editor:")
                print("=" * 60)
                print("""
-- Create state transitions table
CREATE TABLE IF NOT EXISTS document_state_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    from_state VARCHAR(50) NOT NULL,
    to_state VARCHAR(50) NOT NULL,
    user_id TEXT,
    reason TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transitions_document_id ON document_state_transitions(document_id);
CREATE INDEX IF NOT EXISTS idx_transitions_created_at ON document_state_transitions(created_at);
CREATE INDEX IF NOT EXISTS idx_transitions_states ON document_state_transitions(from_state, to_state);

-- Grant permissions
GRANT ALL ON document_state_transitions TO authenticated;
GRANT ALL ON document_state_transitions TO service_role;
                """)
                print("=" * 60)
                print("\nDirect link to Supabase SQL Editor:")
                project_ref = SUPABASE_URL.split('.')[0].replace('https://', '')
                sql_editor_url = f"https://app.supabase.com/project/{project_ref}/sql/new"
                print(sql_editor_url)
                return False
            else:
                print(f"Error checking table: {e}")
                return False
                
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = create_transitions_table()
    if success:
        print("\nTable is ready for use!")
    else:
        print("\nPlease create the table using the SQL above, then run test_state_machine.py")