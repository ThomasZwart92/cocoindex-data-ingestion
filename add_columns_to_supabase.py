"""
Add missing columns to Supabase documents table

Since we can't execute DDL through the API, this script generates the SQL
that needs to be run in the Supabase dashboard.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
PROJECT_ID = SUPABASE_URL.split("//")[1].split(".")[0] if SUPABASE_URL else "ycddtddyffnqqehmjsid"

sql = """
-- Add missing columns to documents table
-- This script adds columns that the application expects but are missing

-- Add access_level column for security (integer based)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS access_level INTEGER DEFAULT 3;

-- Add content column for document text
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS content TEXT;

-- Add security_level column (text based)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS security_level TEXT DEFAULT 'employee' 
CHECK (security_level IN ('public', 'client', 'partner', 'employee', 'management'));

-- Add storage_path for file storage
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS storage_path TEXT;

-- Add url for convenience
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS url TEXT;

-- Add comments for documentation
COMMENT ON COLUMN documents.access_level IS 'Security access level: 1=public, 2=client, 3=partner, 4=employee, 5=management';
COMMENT ON COLUMN documents.content IS 'Full text content of the document';
COMMENT ON COLUMN documents.security_level IS 'Security classification of the document';
COMMENT ON COLUMN documents.storage_path IS 'Path to stored file in Supabase Storage';
COMMENT ON COLUMN documents.url IS 'Direct URL to access the document';

-- Verify the columns were added
SELECT column_name, data_type, column_default 
FROM information_schema.columns 
WHERE table_name = 'documents' 
AND column_name IN ('access_level', 'content', 'security_level', 'storage_path', 'url')
ORDER BY column_name;
"""

print("=" * 70)
print("SUPABASE TABLE UPDATE REQUIRED")
print("=" * 70)
print()
print("The documents table is missing required columns.")
print("Please follow these steps to fix the issue:")
print()
print("1. Open your Supabase SQL Editor:")
print(f"   https://supabase.com/dashboard/project/{PROJECT_ID}/sql/new")
print()
print("2. Copy and paste the following SQL:")
print()
print("-" * 70)
print(sql)
print("-" * 70)
print()
print("3. Click 'Run' to execute the SQL")
print()
print("4. Once complete, the Notion scan should work properly")
print()
print("This will add the following columns to the documents table:")
print("  - access_level (INTEGER): Security level 1-5")
print("  - content (TEXT): Document text content")
print("  - security_level (TEXT): Security classification name")
print("  - storage_path (TEXT): File storage path")
print("  - url (TEXT): Direct document URL")
print()
print("=" * 70)