"""
Fix scan functions to use Supabase client instead of SQLAlchemy
"""

import sys
import io

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("FIXING SCAN FUNCTIONS TO USE SUPABASE CLIENT")
print("=" * 60)

# Read the current processing.py file
with open('app/api/processing.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the imports
old_imports = """from app.services.database import get_db_session, DocumentTable"""
new_imports = """from app.services.supabase_service import SupabaseService"""

content = content.replace(old_imports, new_imports)

# Replace scan_notion_workspace function database operations
old_notion_db = """        # Check for changes
        db = next(get_db_session())
        new_count = 0
        updated_count = 0
        
        for idx, doc_data in enumerate(documents):
            progress = 10 + (80 * idx // len(documents))
            job_tracker.update_job(
                job_id, 
                JobStatus.RUNNING, 
                message=f"Processing document {idx+1}/{len(documents)}", 
                progress=progress
            )
            
            # Check if document exists
            existing = db.query(DocumentTable).filter(
                DocumentTable.source_type == "notion",
                DocumentTable.source_id == doc_data["source_id"]
            ).first()
            
            if existing:
                # Check for updates
                if force_update or doc_data.get("last_edited") > existing.updated_at:
                    existing.content = doc_data["content"]
                    existing.doc_metadata = {**existing.doc_metadata, **doc_data.get("metadata", {})}
                    existing.status = "processing"
                    existing.updated_at = datetime.utcnow()
                    updated_count += 1
            else:
                # Create new document
                document = DocumentTable(
                    title=doc_data["title"],
                    source_type="notion",
                    source_id=doc_data["source_id"],
                    content=doc_data["content"],
                    status="discovered",
                    security_level=security_level,
                    access_level=connector.get_access_level(security_level),
                    doc_metadata={
                        **doc_data.get("metadata", {}),
                        "content_hash": connector.get_content_hash(doc_data["content"])
                    }
                )
                db.add(document)
                new_count += 1
        
        db.commit()"""

new_notion_db = """        # Check for changes
        supabase_service = SupabaseService()
        new_count = 0
        updated_count = 0
        
        for idx, doc_data in enumerate(documents):
            progress = 10 + (80 * idx // len(documents))
            job_tracker.update_job(
                job_id, 
                JobStatus.RUNNING, 
                message=f"Processing document {idx+1}/{len(documents)}", 
                progress=progress
            )
            
            # Check if document exists using Supabase
            existing_docs = supabase_service.list_documents(
                source_type="notion",
                source_id=doc_data["source_id"]
            )
            
            if existing_docs:
                # Check for updates
                existing = existing_docs[0]
                if force_update or doc_data.get("last_edited") > existing.get("updated_at", ""):
                    # Update existing document
                    supabase_service.update_document(
                        existing["id"],
                        {
                            "content": doc_data["content"],
                            "doc_metadata": {**existing.get("doc_metadata", {}), **doc_data.get("metadata", {})},
                            "status": "processing",
                            "updated_at": datetime.utcnow().isoformat()
                        }
                    )
                    updated_count += 1
            else:
                # Create new document
                doc_id = supabase_service.create_document({
                    "name": doc_data["title"],
                    "source_type": "notion",
                    "source_id": doc_data["source_id"],
                    "content": doc_data["content"],
                    "status": "discovered",
                    "security_level": security_level,
                    "access_level": connector.get_access_level(security_level),
                    "doc_metadata": {
                        **doc_data.get("metadata", {}),
                        "content_hash": connector.get_content_hash(doc_data["content"])
                    }
                })
                new_count += 1"""

content = content.replace(old_notion_db, new_notion_db)

# Replace scan_google_drive function database operations
old_gdrive_db = """        # Check for changes
        db = next(get_db_session())
        new_count = 0
        updated_count = 0
        
        for idx, doc_data in enumerate(documents):
            progress = 10 + (80 * idx // len(documents))
            job_tracker.update_job(
                job_id, 
                JobStatus.RUNNING, 
                message=f"Processing document {idx+1}/{len(documents)}", 
                progress=progress
            )
            
            # Check if document exists
            existing = db.query(DocumentTable).filter(
                DocumentTable.source_type == "google_drive",
                DocumentTable.source_id == doc_data["source_id"]
            ).first()
            
            if existing:
                # Check for updates
                if force_update or doc_data.get("last_modified") > existing.updated_at:
                    existing.content = doc_data["content"]
                    existing.doc_metadata = {**existing.doc_metadata, **doc_data.get("metadata", {})}
                    existing.status = "processing"
                    existing.updated_at = datetime.utcnow()
                    updated_count += 1
            else:
                # Create new document
                document = DocumentTable(
                    title=doc_data["title"],
                    source_type="google_drive",
                    source_id=doc_data["source_id"],
                    content=doc_data["content"],
                    status="discovered",
                    security_level=security_level,
                    access_level=connector.get_access_level(security_level),
                    doc_metadata={
                        **doc_data.get("metadata", {}),
                        "content_hash": connector.get_content_hash(doc_data["content"])
                    }
                )
                db.add(document)
                new_count += 1
        
        db.commit()"""

new_gdrive_db = """        # Check for changes
        supabase_service = SupabaseService()
        new_count = 0
        updated_count = 0
        
        for idx, doc_data in enumerate(documents):
            progress = 10 + (80 * idx // len(documents))
            job_tracker.update_job(
                job_id, 
                JobStatus.RUNNING, 
                message=f"Processing document {idx+1}/{len(documents)}", 
                progress=progress
            )
            
            # Check if document exists using Supabase
            existing_docs = supabase_service.list_documents(
                source_type="google_drive",
                source_id=doc_data["source_id"]
            )
            
            if existing_docs:
                # Check for updates
                existing = existing_docs[0]
                if force_update or doc_data.get("last_modified") > existing.get("updated_at", ""):
                    # Update existing document
                    supabase_service.update_document(
                        existing["id"],
                        {
                            "content": doc_data["content"],
                            "doc_metadata": {**existing.get("doc_metadata", {}), **doc_data.get("metadata", {})},
                            "status": "processing",
                            "updated_at": datetime.utcnow().isoformat()
                        }
                    )
                    updated_count += 1
            else:
                # Create new document
                doc_id = supabase_service.create_document({
                    "name": doc_data["title"],
                    "source_type": "google_drive",
                    "source_id": doc_data["source_id"],
                    "content": doc_data["content"],
                    "status": "discovered",
                    "security_level": security_level,
                    "access_level": connector.get_access_level(security_level),
                    "doc_metadata": {
                        **doc_data.get("metadata", {}),
                        "content_hash": connector.get_content_hash(doc_data["content"])
                    }
                })
                new_count += 1"""

content = content.replace(old_gdrive_db, new_gdrive_db)

# Remove the db parameter from the endpoint functions
content = content.replace(
    "    db=Depends(get_db_session)",
    ""
)

# Write back the modified content
with open('app/api/processing.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Fixed scan functions to use Supabase client")
print("\nChanges made:")
print("1. Replaced SQLAlchemy imports with SupabaseService")
print("2. Updated scan_notion_workspace to use Supabase client")
print("3. Updated scan_google_drive to use Supabase client")
print("4. Removed db dependency injection")
print("\nThe scan functions should now work with the Supabase cloud database!")