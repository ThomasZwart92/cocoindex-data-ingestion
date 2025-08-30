"""
Google Drive Connector - Fetches documents from Google Drive
"""
import os
import io
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import mimetypes

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pydantic import BaseModel

from app.utils.retry import retry_on_failure, retry_sync


class GoogleDriveDocument(BaseModel):
    """Represents a Google Drive document"""
    id: str
    name: str
    mime_type: str
    content: Optional[bytes] = None
    content_text: Optional[str] = None
    modified_time: datetime
    created_time: datetime
    web_view_link: str
    size: int
    parent_ids: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True


class GoogleDriveConnector:
    """Connector for fetching documents from Google Drive"""
    
    SUPPORTED_MIME_TYPES = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/msword': '.doc',
        'text/plain': '.txt',
        'text/markdown': '.md',
        'application/vnd.google-apps.document': '.gdoc',
        'application/vnd.google-apps.spreadsheet': '.gsheet',
        'application/vnd.google-apps.presentation': '.gslides',
    }
    
    GOOGLE_DOCS_EXPORT_FORMATS = {
        'application/vnd.google-apps.document': 'text/plain',
        'application/vnd.google-apps.spreadsheet': 'text/csv',
        'application/vnd.google-apps.presentation': 'text/plain',
    }
    
    def __init__(self, service_account_path: Optional[str] = None, security_level: str = "employee"):
        """Initialize Google Drive connector
        
        Args:
            service_account_path: Path to service account JSON file
            security_level: Security level for service account selection
        """
        # Map security levels to environment variable names
        service_account_mapping = {
            "public": "DRIVE_API_KEY_PUBLIC_ACCESS",
            "client": "DRIVE_API_KEY_CLIENT_ACCESS",
            "partner": "DRIVE_API_KEY_PARTNER_ACCESS",
            "employee": "DRIVE_API_KEY_EMPLOYEE_ACCESS",
            "management": "DRIVE_API_KEY_MANAGEMENT_ACCESS"
        }
        
        env_var = service_account_mapping.get(security_level, "DRIVE_API_KEY_EMPLOYEE_ACCESS")
        service_account_file = os.getenv(env_var) or os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH")
        
        # If we got a filename from env, assume it's in the project root
        if service_account_file and not os.path.isabs(service_account_file):
            service_account_file = os.path.join(os.getcwd(), service_account_file)
        
        self.service_account_path = service_account_path or service_account_file
        
        if not self.service_account_path:
            raise ValueError(f"Google Drive service account path not found for security level: {security_level}")
        
        if not os.path.exists(self.service_account_path):
            raise ValueError(f"Service account file not found: {self.service_account_path}")
        
        # Initialize credentials and service
        self.credentials = service_account.Credentials.from_service_account_file(
            self.service_account_path,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.service = build('drive', 'v3', credentials=self.credentials)
    
    async def get_documents(
        self,
        folder_ids: Optional[List[str]] = None,
        file_ids: Optional[List[str]] = None,
        modified_since: Optional[datetime] = None,
        mime_types: Optional[List[str]] = None
    ) -> List[GoogleDriveDocument]:
        """Fetch documents from Google Drive
        
        Args:
            folder_ids: List of folder IDs to fetch documents from
            file_ids: Specific file IDs to fetch
            modified_since: Only fetch files modified after this time
            mime_types: Filter by MIME types
            
        Returns:
            List of GoogleDriveDocument objects
        """
        documents = []
        
        # Fetch from folders
        if folder_ids:
            for folder_id in folder_ids:
                folder_docs = await self._fetch_folder_documents(
                    folder_id, modified_since, mime_types
                )
                documents.extend(folder_docs)
        
        # Fetch specific files
        if file_ids:
            for file_id in file_ids:
                doc = await self._fetch_file(file_id)
                if doc and (not modified_since or doc.modified_time > modified_since):
                    documents.append(doc)
        
        # If no specific sources, fetch from root and shared files
        if not folder_ids and not file_ids:
            # Fetch from root
            root_docs = await self._fetch_folder_documents(
                'root', modified_since, mime_types
            )
            documents.extend(root_docs)
            
            # Also fetch shared files
            shared_docs = await self._fetch_shared_documents(
                modified_since, mime_types
            )
            documents.extend(shared_docs)
        
        return documents
    
    async def _fetch_folder_documents(
        self,
        folder_id: str,
        modified_since: Optional[datetime] = None,
        mime_types: Optional[List[str]] = None
    ) -> List[GoogleDriveDocument]:
        """Fetch documents from a folder
        
        Args:
            folder_id: Folder ID ('root' for root folder)
            modified_since: Only fetch files modified after this time
            mime_types: Filter by MIME types
            
        Returns:
            List of GoogleDriveDocument objects
        """
        documents = []
        page_token = None
        
        # Build query
        query_parts = [f"'{folder_id}' in parents", "trashed = false"]
        
        if modified_since:
            query_parts.append(f"modifiedTime > '{modified_since.isoformat()}Z'")
        
        if mime_types:
            mime_conditions = [f"mimeType = '{mt}'" for mt in mime_types]
            query_parts.append(f"({' or '.join(mime_conditions)})")
        else:
            # Default to supported types
            mime_conditions = [f"mimeType = '{mt}'" for mt in self.SUPPORTED_MIME_TYPES.keys()]
            query_parts.append(f"({' or '.join(mime_conditions)})")
        
        query = " and ".join(query_parts)
        
        while True:
            try:
                # Use run_in_executor to make synchronous call async with retry
                loop = asyncio.get_event_loop()
                
                def list_files():
                    return self.service.files().list(
                        q=query,
                        pageSize=100,
                        fields="nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, webViewLink, size, parents)",
                        pageToken=page_token
                    ).execute()
                
                # Wrap with retry logic
                response = await loop.run_in_executor(
                    None,
                    lambda: retry_sync(list_files, max_attempts=3, initial_wait=2)
                )
                
                files = response.get('files', [])
                
                for file_data in files:
                    doc = await self._parse_file(file_data)
                    if doc:
                        # Fetch content
                        await self._fetch_file_content(doc)
                        documents.append(doc)
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                    
            except Exception as e:
                print(f"Error fetching folder {folder_id}: {e}")
                break
        
        return documents
    
    async def _fetch_file(self, file_id: str) -> Optional[GoogleDriveDocument]:
        """Fetch a specific file
        
        Args:
            file_id: File ID
            
        Returns:
            GoogleDriveDocument object or None
        """
        try:
            loop = asyncio.get_event_loop()
            file_data = await loop.run_in_executor(
                None,
                lambda: self.service.files().get(
                    fileId=file_id,
                    fields="id, name, mimeType, modifiedTime, createdTime, webViewLink, size, parents"
                ).execute()
            )
            
            doc = await self._parse_file(file_data)
            if doc:
                await self._fetch_file_content(doc)
            return doc
            
        except Exception as e:
            print(f"Error fetching file {file_id}: {e}")
            return None
    
    async def _fetch_shared_documents(
        self,
        modified_since: Optional[datetime] = None,
        mime_types: Optional[List[str]] = None
    ) -> List[GoogleDriveDocument]:
        """Fetch documents shared with the service account
        
        Args:
            modified_since: Only fetch files modified after this time
            mime_types: Filter by MIME types
            
        Returns:
            List of GoogleDriveDocument objects
        """
        documents = []
        page_token = None
        
        # Build query for shared files
        query_parts = ["sharedWithMe", "trashed = false"]
        
        if modified_since:
            query_parts.append(f"modifiedTime > '{modified_since.isoformat()}Z'")
        
        if mime_types:
            mime_conditions = [f"mimeType = '{mt}'" for mt in mime_types]
            query_parts.append(f"({' or '.join(mime_conditions)})")
        else:
            # Default to supported types
            mime_conditions = [f"mimeType = '{mt}'" for mt in self.SUPPORTED_MIME_TYPES.keys()]
            query_parts.append(f"({' or '.join(mime_conditions)})")
        
        query = " and ".join(query_parts)
        
        while True:
            try:
                # Use run_in_executor to make synchronous call async with retry
                loop = asyncio.get_event_loop()
                
                def list_files():
                    return self.service.files().list(
                        q=query,
                        pageSize=100,
                        fields="nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, webViewLink, size, parents)",
                        pageToken=page_token
                    ).execute()
                
                # Wrap with retry logic
                response = await loop.run_in_executor(
                    None,
                    lambda: retry_sync(list_files, max_attempts=3, initial_wait=2)
                )
                
                files = response.get('files', [])
                
                for file_data in files:
                    doc = await self._parse_file(file_data)
                    if doc:
                        # Fetch content
                        await self._fetch_file_content(doc)
                        documents.append(doc)
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                    
            except Exception as e:
                print(f"Error fetching shared documents: {e}")
                break
        
        return documents
    
    async def _parse_file(self, file_data: Dict[str, Any]) -> Optional[GoogleDriveDocument]:
        """Parse file data from Drive API response
        
        Args:
            file_data: File data from Drive API
            
        Returns:
            GoogleDriveDocument object or None
        """
        try:
            # Parse timestamps
            modified = datetime.fromisoformat(
                file_data['modifiedTime'].replace('Z', '+00:00')
            )
            created = datetime.fromisoformat(
                file_data['createdTime'].replace('Z', '+00:00')
            )
            
            return GoogleDriveDocument(
                id=file_data['id'],
                name=file_data['name'],
                mime_type=file_data['mimeType'],
                modified_time=modified,
                created_time=created,
                web_view_link=file_data.get('webViewLink', ''),
                size=int(file_data.get('size', 0)),
                parent_ids=file_data.get('parents', [])
            )
            
        except Exception as e:
            print(f"Error parsing file data: {e}")
            return None
    
    async def _fetch_file_content(self, document: GoogleDriveDocument) -> None:
        """Fetch file content
        
        Args:
            document: GoogleDriveDocument object to populate with content
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Handle Google Docs (need export)
            if document.mime_type in self.GOOGLE_DOCS_EXPORT_FORMATS:
                export_mime = self.GOOGLE_DOCS_EXPORT_FORMATS[document.mime_type]
                request = self.service.files().export_media(
                    fileId=document.id,
                    mimeType=export_mime
                )
            else:
                # Regular files
                request = self.service.files().get_media(fileId=document.id)
            
            # Download file
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while not done:
                status, done = await loop.run_in_executor(
                    None,
                    downloader.next_chunk
                )
            
            document.content = file_content.getvalue()
            
            # Try to decode text content
            if document.mime_type in ['text/plain', 'text/markdown', 'text/csv']:
                try:
                    document.content_text = document.content.decode('utf-8')
                except UnicodeDecodeError:
                    pass
            
        except Exception as e:
            print(f"Error fetching content for {document.name}: {e}")
    
    def get_content_hash(self, content: bytes) -> str:
        """Generate hash of content for change detection
        
        Args:
            content: Content bytes to hash
            
        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(content).hexdigest()
    
    async def detect_changes(
        self,
        documents: List[GoogleDriveDocument],
        stored_hashes: Dict[str, str]
    ) -> List[GoogleDriveDocument]:
        """Detect which documents have changed
        
        Args:
            documents: List of fetched documents
            stored_hashes: Dictionary of file_id -> content_hash
            
        Returns:
            List of changed documents
        """
        changed_documents = []
        
        for doc in documents:
            if doc.content:
                current_hash = self.get_content_hash(doc.content)
                stored_hash = stored_hashes.get(doc.id)
                
                if stored_hash != current_hash:
                    changed_documents.append(doc)
        
        return changed_documents
    
    def should_send_to_llamaparse(self, document: GoogleDriveDocument) -> bool:
        """Determine if document should be sent to LlamaParse
        
        Args:
            document: GoogleDriveDocument
            
        Returns:
            True if should send to LlamaParse
        """
        # PDFs and Office documents should go to LlamaParse
        llamaparse_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint'
        ]
        
        return document.mime_type in llamaparse_types
    
    def get_access_level(self, security_level: str) -> int:
        """Get numeric access level from security level string
        
        Args:
            security_level: Security level string
            
        Returns:
            Numeric access level (1-5)
        """
        level_map = {
            "public": 1,
            "client": 2,
            "partner": 3,
            "employee": 4,
            "management": 5
        }
        return level_map.get(security_level, 4)
    
    async def scan_drive(self, folder_id: Optional[str] = None, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """Scan Google Drive and return document data for database storage
        
        Args:
            folder_id: Optional folder ID to scan
            file_types: List of file extensions to scan
            
        Returns:
            List of document dictionaries ready for database storage
        """
        # Fetch documents from drive
        documents = await self.get_documents(folder_ids=[folder_id] if folder_id else None)
        
        # Filter by file types if specified
        if file_types:
            filtered_docs = []
            for doc in documents:
                # Check if file extension matches
                for ext in file_types:
                    if doc.name.lower().endswith(ext):
                        filtered_docs.append(doc)
                        break
            documents = filtered_docs
        
        # Convert to document format for database
        result = []
        for doc in documents:
            result.append({
                "source_id": doc.id,
                "title": doc.name,
                "content": doc.content_text if doc.content_text else "",
                "modified_time": doc.modified_time,
                "metadata": {
                    "mime_type": doc.mime_type,
                    "size": doc.size,
                    "web_view_link": doc.web_view_link,
                    "created_time": doc.created_time.isoformat(),
                    "parent_ids": doc.parent_ids,
                    "should_send_to_llamaparse": self.should_send_to_llamaparse(doc)
                }
            })
        
        return result


async def test_google_drive_connector():
    """Test the Google Drive connector"""
    print("\n" + "="*60)
    print("GOOGLE DRIVE CONNECTOR TEST")
    print("="*60)
    
    try:
        # Initialize connector
        connector = GoogleDriveConnector()
        print("[OK] Google Drive connector initialized")
        
        # Fetch recent documents
        print("\nFetching documents from Drive...")
        documents = await connector.get_documents()
        
        print(f"[OK] Found {len(documents)} documents")
        
        # Display first few documents
        for doc in documents[:3]:
            print(f"\nDocument: {doc.name}")
            print(f"  ID: {doc.id}")
            print(f"  Type: {doc.mime_type}")
            print(f"  Size: {doc.size} bytes")
            print(f"  Modified: {doc.modified_time}")
            print(f"  Send to LlamaParse: {connector.should_send_to_llamaparse(doc)}")
            
            if doc.content_text:
                print(f"  Content preview: {doc.content_text[:200]}...")
            elif doc.content:
                print(f"  Content size: {len(doc.content)} bytes")
        
        # Test change detection
        if documents:
            print("\n" + "-"*40)
            print("Testing change detection...")
            
            # Simulate stored hashes
            stored_hashes = {}
            for doc in documents[:2]:
                if doc.content:
                    stored_hashes[doc.id] = connector.get_content_hash(doc.content)
            
            # Simulate a change
            if documents and documents[0].content:
                stored_hashes[documents[0].id] = "different_hash"
            
            changed = await connector.detect_changes(documents[:2], stored_hashes)
            print(f"[OK] Detected {len(changed)} changed documents")
            
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Please set GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH in your .env file")
        print("And ensure the service account JSON file exists")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_google_drive_connector())