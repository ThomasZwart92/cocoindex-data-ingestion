"""
Notion Connector - Fetches pages from Notion workspace
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import hashlib
import json

import aiohttp
from notion_client import AsyncClient
from pydantic import BaseModel

from app.utils.retry import retry_on_failure


class NotionPage(BaseModel):
    """Represents a Notion page"""
    id: str
    title: str
    content: str
    last_edited_time: datetime
    created_time: datetime
    url: str
    parent_id: Optional[str] = None
    properties: Dict[str, Any] = {}


class NotionConnector:
    """Connector for fetching pages from Notion"""
    
    def __init__(self, api_key: Optional[str] = None, security_level: str = "employee"):
        """Initialize Notion connector
        
        Args:
            api_key: Notion integration API key
            security_level: Security level for API key selection
        """
        # Map security levels to environment variable names
        api_key_mapping = {
            "public": "NOTION_API_KEY_PUBLIC_ACCESS",
            "client": "NOTION_API_KEY_CLIENT_ACCESS",
            "partner": "NOTION_API_KEY_PARTNER_ACCESS",
            "employee": "NOTION_API_KEY_EMPLOYEE_ACCESS",
            "management": "NOTION_API_KEY_MANAGEMENT_ACCESS"
        }
        
        env_var = api_key_mapping.get(security_level, "NOTION_API_KEY_EMPLOYEE_ACCESS")
        self.api_key = api_key or os.getenv(env_var) or os.getenv("NOTION_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"Notion API key not found for security level: {security_level}")
        
        self.client = AsyncClient(auth=self.api_key)
        self._last_scan_cache: Dict[str, datetime] = {}
    
    async def get_workspace_pages(
        self, 
        database_ids: Optional[List[str]] = None,
        page_ids: Optional[List[str]] = None,
        modified_since: Optional[datetime] = None
    ) -> List[NotionPage]:
        """Fetch pages from Notion workspace
        
        Args:
            database_ids: List of database IDs to fetch pages from
            page_ids: Specific page IDs to fetch
            modified_since: Only fetch pages modified after this time
            
        Returns:
            List of NotionPage objects
        """
        pages = []
        
        # Fetch from databases
        if database_ids:
            for db_id in database_ids:
                db_pages = await self._fetch_database_pages(db_id, modified_since)
                pages.extend(db_pages)
        
        # Fetch specific pages
        if page_ids:
            for page_id in page_ids:
                page = await self._fetch_page(page_id)
                if page:
                    if not modified_since:
                        pages.append(page)
                    else:
                        # Make modified_since timezone-aware if it isn't
                        if modified_since.tzinfo is None:
                            modified_since_aware = modified_since.replace(tzinfo=timezone.utc)
                        else:
                            modified_since_aware = modified_since
                        
                        if page.last_edited_time > modified_since_aware:
                            pages.append(page)
        
        # If no specific sources, search all accessible pages
        if not database_ids and not page_ids:
            pages = await self._search_all_pages(modified_since)
        
        return pages
    
    @retry_on_failure(max_attempts=3, min_wait=2, max_wait=30)
    async def _fetch_database_pages(
        self, 
        database_id: str, 
        modified_since: Optional[datetime] = None
    ) -> List[NotionPage]:
        """Fetch pages from a Notion database with retry logic
        
        Args:
            database_id: Database ID
            modified_since: Only fetch pages modified after this time
            
        Returns:
            List of NotionPage objects
        """
        pages = []
        has_more = True
        start_cursor = None
        
        filter_obj = None
        if modified_since:
            filter_obj = {
                "timestamp": "last_edited_time",
                "last_edited_time": {
                    "after": modified_since.isoformat()
                }
            }
        
        while has_more:
            try:
                response = await self.client.databases.query(
                    database_id=database_id,
                    filter=filter_obj,
                    start_cursor=start_cursor,
                    page_size=100
                )
                
                for page_data in response["results"]:
                    page = await self._parse_page(page_data)
                    if page:
                        pages.append(page)
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
                
            except Exception as e:
                print(f"Error fetching database {database_id}: {e}")
                break
        
        return pages
    
    @retry_on_failure(max_attempts=3, min_wait=2, max_wait=30)
    async def _fetch_page(self, page_id: str) -> Optional[NotionPage]:
        """Fetch a specific page with retry logic
        
        Args:
            page_id: Page ID
            
        Returns:
            NotionPage object or None
        """
        try:
            page_data = await self.client.pages.retrieve(page_id=page_id)
            return await self._parse_page(page_data)
        except Exception as e:
            print(f"Error fetching page {page_id}: {e}")
            raise  # Re-raise to trigger retry
    
    @retry_on_failure(max_attempts=3, min_wait=2, max_wait=30)
    async def _search_all_pages(
        self, 
        modified_since: Optional[datetime] = None
    ) -> List[NotionPage]:
        """Search all accessible pages in workspace with retry logic
        
        Args:
            modified_since: Only fetch pages modified after this time
            
        Returns:
            List of NotionPage objects
        """
        pages = []
        has_more = True
        start_cursor = None
        
        filter_obj = {
            "property": "object",
            "value": "page"
        }
        
        while has_more:
            try:
                response = await self.client.search(
                    filter=filter_obj,
                    start_cursor=start_cursor,
                    page_size=100
                )
                
                for page_data in response["results"]:
                    if page_data["object"] == "page":
                        page = await self._parse_page(page_data)
                        if page:
                            if not modified_since:
                                pages.append(page)
                            else:
                                # Make modified_since timezone-aware if it isn't
                                if modified_since.tzinfo is None:
                                    modified_since_aware = modified_since.replace(tzinfo=timezone.utc)
                                else:
                                    modified_since_aware = modified_since
                                
                                if page.last_edited_time > modified_since_aware:
                                    pages.append(page)
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
                
            except Exception as e:
                print(f"Error searching pages: {e}")
                break
        
        return pages
    
    async def _parse_page(self, page_data: Dict[str, Any]) -> Optional[NotionPage]:
        """Parse page data from Notion API response
        
        Args:
            page_data: Page data from Notion API
            
        Returns:
            NotionPage object or None
        """
        try:
            # Extract title
            title = self._extract_title(page_data.get("properties", {}))
            if not title:
                title = f"Untitled Page {page_data['id']}"
            
            # Extract content blocks
            content = await self._fetch_page_content(page_data["id"])
            
            # Parse timestamps
            last_edited = datetime.fromisoformat(
                page_data["last_edited_time"].replace("Z", "+00:00")
            )
            created = datetime.fromisoformat(
                page_data["created_time"].replace("Z", "+00:00")
            )
            
            # Build URL
            url = page_data.get("url", f"https://notion.so/{page_data['id'].replace('-', '')}")
            
            return NotionPage(
                id=page_data["id"],
                title=title,
                content=content,
                last_edited_time=last_edited,
                created_time=created,
                url=url,
                parent_id=page_data.get("parent", {}).get("database_id"),
                properties=page_data.get("properties", {})
            )
            
        except Exception as e:
            print(f"Error parsing page: {e}")
            return None
    
    def _extract_title(self, properties: Dict[str, Any]) -> str:
        """Extract title from page properties
        
        Args:
            properties: Page properties
            
        Returns:
            Title string
        """
        # Try common title property names
        title_props = ["title", "Title", "Name", "name", "Page"]
        
        for prop_name in title_props:
            if prop_name in properties:
                prop = properties[prop_name]
                if prop["type"] == "title" and prop.get("title"):
                    return self._get_text_from_rich_text(prop["title"])
        
        # Try first title-type property
        for prop in properties.values():
            if prop["type"] == "title" and prop.get("title"):
                return self._get_text_from_rich_text(prop["title"])
        
        return ""
    
    def _get_text_from_rich_text(self, rich_text: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion rich text
        
        Args:
            rich_text: Notion rich text array
            
        Returns:
            Plain text string
        """
        return "".join([text.get("plain_text", "") for text in rich_text])
    
    async def _fetch_page_content(self, page_id: str) -> str:
        """Fetch and convert page content to markdown
        
        Args:
            page_id: Page ID
            
        Returns:
            Markdown content
        """
        blocks = []
        has_children = True
        start_cursor = None
        
        while has_children:
            try:
                response = await self.client.blocks.children.list(
                    block_id=page_id,
                    start_cursor=start_cursor,
                    page_size=100
                )
                
                blocks.extend(response["results"])
                has_children = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
                
            except Exception as e:
                print(f"Error fetching blocks for {page_id}: {e}")
                break
        
        # Convert blocks to markdown
        markdown_lines = []
        for block in blocks:
            markdown = await self._block_to_markdown(block)
            if markdown:
                markdown_lines.append(markdown)
        
        return "\n\n".join(markdown_lines)
    
    async def _block_to_markdown(self, block: Dict[str, Any]) -> str:
        """Convert a Notion block to markdown
        
        Args:
            block: Notion block data
            
        Returns:
            Markdown string
        """
        block_type = block["type"]
        
        # Handle link_preview specially - extract the URL
        if block_type == "link_preview":
            url = block.get(block_type, {}).get("url", "")
            if url:
                return f"[Link: {url}]({url})"
            return "[Link]"
        
        # Handle tables - fetch table rows
        elif block_type == "table":
            if block.get("has_children"):
                table_content = await self._fetch_table_content(block["id"])
                return table_content
            return "[Empty table]"
        
        # Text blocks
        elif block_type == "paragraph":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            return text
        
        elif block_type in ["heading_1", "heading_2", "heading_3"]:
            level = int(block_type[-1])
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            
            # Fetch nested content under headings (Notion allows nesting under headings)
            nested_content = []
            if block.get("has_children", False):
                try:
                    # Recursively fetch children blocks
                    children_response = await self.client.blocks.children.list(
                        block_id=block["id"],
                        page_size=100
                    )
                    
                    for child_block in children_response.get("results", []):
                        child_markdown = await self._block_to_markdown(child_block)
                        if child_markdown:
                            nested_content.append(child_markdown)
                            
                except Exception as e:
                    print(f"Error fetching heading children: {e}")
            
            # Return heading and nested content
            result = f"{'#' * level} {text}"
            if nested_content:
                result += "\n\n" + "\n\n".join(nested_content)
            return result
        
        elif block_type == "bulleted_list_item":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            return f"- {text}"
        
        elif block_type == "numbered_list_item":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            return f"1. {text}"
        
        elif block_type == "to_do":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            checked = "x" if block[block_type].get("checked", False) else " "
            return f"- [{checked}] {text}"
        
        elif block_type == "toggle":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            
            # Fetch nested content inside the toggle
            nested_content = []
            if block.get("has_children", False):
                try:
                    # Recursively fetch children blocks
                    children_response = await self.client.blocks.children.list(
                        block_id=block["id"],
                        page_size=100
                    )
                    
                    for child_block in children_response.get("results", []):
                        child_markdown = await self._block_to_markdown(child_block)
                        if child_markdown:
                            # Indent nested content
                            indented = "\n".join(f"  {line}" for line in child_markdown.split("\n"))
                            nested_content.append(indented)
                            
                except Exception as e:
                    print(f"Error fetching toggle children: {e}")
            
            # Return toggle header and nested content
            result = f"▼ {text}"
            if nested_content:
                result += "\n" + "\n".join(nested_content)
            return result
        
        elif block_type == "quote":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            return f"> {text}"
        
        elif block_type == "code":
            text = self._get_text_from_rich_text(block[block_type].get("rich_text", []))
            language = block[block_type].get("language", "")
            return f"```{language}\n{text}\n```"
        
        elif block_type == "divider":
            return "---"
        
        # Nested blocks
        elif block_type == "child_page":
            title = block[block_type].get("title", "Untitled")
            return f"[{title}](notion://{block['id']})"
        
        elif block_type == "child_database":
            title = block[block_type].get("title", "Database")
            return f"[{title}](notion://{block['id']})"
        
        # Media blocks (simplified)
        elif block_type == "image":
            caption = self._get_text_from_rich_text(block[block_type].get("caption", []))
            return f"![{caption}](image)"
        
        elif block_type == "video":
            caption = self._get_text_from_rich_text(block[block_type].get("caption", []))
            return f"[Video: {caption}](video)"
        
        elif block_type == "file":
            caption = self._get_text_from_rich_text(block[block_type].get("caption", []))
            return f"[File: {caption}](file)"
        
        elif block_type == "pdf":
            caption = self._get_text_from_rich_text(block[block_type].get("caption", []))
            return f"[PDF: {caption}](pdf)"
        
        elif block_type == "bookmark":
            url = block[block_type].get("url", "")
            caption = self._get_text_from_rich_text(block[block_type].get("caption", []))
            return f"[{caption or url}]({url})"
        
        elif block_type == "embed":
            url = block[block_type].get("url", "")
            return f"[Embed]({url})"
        
        elif block_type == "table_of_contents":
            return "[Table of Contents]"
        
        elif block_type == "link_to_page":
            page_id = block[block_type].get("page_id", "")
            return f"[Link to page](notion://{page_id})"
        
        else:
            # Unknown block type
            return f"[{block_type}]"
    
    async def _fetch_table_content(self, table_id: str) -> str:
        """Fetch and format table content as markdown
        
        Args:
            table_id: Table block ID
            
        Returns:
            Markdown formatted table
        """
        try:
            # Fetch table rows
            response = await self.client.blocks.children.list(
                block_id=table_id,
                page_size=100
            )
            
            rows = response.get("results", [])
            if not rows:
                return "[Empty table]"
            
            # Convert rows to markdown table
            table_lines = []
            
            for i, row in enumerate(rows):
                if row.get("type") == "table_row":
                    cells = row.get("table_row", {}).get("cells", [])
                    # Convert each cell to text
                    cell_texts = []
                    for cell in cells:
                        cell_text = self._get_text_from_rich_text(cell) if cell else ""
                        cell_texts.append(cell_text)
                    
                    # Create markdown table row
                    table_lines.append("| " + " | ".join(cell_texts) + " |")
                    
                    # Add header separator after first row
                    if i == 0:
                        table_lines.append("|" + "|".join([" --- " for _ in cell_texts]) + "|")
            
            return "\n".join(table_lines) if table_lines else "[Empty table]"
            
        except Exception as e:
            print(f"Error fetching table content: {e}")
            return "[Table error]"
    
    def get_content_hash(self, content: str) -> str:
        """Generate hash of content for change detection
        
        Args:
            content: Content to hash
            
        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def detect_changes(
        self, 
        pages: List[NotionPage], 
        stored_hashes: Dict[str, str]
    ) -> List[NotionPage]:
        """Detect which pages have changed
        
        Args:
            pages: List of fetched pages
            stored_hashes: Dictionary of page_id -> content_hash
            
        Returns:
            List of changed pages
        """
        changed_pages = []
        
        for page in pages:
            current_hash = self.get_content_hash(page.content)
            stored_hash = stored_hashes.get(page.id)
            
            if stored_hash != current_hash:
                changed_pages.append(page)
        
        return changed_pages
    
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
    
    async def scan_workspace(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scan workspace and return document data for database storage
        
        Args:
            workspace_id: Optional workspace ID to scan
            
        Returns:
            List of document dictionaries ready for database storage
        """
        # Fetch pages from workspace
        pages = await self.get_workspace_pages()
        
        # Convert to document format for database
        documents = []
        for page in pages:
            documents.append({
                "source_id": page.id,
                "title": page.title,
                "content": page.content,
                "last_edited": page.last_edited_time,
                "metadata": {
                    "url": page.url,
                    "created_time": page.created_time.isoformat(),
                    "parent_id": page.parent_id,
                    "properties": page.properties
                }
            })
        
        return documents


async def test_notion_connector():
    """Test the Notion connector"""
    print("\n" + "="*60)
    print("NOTION CONNECTOR TEST")
    print("="*60)
    
    try:
        # Initialize connector
        connector = NotionConnector()
        print("[OK] Notion connector initialized")
        
        # Fetch recent pages
        print("\nFetching pages from workspace...")
        pages = await connector.get_workspace_pages()
        
        print(f"[OK] Found {len(pages)} pages")
        
        # Display first few pages
        for page in pages[:3]:
            print(f"\nPage: {page.title}")
            print(f"  ID: {page.id}")
            print(f"  Last edited: {page.last_edited_time}")
            print(f"  Content preview: {page.content[:200]}...")
        
        # Test change detection
        if pages:
            print("\n" + "-"*40)
            print("Testing change detection...")
            
            # Simulate stored hashes
            stored_hashes = {}
            for page in pages[:2]:
                stored_hashes[page.id] = connector.get_content_hash(page.content)
            
            # Simulate a change by modifying hash
            if pages:
                stored_hashes[pages[0].id] = "different_hash"
            
            changed = await connector.detect_changes(pages[:2], stored_hashes)
            print(f"[OK] Detected {len(changed)} changed pages")
            
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Please set NOTION_API_KEY in your .env file")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(test_notion_connector())