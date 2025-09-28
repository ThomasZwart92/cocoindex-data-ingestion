"""
LlamaParse Service

Thin wrapper around Llama Cloud (LlamaIndex) parsing API to convert PDFs/DOCX
into structured outputs. Provides helpers to obtain markdown and structured
artifacts (tables/images metadata) suitable for downstream chunking.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import base64
import urllib.parse

import aiohttp
import asyncio
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LlamaParseService:
    """Client for Llama Cloud parsing API.

    Notes:
    - Requires `settings.llamaparse_api_key`.
    - `settings.llamaparse_base_url` defaults to Llama Cloud US endpoint; `.env`
      may override to EU as needed.
    - This client uses aiohttp for async calls; callers should be async or run
      via an event loop.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.llamaparse_api_key
        self.base_url = base_url or settings.llamaparse_base_url
        if not self.api_key:
            logger.warning("LlamaParse API key not configured; will fall back to passthrough parsing")

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy JSON API call. Some deployments expose /v1/parse; keep for compatibility."""
        url = self._build_url(path)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(payload), timeout=120) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"LlamaParse error {resp.status}: {text[:200]}")
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}

    def _build_url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        if path.startswith("/"):
            return f"{base}{path}"
        return f"{base}/{path}"

    def _parsing_endpoint(self, suffix: str, base_override: Optional[str] = None) -> str:
        """Build URL for /api/parsing endpoints regardless of base form.

        Accepts base either as https://... (no /api/parsing) or already pointing
        to .../api/parsing. This ensures compatibility with both forms.
        """
        base = (base_override or self.base_url).rstrip("/")
        # Support multiple base forms:
        # - https://.../api/v1          -> /parsing/<suffix>
        # - https://.../api             -> /v1/parsing/<suffix>
        # - https://.../api/parsing     -> /<suffix>
        # - https://...                 -> /api/v1/parsing/<suffix>
        if base.endswith("/api/parsing"):
            return f"{base}/{suffix.lstrip('/')}"
        if base.endswith("/api/v1"):
            return f"{base}/parsing/{suffix.lstrip('/')}"
        if base.endswith("/api"):
            return f"{base}/v1/parsing/{suffix.lstrip('/')}"
        return f"{base}/api/v1/parsing/{suffix.lstrip('/')}"

    async def _upload_and_poll(self, content: bytes, filename: str, result_type: str = "markdown", timeout: int = 300, include_structured: bool = True, parsing_mode: str = "balanced") -> Dict[str, Any]:
        """Use official upload + poll flow exposed by Llama Cloud.
        
        Args:
            content: File content bytes
            filename: Name of the file
            result_type: Type of result to fetch (markdown, text, json, etc.)
            timeout: Maximum time to wait for parsing completion
            include_structured: Whether to fetch structured data (tables/images)
            parsing_mode: Parsing preset (cost_effective, balanced, agentic, agentic_plus)
        """
        upload_url = self._parsing_endpoint("upload")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Prepare form data with parsing parameters
        data = {
            "parsing_mode": parsing_mode,
            "job_timeout_in_seconds": str(timeout),
        }
        files = {"file": (filename, content, self._infer_mime(filename))}
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upload_url, headers=headers, files=files, data=data)
            if resp.status_code >= 400:
                # Fallback to default US base if a custom base yields 404
                if resp.status_code == 404 and self.base_url != "https://api.cloud.llamaindex.ai/api/v1":
                    fb_base = "https://api.cloud.llamaindex.ai/api/v1"
                    fb_upload = self._parsing_endpoint("upload", base_override=fb_base)
                    resp = await client.post(fb_upload, headers=headers, files=files, data=data)
                    if resp.status_code >= 400:
                        raise RuntimeError(f"LlamaParse upload failed {resp.status_code}: {resp.text[:200]}")
                    job_id = resp.json().get("id")
                    if not job_id:
                        raise RuntimeError("LlamaParse upload did not return job id (fallback)")
                    result_url = self._parsing_endpoint(f"job/{job_id}/result/{result_type}", base_override=fb_base)
                    # Poll until ready
                    start = __import__("time").time()
                    while True:
                        # Check job status with fallback base
                        job_status, job_data = await self._check_job_status(client, headers, job_id, fb_base)
                        
                        if job_status == "SUCCESS":
                            r = await client.get(result_url, headers=headers)
                            if r.status_code == 200:
                                data = r.json()
                                markdown = data.get("markdown") or data.get("text") or data.get("md") or ""
                                tables, images = [], []
                                if include_structured:
                                    tables, images = await self._fetch_structured_data(client, headers, job_id, fb_base)
                                
                                pages = data.get("pages", [])
                                logger.info(f"Successfully parsed document {job_id} (via fallback): {len(markdown)} chars, {len(tables)} tables, {len(images)} images")
                                return {"markdown": markdown, "pages": pages, "tables": tables, "images": images, "raw": data}
                            else:
                                raise RuntimeError(f"LlamaParse result error {r.status_code}: {r.text[:200]}")
                        
                        elif job_status in ["ERROR", "FAILED", "EXPIRED"]:
                            error_msg = job_data.get("error", job_data.get("message", "Unknown error"))
                            raise RuntimeError(f"LlamaParse job failed with status: {job_status}. Error: {error_msg}")
                        
                        elif job_status in ["PENDING", "PROCESSING", "UNKNOWN"]:
                            if __import__("time").time() - start > timeout:
                                raise RuntimeError(f"LlamaParse result polling timed out after {timeout}s (fallback)")
                            await asyncio.sleep(2.0)
                        
                        else:
                            # Try fetching result anyway
                            r = await client.get(result_url, headers=headers)
                            if r.status_code == 200:
                                data = r.json()
                                markdown = data.get("markdown") or data.get("text") or data.get("md") or ""
                                tables, images = [], []
                                if include_structured:
                                    tables, images = await self._fetch_structured_data(client, headers, job_id, fb_base)
                                return {"markdown": markdown, "pages": [], "tables": tables, "images": images, "raw": data}
                            elif r.status_code in (404, 202):
                                if __import__("time").time() - start > timeout:
                                    raise RuntimeError(f"LlamaParse result polling timed out after {timeout}s (fallback)")
                                await asyncio.sleep(2.0)
                            else:
                                raise RuntimeError(f"LlamaParse result error {r.status_code}: {r.text[:200]}")
                else:
                    raise RuntimeError(f"LlamaParse upload failed {resp.status_code}: {resp.text[:200]}")
            job_id = resp.json().get("id")
            if not job_id:
                raise RuntimeError("LlamaParse upload did not return job id")
            result_url = self._parsing_endpoint(f"job/{job_id}/result/{result_type}")
            # Poll until ready
            start = __import__("time").time()
            while True:
                # First check job status
                job_status, job_data = await self._check_job_status(client, headers, job_id)
                
                if job_status == "SUCCESS":
                    # Job is complete, fetch results
                    r = await client.get(result_url, headers=headers)
                    if r.status_code == 200:
                        data = r.json()
                        # Normalize similar to JSON API
                        markdown = data.get("markdown") or data.get("text") or data.get("md") or ""
                        tables, images = [], []
                        if include_structured:
                            tables, images = await self._fetch_structured_data(client, headers, job_id)
                        
                        # Extract pages if available
                        pages = data.get("pages", [])
                        
                        logger.info(f"Successfully parsed document {job_id}: {len(markdown)} chars, {len(tables)} tables, {len(images)} images")
                        return {"markdown": markdown, "pages": pages, "tables": tables, "images": images, "raw": data}
                    else:
                        raise RuntimeError(f"LlamaParse result error {r.status_code}: {r.text[:200]}")
                
                elif job_status in ["ERROR", "FAILED", "EXPIRED"]:
                    error_msg = job_data.get("error", job_data.get("message", "Unknown error"))
                    raise RuntimeError(f"LlamaParse job failed with status: {job_status}. Error: {error_msg}")
                
                elif job_status in ["PENDING", "PROCESSING", "UNKNOWN"]:
                    # Still processing, continue polling
                    if __import__("time").time() - start > timeout:
                        raise RuntimeError(f"LlamaParse result polling timed out after {timeout}s")
                    await asyncio.sleep(2.0)  # Poll every 2 seconds
                
                else:
                    # Unknown status, try fetching result anyway
                    r = await client.get(result_url, headers=headers)
                    if r.status_code == 200:
                        data = r.json()
                        markdown = data.get("markdown") or data.get("text") or data.get("md") or ""
                        tables, images = [], []
                        if include_structured:
                            tables, images = await self._fetch_structured_data(client, headers, job_id)
                        return {"markdown": markdown, "pages": [], "tables": tables, "images": images, "raw": data}
                    elif r.status_code in (404, 202):
                        # Still processing
                        if __import__("time").time() - start > timeout:
                            raise RuntimeError(f"LlamaParse result polling timed out after {timeout}s")
                        await asyncio.sleep(2.0)
                    else:
                        raise RuntimeError(f"LlamaParse result error {r.status_code}: {r.text[:200]}")

    async def _fetch_structured_data(self, client: httpx.AsyncClient, headers: dict, job_id: str, base_override: Optional[str] = None) -> tuple[list, list]:
        """Fetch structured data (tables and images) from multiple endpoints.
        
        Returns:
            Tuple of (tables, images) lists
        """
        tables = []
        images = []
        
        # Try multiple endpoints for structured data
        structured_endpoints = [
            "json",  # Standard JSON result
            "structured",  # Structured result endpoint
        ]
        
        for endpoint in structured_endpoints:
            try:
                url = self._parsing_endpoint(f"job/{job_id}/result/{endpoint}", base_override=base_override)
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    # Handle different response formats
                    if isinstance(data, dict):
                        tables = data.get("tables") or data.get("table") or tables
                        images = data.get("images") or data.get("image") or images
                    elif isinstance(data, list) and data:
                        # Some endpoints return list of pages
                        for page in data:
                            if isinstance(page, dict):
                                page_tables = page.get("tables") or page.get("table") or []
                                page_images = page.get("images") or page.get("image") or []
                                tables.extend(page_tables if isinstance(page_tables, list) else [page_tables])
                                images.extend(page_images if isinstance(page_images, list) else [page_images])
                    if tables or images:
                        break
            except Exception as e:
                logger.debug(f"Failed to fetch structured data from {endpoint}: {e}")
                continue
        
        return tables, images
    
    async def _check_job_status(self, client: httpx.AsyncClient, headers: dict, job_id: str, base_override: Optional[str] = None) -> tuple[str, dict]:
        """Check the status of a parsing job.
        
        Returns:
            Tuple of (status, full_response_data)
        """
        try:
            status_url = self._parsing_endpoint(f"job/{job_id}", base_override=base_override)
            resp = await client.get(status_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status", "UNKNOWN"), data
        except Exception as e:
            logger.debug(f"Failed to check job status: {e}")
        return "UNKNOWN", {}
    
    @staticmethod
    def _infer_mime(filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return "application/pdf"
        if lower.endswith(".docx"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if lower.endswith(".doc"):
            return "application/msword"
        if lower.endswith(".pptx"):
            return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        if lower.endswith(".xlsx"):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if lower.endswith(".txt"):
            return "text/plain"
        if lower.endswith(".html") or lower.endswith(".htm"):
            return "text/html"
        return "application/octet-stream"

    async def parse_from_url(
        self,
        file_url: str,
        tier: str = "balanced",
        include_structured: bool = True,
    ) -> Dict[str, Any]:
        """Parse a document by remote URL by downloading and uploading to Llama Cloud.
        
        Args:
            file_url: URL of the document to parse
            tier: Parsing mode/tier (cost_effective, balanced, agentic, agentic_plus)
            include_structured: Whether to fetch structured data (tables/images)
        """
        if not self.api_key:
            logger.info("No LlamaParse key; returning empty parse result")
            return {"markdown": None, "pages": [], "tables": [], "images": []}
        
        # Map tier names to parsing modes
        parsing_mode_map = {
            "cost_effective": "cost_effective",
            "balanced": "balanced",
            "agentic": "agentic",
            "agentic_plus": "agentic_plus",
            # Legacy tier names
            "simple": "cost_effective",
            "standard": "balanced",
            "premium": "agentic",
        }
        parsing_mode = parsing_mode_map.get(tier.lower(), "balanced")
        
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            r = await client.get(file_url)
            if r.status_code >= 400:
                raise RuntimeError(f"Failed to fetch source_url: {r.status_code}")
            
            # Check if we got HTML instead of binary content (redirect page)
            content_type = r.headers.get("content-type", "").lower()
            if "text/html" in content_type and len(r.content) < 1000:
                # Likely a redirect page, log warning
                logger.warning(f"Got HTML content instead of document from {file_url}, content may be a redirect page")
            
            # Guess filename from URL (use final URL after redirects)
            final_url = str(r.url) if hasattr(r, 'url') else file_url
            parsed = urllib.parse.urlparse(final_url)
            name = parsed.path.split("/")[-1] or "document.pdf"
            return await self._upload_and_poll(
                r.content, name, 
                result_type="markdown", 
                include_structured=include_structured,
                parsing_mode=parsing_mode
            )

    async def parse_from_bytes(
        self,
        content: bytes,
        filename: str = "document.pdf",
        tier: str = "balanced",
        include_structured: bool = True,
    ) -> Dict[str, Any]:
        """Parse a document by uploading bytes using Llama Cloud upload flow.
        
        Args:
            content: Document content as bytes
            filename: Name of the file (used to determine MIME type)
            tier: Parsing mode/tier (cost_effective, balanced, agentic, agentic_plus)
            include_structured: Whether to fetch structured data (tables/images)
        """
        if not self.api_key:
            logger.info("No LlamaParse key; returning empty parse result")
            return {"markdown": None, "pages": [], "tables": [], "images": []}
        
        # Map tier names to parsing modes
        parsing_mode_map = {
            "cost_effective": "cost_effective",
            "balanced": "balanced",
            "agentic": "agentic",
            "agentic_plus": "agentic_plus",
            # Legacy tier names
            "simple": "cost_effective",
            "standard": "balanced",
            "premium": "agentic",
        }
        parsing_mode = parsing_mode_map.get(tier.lower(), "balanced")
        
        logger.info(f"Uploading document bytes to Llama Cloud for parsing (mode: {parsing_mode})")
        return await self._upload_and_poll(
            content, filename, 
            result_type="markdown", 
            include_structured=include_structured,
            parsing_mode=parsing_mode
        )

    @staticmethod
    def _normalize_response(resp: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize heterogeneous API responses to a stable shape."""
        if not resp:
            return {"markdown": None, "pages": [], "tables": [], "images": []}
        markdown = resp.get("markdown") or resp.get("content") or resp.get("md")
        pages = resp.get("pages", [])
        tables = resp.get("tables", [])
        images = resp.get("images", [])
        return {
            "markdown": markdown,
            "pages": pages,
            "tables": tables,
            "images": images,
            "raw": resp,
        }
