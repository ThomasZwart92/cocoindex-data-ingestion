"""
Metadata Extraction Service
Handles extraction of structured metadata from documents using LLM and pattern matching
"""
import re
import logging
from typing import Dict, List, Any
from datetime import datetime

from app.models.metadata_taxonomy import DocumentCategory, TagTaxonomy
from app.services.metadata_extraction_service import MetadataExtractionService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)


def extract_product_models(content: str) -> List[str]:
    """Extract product model numbers and identifiers from content"""
    models = []
    
    # Pattern for NC series (non-conformity identifiers)
    nc_pattern = r'\bNC\d{4}\b'
    models.extend(re.findall(nc_pattern, content))
    
    # Pattern for PC series
    pc_pattern = r'\bPC\d{4}\b'
    models.extend(re.findall(pc_pattern, content))
    
    # Pattern for SM series
    sm_pattern = r'\bSM\d{3}\b'
    models.extend(re.findall(sm_pattern, content))
    
    # Deduplicate
    return list(set(models))


def extract_components(content: str) -> List[str]:
    """Extract technical component mentions from content"""
    content_lower = content.lower()
    found_components = []
    
    for component in TagTaxonomy.COMPONENTS:
        # Check for exact word match (with word boundaries)
        pattern = r'\b' + re.escape(component.lower()) + r'\b'
        if re.search(pattern, content_lower):
            found_components.append(component)
    
    return found_components


def extract_issues(content: str) -> List[str]:
    """Extract issue and problem types from content"""
    content_lower = content.lower()
    found_issues = []
    
    for issue in TagTaxonomy.ISSUES:
        # Convert hyphenated terms to also match with spaces
        issue_variations = [
            issue.lower(),
            issue.lower().replace("-", " "),
            issue.lower().replace("-", "")
        ]
        
        for variation in issue_variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, content_lower):
                found_issues.append(issue)
                break
    
    return found_issues


def combine_and_deduplicate_tags(
    llm_tags: List[str],
    product_tags: List[str],
    component_tags: List[str],
    issue_tags: List[str]
) -> List[str]:
    """Combine tags from multiple sources and deduplicate"""
    all_tags = []
    
    # Add all tags
    all_tags.extend(llm_tags if llm_tags else [])
    all_tags.extend(product_tags)
    all_tags.extend(component_tags)
    all_tags.extend(issue_tags)
    
    # Normalize and deduplicate
    normalized_tags = []
    seen = set()
    
    for tag in all_tags:
        # Normalize tag (lowercase for comparison)
        normalized = tag.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            # Keep original casing for product models
            if re.match(r'^[A-Z]+\d+', tag):
                normalized_tags.append(tag)
            else:
                normalized_tags.append(normalized)
    
    # Sort tags by category (products first, then alphabetical)
    def tag_sort_key(tag):
        if re.match(r'^[A-Z]+\d+', tag):  # Product model
            return (0, tag)
        elif tag in TagTaxonomy.COMPONENTS:
            return (1, tag)
        elif tag in TagTaxonomy.ISSUES:
            return (2, tag)
        else:
            return (3, tag)
    
    return sorted(normalized_tags, key=tag_sort_key)[:15]  # Limit to 15 tags


async def extract_metadata_for_document(document_id: str) -> Dict[str, Any]:
    """
    Extract metadata for a specific document using LLM and pattern matching.
    
    This function:
    1. Fetches the document from Supabase
    2. Extracts metadata using the MetadataExtractionService (LLM + regex)
    3. Updates the document's metadata in Supabase
    
    Args:
        document_id: UUID of the document to process
        
    Returns:
        Dict containing success status and extracted metadata or error information
    """
    logger.info(f"Starting metadata extraction for document {document_id}")
    
    try:
        # Get document from Supabase
        supabase_service = SupabaseService()
        doc_response = supabase_service.client.table("documents").select("*").eq(
            "id", document_id
        ).single().execute()
        
        if not doc_response.data:
            raise ValueError(f"Document {document_id} not found")
        
        document = doc_response.data
        
        # Check if document has content
        if not document.get("content"):
            logger.warning(f"Document {document_id} has no content for extraction")
            return {
                "error": "No content available for metadata extraction",
                "document_id": document_id
            }
        
        # Extract using the service (handles LLM + regex enrichment and audit fields)
        service = MetadataExtractionService()
        extracted = service.extract(document["content"])  # handles excerpting internally
        
        # Prepare metadata update with all fields including audit information
        metadata_update = {
            "category": extracted.get("category"),
            "tags": extracted.get("tags", []),
            "author": extracted.get("author"),
            "department": extracted.get("department"),
            "version": extracted.get("version"),
            "description": extracted.get("description"),
            "ai_extracted": True,
            "extraction_timestamp": extracted.get("extraction_timestamp"),
            "confidence_scores": extracted.get("confidence_scores", {}),
            "extraction_model": extracted.get("extraction_model"),
            "prompt_version": extracted.get("prompt_version"),
        }
        
        # Merge with existing metadata and update document
        existing_metadata = document.get("metadata", {}) or {}
        new_metadata = {**existing_metadata, **metadata_update}
        
        supabase_service.client.table("documents").update({
            "metadata": new_metadata,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", document_id).execute()
        
        logger.info(f"Successfully extracted metadata for document {document_id}")
        logger.debug(f"Extracted tags: {metadata_update.get('tags', [])}")
        logger.debug(f"Category: {metadata_update.get('category')}, "
                    f"Department: {metadata_update.get('department')}")
        
        return {
            "success": True,
            "document_id": document_id,
            "metadata": metadata_update
        }
            
    except Exception as e:
        logger.error(f"Error extracting metadata for document {document_id}: {e}")
        return {
            "error": str(e),
            "document_id": document_id
        }