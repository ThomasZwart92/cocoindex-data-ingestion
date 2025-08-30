"""
Metadata Extraction Flow using CocoIndex
Extracts structured metadata from documents using LLM
"""
import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass
import cocoindex
from datetime import datetime
import openai
import os

from app.models.metadata_taxonomy import DocumentCategory, TagTaxonomy, ExtractedMetadata
from app.services.supabase_service import SupabaseService
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetadataExtractionOutput:
    """Structure for metadata extraction output"""
    category: str
    tags: List[str]
    author: str
    department: str
    version: str
    description: str
    confidence_scores: Dict[str, float]


@cocoindex.flow_def(name="MetadataExtraction")
def metadata_extraction_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """
    CocoIndex flow for extracting metadata from document content
    """
    # Add document source (single document)
    data_scope["document"] = flow_builder.add_source(
        cocoindex.sources.LocalMemory()
    )
    
    # Add collector for extracted metadata
    metadata_output = data_scope.add_collector()
    
    # Process document
    with data_scope["document"].row() as doc:
        # Extract structured metadata using LLM
        doc["extracted_metadata"] = doc["content"].transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-5-mini"
                ),
                output_type=MetadataExtractionOutput,
                instruction=create_extraction_prompt()
            )
        )
        
        # Extract product models and technical terms
        doc["product_tags"] = doc["content"].transform(
            ExtractProductModels()
        )
        
        # Extract component mentions
        doc["component_tags"] = doc["content"].transform(
            ExtractComponents()
        )
        
        # Extract issue types
        doc["issue_tags"] = doc["content"].transform(
            ExtractIssues()
        )
        
        # Combine all tags and deduplicate
        doc["all_tags"] = doc.transform(
            CombineAndDeduplicateTags(),
            extracted_tags=doc["extracted_metadata"],
            product_tags=doc["product_tags"],
            component_tags=doc["component_tags"],
            issue_tags=doc["issue_tags"]
        )
        
        # Collect final metadata
        metadata_output.collect(
            document_id=doc["id"],
            category=doc["extracted_metadata"]["category"],
            tags=doc["all_tags"],
            author=doc["extracted_metadata"]["author"],
            department=doc["extracted_metadata"]["department"],
            version=doc["extracted_metadata"]["version"],
            description=doc["extracted_metadata"]["description"],
            confidence_scores=doc["extracted_metadata"]["confidence_scores"],
            extraction_timestamp=datetime.utcnow().isoformat()
        )
    
    return metadata_output


def create_extraction_prompt() -> str:
    """Create detailed prompt for metadata extraction"""
    categories = [f"- {cat.value}: {DocumentCategory.get_display_name(cat.value)}" 
                  for cat in DocumentCategory]
    
    departments = [
        "engineering", "technical_support", "client_success", 
        "supply_chain", "logistics", "sales", "finance", 
        "marketing", "people_culture", "special_projects"
    ]
    
    return f"""Extract structured metadata from this document and return as JSON.

CATEGORIES (choose exactly one):
{chr(10).join(categories)}

DEPARTMENTS (choose one if applicable):
{', '.join(departments)}

EXTRACTION RULES:
1. Category: Select the single most appropriate category based on document purpose and structure
2. Tags: Extract 5-10 relevant tags including:
   - Product models or identifiers mentioned in the document (e.g., NC2068, PC1234)
   - Key topics and themes
   - Technical components
   - Problem types or issues
   - Actions or procedures
3. Author: Extract author name if mentioned (look for "by", "author", "written by", signatures)
4. Department: Infer the most likely department based on content
5. Version: Extract version number if present (e.g., v1.0, Rev 2, Version 3.1)
6. Description: Write a concise 1-2 sentence description of the document's purpose

Return a JSON object with this structure:
{{
    "category": "category_value",
    "tags": ["tag1", "tag2", "tag3"],
    "author": "author_name or null",
    "department": "department_name or null",
    "version": "version_string or null",
    "description": "brief description",
    "confidence_scores": {{
        "category": 0.95,
        "tags": 0.85,
        "author": 0.70,
        "department": 0.80,
        "version": 0.60,
        "description": 0.90
    }}
}}

IMPORTANT: 
- Focus on factual extraction, not interpretation
- NC#### identifiers are non-conformity reports, not product models
- Return valid JSON only"""


@cocoindex.op.function()
class ExtractProductModels:
    """Extract product model numbers and identifiers from content"""
    
    def __call__(self, content: str) -> List[str]:
        """Extract product models and identifiers using regex patterns"""
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


@cocoindex.op.function()
class ExtractComponents:
    """Extract technical component mentions"""
    
    def __call__(self, content: str) -> List[str]:
        """Extract component mentions from content"""
        content_lower = content.lower()
        found_components = []
        
        for component in TagTaxonomy.COMPONENTS:
            # Check for exact word match (with word boundaries)
            pattern = r'\b' + re.escape(component.lower()) + r'\b'
            if re.search(pattern, content_lower):
                found_components.append(component)
        
        return found_components


@cocoindex.op.function()
class ExtractIssues:
    """Extract issue and problem types"""
    
    def __call__(self, content: str) -> List[str]:
        """Extract issue types from content"""
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


@cocoindex.op.function()
class CombineAndDeduplicateTags:
    """Combine tags from multiple sources and deduplicate"""
    
    def __call__(
        self,
        extracted_tags: MetadataExtractionOutput,
        product_tags: List[str],
        component_tags: List[str],
        issue_tags: List[str]
    ) -> List[str]:
        """Combine and deduplicate tags"""
        all_tags = []
        
        # Add LLM-extracted tags
        if extracted_tags and hasattr(extracted_tags, 'tags'):
            all_tags.extend(extracted_tags.tags)
        
        # Add regex-extracted tags
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
    Extract metadata for a specific document using direct OpenAI call
    Following CocoIndex patterns but without LocalMemory source
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
        
        # Direct extraction using OpenAI
        content = document["content"][:10000]  # Limit content for extraction
        
        # Extract metadata using LLM
        prompt = create_extraction_prompt()
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Using the correct model name
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract metadata from this document:\n\n{content}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse the LLM response
        try:
            extracted_data = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            extracted_data = {}
        
        # Extract additional tags using regex patterns
        product_tags = extract_product_models(content)
        component_tags = extract_components(content)
        issue_tags = extract_issues(content)
        
        # Combine all tags
        all_tags = combine_and_deduplicate_tags(
            extracted_data.get("tags", []),
            product_tags,
            component_tags,
            issue_tags
        )
        
        # Prepare the final extracted metadata
        extracted = {
            "category": extracted_data.get("category"),
            "tags": all_tags,
            "author": extracted_data.get("author"),
            "department": extracted_data.get("department"),
            "version": extracted_data.get("version"),
            "description": extracted_data.get("description"),
            "confidence_scores": extracted_data.get("confidence_scores", {}),
            "extraction_timestamp": datetime.utcnow().isoformat()
        }
        
        # Prepare metadata update
        metadata_update = {
            "category": extracted.get("category"),
            "tags": extracted.get("tags", []),
            "author": extracted.get("author"),
            "department": extracted.get("department"),
            "version": extracted.get("version"),
            "description": extracted.get("description"),
            "ai_extracted": True,
            "extraction_timestamp": extracted.get("extraction_timestamp"),
            "confidence_scores": extracted.get("confidence_scores", {})
        }
        
        # Update document metadata in Supabase
        existing_metadata = document.get("metadata", {}) or {}
        new_metadata = {**existing_metadata, **metadata_update}
        
        supabase_service.client.table("documents").update({
            "metadata": new_metadata,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", document_id).execute()
        
        logger.info(f"Successfully extracted metadata for document {document_id}")
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