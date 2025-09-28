"""Document parser using LlamaParse"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from llama_index.core.bridge import pydantic as _li_pydantic

# Expose legacy validator alias for llama_parse compatibility
if not hasattr(_li_pydantic, 'validator'):
    _li_pydantic.validator = _li_pydantic.field_validator

from llama_parse import LlamaParse
from app.config import settings
from app.models.document import ParseTier
from app.utils.path_validator import PathValidator, PathSecurityError

logger = logging.getLogger(__name__)

class DocumentParser:
    """Parse documents using LlamaParse"""
    
    def __init__(self):
        self.api_key = settings.llamaparse_api_key
        self.base_url = settings.llamaparse_base_url
    
    async def parse_document(
        self,
        file_path: str,
        tier: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Parse a document asynchronously (wrapper for compatibility)
        
        Args:
            file_path: Path to document
            tier: Parsing tier (balanced, agentic, agentic_plus)
            
        Returns:
            Dict with success, content, metadata, etc.
        """
        # Convert tier string to enum
        tier_map = {
            "balanced": ParseTier.BALANCED,
            "agentic": ParseTier.AGENTIC,
            "agentic_plus": ParseTier.AGENTIC_PLUS
        }
        parse_tier = tier_map.get(tier, ParseTier.BALANCED)
        
        # For simple files, just read the content
        file_path_obj = Path(file_path)
        
        if file_path_obj.suffix in ['.txt', '.md']:
            # Simple text files don't need LlamaParse
            try:
                content = file_path_obj.read_text(encoding='utf-8')
                return {
                    "success": True,
                    "content": content,
                    "metadata": {
                        "tier": tier,
                        "parser": "simple",
                        "confidence": 1.0
                    },
                    "images": []
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "content": "",
                    "metadata": {},
                    "images": []
                }
        
        # For complex files, use the existing parse method
        try:
            result = self.parse(
                document_path=str(file_path),
                document_name=file_path_obj.name,
                parse_tier=parse_tier
            )
            
            return {
                "success": True,
                "content": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "images": result.get("images", [])
            }
        except Exception as e:
            logger.error(f"Parse failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": "",
                "metadata": {},
                "images": []
            }
    
    def parse(
        self, 
        document_path: str,
        document_name: str,
        parse_tier: ParseTier = ParseTier.BALANCED
    ) -> Dict[str, Any]:
        """
        Parse a document using LlamaParse
        
        Returns:
            Dict containing:
            - text: Parsed text content
            - metadata: Document metadata
            - images: List of extracted images
            - confidence: Parsing confidence score
        """
        logger.info(f"Parsing document: {document_name} with tier: {parse_tier.value}")
        
        try:
            # Configure parser based on tier
            parser_config = self._get_parser_config(parse_tier)
            
            # Initialize parser
            parser = LlamaParse(
                api_key=self.api_key,
                # Let it auto-detect EU endpoint from API key
                result_type="markdown",
                verbose=True,
                **parser_config
            )
            
            # Validate and parse the document
            try:
                # Validate path (allows URLs)
                validated_path = PathValidator.validate_path(document_path, allow_urls=True)
                
                if validated_path.startswith("http"):
                    # URL document: download first, then upload via Llama Cloud
                    import httpx, tempfile
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.get(validated_path)
                        resp.raise_for_status()
                        # Write to a temporary PDF file
                        suffix = ".pdf" if validated_path.lower().endswith(".pdf") else ".bin"
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                            tmp.write(resp.content)
                            tmp.flush()
                            documents = parser.load_data(tmp.name)
                else:
                    # Local file
                    file_path = Path(validated_path)
                    if not file_path.exists():
                        # Do not fall back to test content; surface a clear error instead
                        logger.error(f"File not found: {validated_path}")
                        raise FileNotFoundError(f"File not found: {validated_path}")
                    
                    documents = parser.load_data(str(file_path))
            except PathSecurityError as e:
                logger.error(f"Path validation failed: {e}")
                raise ValueError(f"Invalid document path: {e}")
            
            # Extract content
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Combine all document texts
            full_text = "\n\n".join([doc.text for doc in documents])
            
            # Extract metadata
            metadata = {}
            if hasattr(documents[0], 'metadata'):
                metadata = documents[0].metadata
            
            # Extract images (if any)
            images = self._extract_images(documents)
            
            # Calculate confidence based on content
            confidence = self._calculate_confidence(full_text, parse_tier)
            
            return {
                "text": full_text,
                "metadata": metadata,
                "images": images,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to parse document '{document_name}': {str(e)}")
            # Do not return placeholder content; propagate error so callers can handle it
            raise
    
    def _get_parser_config(self, parse_tier: ParseTier) -> Dict[str, Any]:
        """Get parser configuration based on tier"""
        configs = {
            ParseTier.BALANCED: {
                "parsing_instruction": "Extract all text content maintaining structure",
                "skip_diagonal_text": True,
                "invalidate_cache": False
            },
            ParseTier.AGENTIC: {
                "parsing_instruction": "Extract and structure all content with high accuracy",
                "skip_diagonal_text": False,
                "invalidate_cache": False,
                "premium_mode": True
            },
            ParseTier.AGENTIC_PLUS: {
                "parsing_instruction": "Extract all content with maximum accuracy and detail",
                "skip_diagonal_text": False,
                "invalidate_cache": True,
                "premium_mode": True,
                "extract_tables": True
            }
        }
        return configs.get(parse_tier, configs[ParseTier.BALANCED])
    
    def _extract_images(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Extract images from parsed documents"""
        images = []
        for i, doc in enumerate(documents):
            if hasattr(doc, 'images'):
                for j, img in enumerate(doc.images or []):
                    images.append({
                        "index": len(images),
                        "page": i + 1,
                        "data": img,
                        "type": "extracted"
                    })
        return images
    
    def _calculate_confidence(self, text: str, parse_tier: ParseTier) -> float:
        """Calculate parsing confidence score"""
        base_confidence = {
            ParseTier.BALANCED: 0.7,
            ParseTier.AGENTIC: 0.85,
            ParseTier.AGENTIC_PLUS: 0.95
        }
        
        confidence = base_confidence.get(parse_tier, 0.7)
        
        # Adjust based on content quality indicators
        if len(text) > 1000:
            confidence += 0.05
        if len(text) > 5000:
            confidence += 0.05
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _get_test_content(self, document_name: str) -> Dict[str, Any]:
        """Get test content for development"""
        test_text = f"""
# {document_name}

## Executive Summary
This document provides a comprehensive overview of modern data processing architectures
and best practices for building scalable document ingestion systems.

## Introduction
Document processing has evolved significantly with the advent of large language models
and advanced parsing techniques. This system leverages cutting-edge technology to extract,
process, and analyze documents at scale.

## Key Components

### 1. Document Parsing
- Advanced OCR capabilities
- Multi-format support (PDF, DOCX, HTML, etc.)
- Layout preservation
- Table extraction

### 2. Text Processing
- Intelligent chunking strategies
- Semantic segmentation
- Context preservation
- Hierarchical structuring

### 3. Entity Extraction
- Named entity recognition
- Relationship mapping
- Confidence scoring
- Multi-model validation

### 4. Knowledge Graph Construction
- Entity resolution
- Relationship inference
- Graph optimization
- Query capabilities

## Technical Architecture

The system is built on a microservices architecture with the following components:

1. **Ingestion Service**: Handles document upload and initial processing
2. **Parser Service**: Converts documents to structured text
3. **Chunking Service**: Segments documents into processable units
4. **Embedding Service**: Generates vector representations
5. **Entity Service**: Extracts and validates entities
6. **Graph Service**: Builds and maintains knowledge graphs

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Parse Time | < 5s | 3.2s |
| Chunk Accuracy | > 95% | 97% |
| Entity F1 Score | > 0.85 | 0.89 |
| Graph Completeness | > 90% | 92% |

## Conclusion

This document processing system represents state-of-the-art capabilities in
automated document analysis and knowledge extraction.
"""
        
        return {
            "text": test_text.strip(),
            "metadata": {
                "title": document_name,
                "page_count": 3,
                "word_count": len(test_text.split()),
                "source": "test"
            },
            "images": [],
            "confidence": 0.95
        }
