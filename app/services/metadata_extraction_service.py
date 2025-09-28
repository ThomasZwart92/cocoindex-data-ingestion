"""
Metadata Extraction Service
Provides LLM-based extraction of document metadata and helpers to persist results.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import settings
from app.models.metadata_taxonomy import DocumentCategory, TagTaxonomy

logger = logging.getLogger(__name__)


PROMPT_VERSION = "v1"


def _build_prompt() -> str:
    """Create the metadata extraction prompt."""
    categories = [f"- {cat.value}: {DocumentCategory.get_display_name(cat.value)}" for cat in DocumentCategory]
    departments = [
        "engineering", "technical_support", "client_success",
        "supply_chain", "logistics", "sales", "finance",
        "marketing", "people_culture", "special_projects",
    ]
    return (
        "Extract structured metadata from this document and return as JSON.\n\n"
        f"CATEGORIES (choose exactly one):\n{chr(10).join(categories)}\n\n"
        f"DEPARTMENTS (choose one if applicable):\n{', '.join(departments)}\n\n"
        "EXTRACTION RULES:\n"
        "1. Category: Select the single most appropriate category based on purpose and structure\n"
        "2. Tags: Extract 5-10 relevant tags including product models (e.g., NC2068), topics, components, issues, actions\n"
        "3. Author: Extract author if present\n"
        "4. Department: Infer department if clear\n"
        "5. Version: Extract version if present (e.g., v1.0, Rev 2)\n"
        "6. Description: 1-2 sentence description of document purpose\n\n"
        "Return a JSON object:\n"
        "{\n"
        "  \"category\": \"category_value\",\n"
        "  \"tags\": [\"tag1\", \"tag2\"],\n"
        "  \"author\": \"author or null\",\n"
        "  \"department\": \"department or null\",\n"
        "  \"version\": \"version or null\",\n"
        "  \"description\": \"brief description\",\n"
        "  \"confidence_scores\": {\n"
        "    \"category\": 0.95, \"tags\": 0.85, \"author\": 0.70, \"department\": 0.80, \"version\": 0.60, \"description\": 0.90\n"
        "  }\n"
        "}\n\n"
        "IMPORTANT: Return valid JSON only. NC#### are non-conformity IDs, not product models."
    )


def _regex_product_models(text: str) -> List[str]:
    import re
    models: List[str] = []
    for pattern in (r"\bNC\d{4}\b", r"\bPC\d{4}\b", r"\bSM\d{3}\b"):
        models.extend(re.findall(pattern, text))
    return list(set(models))


def _regex_components(text: str) -> List[str]:
    import re
    found: List[str] = []
    tl = text.lower()
    for comp in TagTaxonomy.COMPONENTS:
        if re.search(r"\b" + re.escape(comp.lower()) + r"\b", tl):
            found.append(comp)
    return found


def _regex_issues(text: str) -> List[str]:
    import re
    found: List[str] = []
    tl = text.lower()
    for issue in TagTaxonomy.ISSUES:
        variations = [issue.lower(), issue.lower().replace("-", " "), issue.lower().replace("-", "")]
        for v in variations:
            if re.search(r"\b" + re.escape(v) + r"\b", tl):
                found.append(issue)
                break
    return found


def _merge_tags(llm_tags: List[str], content: str) -> List[str]:
    import re
    all_tags: List[str] = []
    if llm_tags:
        all_tags.extend(llm_tags)
    all_tags.extend(_regex_product_models(content))
    all_tags.extend(_regex_components(content))
    all_tags.extend(_regex_issues(content))

    seen = set()
    normalized: List[str] = []
    for tag in all_tags:
        norm = tag.lower().strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        # Keep original case for product-like codes
        normalized.append(tag if re.match(r"^[A-Z]+\d+", tag) else norm)

    def sort_key(t: str):
        if re.match(r"^[A-Z]+\d+", t):
            return (0, t)
        elif t in TagTaxonomy.COMPONENTS:
            return (1, t)
        elif t in TagTaxonomy.ISSUES:
            return (2, t)
        return (3, t)

    return sorted(normalized, key=sort_key)[:15]


class MetadataExtractionService:
    """Encapsulates LLM-based metadata extraction."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or "gpt-4o-mini"

    def extract(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content using OpenAI chat completions."""
        # Use at most 10k chars for cost/performance
        excerpt = content[:10000]
        prompt = _build_prompt()

        # Prefer the modern client API
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key or None)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Extract metadata from this document:\n\n{excerpt}"},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI client API failed, falling back: {e}")
            # Fallback to legacy style if available
            import openai as _openai
            if settings.openai_api_key:
                _openai.api_key = settings.openai_api_key
            try:
                resp = _openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Extract metadata from this document:\n\n{excerpt}"},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
            except Exception as e2:
                logger.error(f"Both OpenAI paths failed for metadata extraction: {e2}")
                raw = "{}"

        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.error("Metadata LLM returned non-JSON output; defaulting to empty")
            data = {}

        # Merge with regex-derived tags and audit info
        tags = _merge_tags(data.get("tags", []) or [], excerpt)
        return {
            "category": data.get("category"),
            "tags": tags,
            "author": data.get("author"),
            "department": data.get("department"),
            "version": data.get("version"),
            "description": data.get("description"),
            "confidence_scores": data.get("confidence_scores", {}),
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "extraction_model": self.model,
            "prompt_version": PROMPT_VERSION,
        }
