"""
Relationship Extraction Service
Hybrid approach: Rule-based + LLM extraction
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from app.models.relationships import (
    RelationshipType, 
    RelationshipProperties,
    Relationship,
    get_entity_type,
    ENTITY_TYPES
)
from app.services.llm_service import LLMService, LLMProvider


class RelationshipExtractor:
    """Extract relationships between entities using hybrid approach"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        
        # Patterns for rule-based extraction
        self.patterns = {
            "COMPONENT_OF": [
                r"([\w\s]+?)\s+(?:is|are)\s+(?:a|an)?\s*(?:component|part|module|feature)\s+(?:of|in)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:includes|contains|has)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:built into|integrated into)\s+([\w\s]+)"
            ],
            "CONNECTS_TO": [
                r"([\w\s]+?)\s+connects?\s+(?:to|with)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:interfaces?|communicates?)\s+with\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:links?|pairs?)\s+(?:to|with)\s+([\w\s]+)"
            ],
            "DEPENDS_ON": [
                r"([\w\s]+?)\s+(?:depends?|relies?)\s+(?:on|upon)\s+([\w\s]+)",
                r"([\w\s]+?)\s+requires?\s+([\w\s]+)",
                r"([\w\s]+?)\s+needs?\s+([\w\s]+?)\s+to\s+(?:function|work|operate)"
            ],
            "CAUSES": [
                r"([\w\s]+?)\s+causes?\s+([\w\s]+)",
                r"([\w\s]+?)\s+leads?\s+to\s+([\w\s]+)",
                r"([\w\s]+?)\s+results?\s+in\s+([\w\s]+)"
            ],
            "PREVENTS": [
                r"([\w\s]+?)\s+prevents?\s+([\w\s]+)",
                r"([\w\s]+?)\s+avoids?\s+([\w\s]+)",
                r"([\w\s]+?)\s+protects?\s+against\s+([\w\s]+)"
            ],
            "MITIGATES": [
                r"([\w\s]+?)\s+mitigates?\s+([\w\s]+)",
                r"([\w\s]+?)\s+reduces?\s+([\w\s]+)",
            ],
            "REPLACES": [
                r"([\w\s]+?)\s+(?:replaces?|supersedes?|upgrades?)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:is|are)\s+(?:the)?\s*(?:replacement|successor|upgrade)\s+(?:for|of|to)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:deprecated|obsolete).*?(?:replaced|succeeded)\s+by\s+([\w\s]+)"
            ],
            "RESPONSIBLE_FOR": [
                r"([\w\s]+?)\s+(?:team|department|group)\s+(?:is)?\s*responsible\s+for\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:owns?|maintains?|manages?)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:oversees?|handles?)\s+([\w\s]+)"
            ],
            "DEFINES": [
                r"([\w\s]+?)\s+defines?\s+([\w\s]+?)(?:\s+as|\s+to be|\.|,)",
                r"We define\s+([\w\s]+?)\s+as\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:establishes?|sets?)\s+([\w\s]+)"
            ],
            # Map SERVES patterns to USES/RESPONSIBLE if needed later; omitted here
            "IMPACTS": [
                r"([\w\s]+?)\s+impacts?\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:affects?|influences?)\s+([\w\s]+)"
            ]
        }
    
    async def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
        use_rules: bool = True
    ) -> List[Relationship]:
        """
        Extract relationships from text using hybrid approach
        
        Args:
            text: Document text to analyze
            entities: List of entities found in text
            document_metadata: Metadata about the document
            use_llm: Whether to use LLM extraction
            use_rules: Whether to use rule-based extraction
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Step 1: Create entity lookup for efficient matching
        entity_lookup = self._create_entity_lookup(entities)
        
        # Step 2: Rule-based extraction
        if use_rules:
            rule_relationships = self._extract_with_rules(text, entity_lookup)
            relationships.extend(rule_relationships)
        
        # Step 3: LLM extraction
        if use_llm and self.llm_service:
            llm_relationships = await self._extract_with_llm(
                text, entities, document_metadata
            )
            relationships.extend(llm_relationships)
        
        # Step 4: Document-Entity relationships (always add)
        doc_relationships = self._create_document_relationships(
            entities, document_metadata
        )
        relationships.extend(doc_relationships)
        
        # Step 5: Deduplicate and validate
        relationships = self._deduplicate_relationships(relationships)
        
        return relationships
    
    def _create_entity_lookup(self, entities: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Create lookup dictionary for efficient entity matching"""
        lookup = {}
        for entity in entities:
            name = entity.get("name", "")
            lookup[name.lower()] = entity
            
            # Add variations
            if " " in name:
                # Add acronym
                acronym = "".join(word[0] for word in name.split()).lower()
                lookup[acronym] = entity
            
            # Add without spaces
            lookup[name.replace(" ", "").lower()] = entity
            
        return lookup
    
    def _extract_with_rules(
        self, 
        text: str, 
        entity_lookup: Dict[str, Dict]
    ) -> List[Relationship]:
        """Extract relationships using regex patterns"""
        relationships = []
        
        for rel_type_str, patterns in self.patterns.items():
            rel_type = RelationshipType[rel_type_str]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    source_text = match.group(1).strip().lower()
                    target_text = match.group(2).strip().lower()
                    
                    # Check if both entities exist (try exact match first)
                    source_entity = entity_lookup.get(source_text)
                    target_entity = entity_lookup.get(target_text)
                    
                    # Safer fallback matching: require word-boundary or compact match; avoid loose substrings
                    def _safe_lookup(phrase: str) -> Optional[Dict]:
                        if not phrase:
                            return None
                        # Try compact variant (spaces removed) since lookup includes that
                        compact = phrase.replace(" ", "")
                        ent = entity_lookup.get(compact)
                        if ent:
                            return ent
                        # Try word-boundary exact phrase inside key
                        try:
                            wb = re.compile(rf"\b{re.escape(phrase)}\b")
                        except re.error:
                            wb = None
                        if wb:
                            for key, entity in entity_lookup.items():
                                if wb.search(key):
                                    return entity
                        return None

                    if not source_entity:
                        source_entity = _safe_lookup(source_text)

                    if not target_entity:
                        target_entity = _safe_lookup(target_text)
                    
                    if source_entity and target_entity:
                        # Validate against allowed source/target types for this relationship
                        s_type = get_entity_type(source_entity["name"])
                        t_type = get_entity_type(target_entity["name"])
                        if not rel_type.validate_entities(s_type, t_type):
                            continue

                        relationship = Relationship(
                            source_entity=source_entity["name"],
                            source_type=s_type,
                            relationship_type=rel_type,
                            target_entity=target_entity["name"],
                            target_type=t_type,
                            properties=RelationshipProperties(
                                confidence=0.8,  # Lower confidence for rule-based
                                source_text=match.group(0),
                                extracted_at=datetime.now(),
                                extraction_method="rule_based"
                            )
                        )
                        relationships.append(relationship)
        
        return relationships
    
    async def _extract_with_llm(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Relationship]:
        """Extract relationships using LLM returning a JSON payload.

        Note: Previously this incorrectly called extract_metadata(), which prompts for
        document metadata (title/author/etc). We now call call_llm() directly with a
        relationships-specific prompt and parse JSON from response.content.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[REL-DEBUG] Starting LLM relationship extraction with {len(entities)} entities, text length: {len(text)}")
        relationships: List[Relationship] = []

        # Build context about document type for better guidance
        doc_context = ""
        if document_metadata:
            if isinstance(document_metadata, dict):
                doc_type = document_metadata.get("type", "document")
                department = document_metadata.get("department", "unknown")
            else:
                doc_type = getattr(document_metadata, "type", "document")
                department = getattr(document_metadata, "department", "unknown")
            doc_context = f"This is a {doc_type} from the {department} department."

        # Compose relationships extraction prompt
        prompt = self._build_llm_prompt(text, entities, doc_context)

        # Call GPT-5-mini with reasoning effort for deeper relationship inference
        system_prompt = (
            "You are a technical analyst extracting relationship networks from documentation. "
            "Use reasoning to identify both explicit and implicit relationships. "
            "Consider contextual clues, co-occurrence patterns, and functional dependencies. "
            "Respond ONLY with strict JSON matching the requested schema."
        )

        # NO FALLBACK: Let exceptions propagate to expose LLM API errors
        # NOTE: GPT-5 with reasoning mode does NOT support response_format parameter
        # We rely on prompt engineering to get JSON output instead
        llm_resp = await self.llm_service.call_llm(
            prompt=prompt,
            provider=LLMProvider.OPENAI,
            model="gpt-5-mini",  # GPT-5-mini with reasoning capabilities
            reasoning_effort="low",  # Use "low" to avoid JSON truncation (medium/high produce verbose output)
            # Increase completion budget to reduce truncation in larger docs
            max_tokens=10000,  # Increased to handle full JSON response
            max_completion_tokens=10000,
            system_prompt=system_prompt,
            timeout=120,  # Increased timeout for reasoning
            # response_format removed: Not compatible with reasoning mode
        )

        content = (llm_resp.content or "").strip()

        # DEBUG: Log raw GPT-5 response to diagnose format issues
        logger.warning(f"[REL-DEBUG] Raw LLM response length: {len(content)} chars")
        logger.warning(f"[REL-DEBUG] Raw LLM response (first 500 chars): {content[:500]}")
        if not content:
            logger.error("[REL-DEBUG] LLM returned empty content!")
            raise ValueError("LLM returned empty content for relationship extraction")

        # Strip common code fences
        if content.startswith("```json"):
            content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        elif content.startswith("```"):
            content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        data: Dict[str, Any]
        try:
            data = json.loads(content)
            logger.warning(f"[REL-DEBUG] Successfully parsed JSON with {len(data.get('relationships', []))} relationships")
        except json.JSONDecodeError as je:
            logger.error(f"[REL-DEBUG] Failed to parse LLM response as JSON: {je}")
            logger.error(f"[REL-DEBUG] Failed content (full): {content}")
            raise ValueError(f"LLM returned invalid JSON for relationship extraction: {je}") from je

        for rel_data in (data.get("relationships") or []):
            try:
                # Map relationship type label -> enum (fallback to RELATES_TO)
                rel_label = rel_data.get("relationship_type", "RELATES_TO")
                rel_type = RelationshipType.get_by_label(rel_label) or RelationshipType.RELATES_TO

                # Prefer provided canonical IDs if present
                source_id = rel_data.get("source_id")
                target_id = rel_data.get("target_id")
                source_name = rel_data.get("source_name") or rel_data.get("source_entity")
                target_name = rel_data.get("target_name") or rel_data.get("target_entity")

                props = RelationshipProperties(
                    confidence=float(rel_data.get("confidence", 0.7) or 0.7),
                    source_text=rel_data.get("context", ""),
                    extracted_at=datetime.now(),
                    extraction_method="llm_gpt4",
                    additional_properties={
                        **(rel_data.get("properties", {}) or {}),
                        **({"source_id": source_id} if source_id else {}),
                        **({"target_id": target_id} if target_id else {}),
                        **({"source_name": source_name} if source_name else {}),
                        **({"target_name": target_name} if target_name else {}),
                    },
                )

                relationships.append(
                    Relationship(
                        source_entity=source_id or rel_data.get("source_entity") or "",
                        source_type=rel_data.get("source_type", "Entity"),
                        relationship_type=rel_type,
                        target_entity=target_id or rel_data.get("target_entity") or "",
                        target_type=rel_data.get("target_type", "Entity"),
                        properties=props,
                    )
                )
            except Exception as e:
                logger.error(f"[REL-DEBUG] Error parsing individual relationship: {e}, data: {rel_data}")
                # Continue parsing other relationships but log the error
                continue

        return relationships
    
    def _build_llm_prompt(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        doc_context: str
    ) -> str:
        """Build prompt for LLM relationship extraction"""
        
        # Get relationship descriptions
        rel_descriptions = []
        for rel_type in RelationshipType:
            desc = f"- {rel_type.label}: {', '.join(rel_type.suggested_properties.keys())}"
            rel_descriptions.append(desc)
        
        # Provide the LLM with entities including IDs and instruct it to return IDs
        entity_list = [
            {"id": e.get("id"), "name": e.get("name"), "type": e.get("type"), "aliases": e.get("aliases", [])}
            for e in (entities or [])
        ]

        prompt = f"""
You are analyzing documentation for a smart water dispenser company.
{doc_context}

Extract ALL relationships between the following entities (each has a stable canonical "id").
Use only these entities; do not invent new entities. Refer to entities by their "id" in outputs:
{json.dumps(entity_list, indent=2)}

Use these relationship types:
{chr(10).join(rel_descriptions)}

For each relationship, provide:
1. source_id: ID of source entity (REQUIRED, must match an id above)
2. source_name: Name of the source entity (echo for readability)
3. source_type: Type of source entity
4. relationship_type: One of the types above
5. target_id: ID of target entity (REQUIRED, must match an id above)
6. target_name: Name of the target entity (echo for readability)
7. target_type: Type of target entity
8. confidence: 0.0 to 1.0 (0.9+ for explicit, 0.6-0.8 for inferred)
9. context: Quote from text supporting this relationship
10. properties: Additional properties relevant to the relationship type

EXTRACTION STRATEGY - Extract BOTH types of relationships:

A. EXPLICIT relationships (directly stated):
   Example: "The X500 connects to the mobile app via Bluetooth"
   → X500 CONNECTS_TO mobile app (confidence: 0.95)

B. IMPLICIT relationships (inferred from context):
   Example: "The X500 troubleshooting guide covers pump failures"
   → X500 RELATES_TO pump failures (confidence: 0.75, inferred from troubleshooting context)

   Example: "The Engineering team maintains the firmware and oversees hardware updates"
   → Engineering team RESPONSIBLE_FOR firmware (explicit)
   → Engineering team RESPONSIBLE_FOR hardware (explicit)
   → firmware RELATES_TO hardware (implicit co-management relationship, confidence: 0.7)

C. CO-FUNCTIONAL relationships:
   When entities work together, appear in same context, or jointly achieve outcomes:
   Example: "The control board and temperature sensor work together to regulate water temperature"
   → control board CONNECTS_TO temperature sensor (explicit)
   → control board DEPENDS_ON temperature sensor (implied functional dependency, confidence: 0.75)

D. TRANSITIVE relationships:
   Consider relationship chains where applicable:
   Example: "Firmware A runs on Control Board B, which powers Component C"
   → Firmware A RUNS_ON Control Board B
   → Control Board B COMPONENT_OF Component C
   → Firmware A IMPACTS Component C (inferred transitive, confidence: 0.65)

Rules:
- Always return source_id and target_id that exist in the provided list
- Do not fabricate ids or entities; if unsure, omit the relationship
- Extract both explicit AND inferred relationships
- For inferred relationships, use lower confidence (0.6-0.8) and explain reasoning in context
- Look for co-occurrence patterns, functional dependencies, and contextual associations
- Consider document type: technical docs emphasize DEPENDS_ON/COMPONENT_OF, support docs emphasize IMPACTS/CAUSES

Focus on:
- Technical dependencies and connections (for engineering docs)
- Troubleshooting cause-effect chains (for support docs)
- Target audiences and responsibilities (for sales/marketing)
- Component hierarchies and compatibility relationships
- Functional co-dependencies and interaction patterns

Text to analyze:
{text[:15000]}

Return as JSON:
{{
  "relationships": [
    {{
      "source_id": "<id from list>",
      "source_name": "Model X500",
      "source_type": "Product",
      "relationship_type": "COMPONENT_OF",
      "target_id": "<id from list>",
      "target_name": "Premium Line",
      "target_type": "Product",
      "confidence": 0.95,
      "context": "\"The X500 is our flagship model in the Premium Line\"",
      "properties": {{
        "component_type": "product_model",
        "position": "flagship",
        "extraction_type": "explicit"
      }}
    }},
    {{
      "source_id": "<id from list>",
      "source_name": "Control Board",
      "source_type": "Component",
      "relationship_type": "RELATES_TO",
      "target_id": "<id from list>",
      "target_name": "Temperature Sensor",
      "target_type": "Component",
      "confidence": 0.75,
      "context": "\"Both mentioned in temperature regulation section\"",
      "properties": {{
        "reasoning": "Co-functional relationship inferred from shared context",
        "extraction_type": "implicit"
      }}
    }}
  ]
}}
"""
        return prompt
    
    def _create_document_relationships(
        self,
        entities: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Relationship]:
        """Create relationships between document and entities"""
        relationships = []
        
        if not document_metadata:
            return relationships
        
        doc_id = document_metadata.get("id", "unknown_doc")
        doc_title = document_metadata.get("title", "Document")
        
        for entity in entities:
            # Document DOCUMENTS entity
            relationship = Relationship(
                source_entity=doc_title,
                source_type="Document",
                relationship_type=RelationshipType.DOCUMENTS,
                target_entity=entity["name"],
                target_type=get_entity_type(entity["name"]),
                properties=RelationshipProperties(
                    confidence=0.9,
                    source_text=f"Document mentions {entity['name']}",
                    extracted_at=datetime.now(),
                    extraction_method="automatic",
                    additional_properties={
                        "mention_count": entity.get("count", 1),
                        "first_mention": entity.get("first_position", 0),
                        # Provide document id/name so downstream can map without fuzzy name resolution
                        "source_id": str(doc_id),
                        "source_name": doc_title,
                        "target_name": entity.get("name"),
                    }
                )
            )
            relationships.append(relationship)
        
        return relationships
    
    def _deduplicate_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """Remove duplicate relationships, keeping highest confidence"""
        seen = {}
        
        for rel in relationships:
            # Create key for relationship
            key = (
                rel.source_entity,
                rel.relationship_type.label,
                rel.target_entity
            )
            
            if key not in seen:
                seen[key] = rel
            else:
                # Keep higher confidence
                if rel.properties.confidence > seen[key].properties.confidence:
                    seen[key] = rel
        
        return list(seen.values())
    
    async def extract_with_context(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        department: str,
        doc_type: str
    ) -> List[Relationship]:
        """
        Extract relationships with specific department context
        
        Args:
            text: Document text
            entities: Entities in document
            department: Department (Sales, Engineering, Support, etc.)
            doc_type: Document type (manual, spec, playbook, etc.)
            
        Returns:
            List of relationships optimized for department
        """
        # Adjust extraction based on department
        use_rules = True
        use_llm = True
        
        if department == "Engineering":
            # Focus on technical relationships
            self.patterns["DEPENDS_ON"].append(
                r"(\w+)\s+firmware\s+(?:for|on)\s+(\w+)"
            )
            self.patterns["COMPATIBLE_WITH"] = [
                r"(\w+)\s+compatible\s+with\s+(\w+)",
                r"(\w+)\s+supports?\s+(\w+)"
            ]
            
        elif department == "Support":
            # Focus on troubleshooting
            self.patterns["TROUBLESHOOTS"] = [
                r"error\s+(?:code)?\s*([\w\s]+?).*?(?:indicates?|means?|shows?)\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:can be)?\s*troubleshooted?\s+by\s+([\w\s]+)"
            ]
            
        elif department == "Sales" or department == "Marketing":
            # Focus on market relationships
            self.patterns["TARGETS"] = [
                r"(\w+)\s+(?:targets?|aimed at|for)\s+(\w+)\s+(?:segment|market|customers?)",
                r"(\w+)\s+(?:serves?|designed for)\s+(\w+)"
            ]
        
        # Extract with context
        metadata = {
            "department": department,
            "type": doc_type,
            "extraction_time": datetime.now().isoformat()
        }
        
        relationships = await self.extract_relationships(
            text, entities, metadata, use_llm, use_rules
        )
        
        return relationships


# Test function
async def test_relationship_extraction():
    """Test relationship extraction"""
    print("\n" + "="*60)
    print("RELATIONSHIP EXTRACTION TEST")
    print("="*60)
    
    # Sample text
    text = """
    The Model X500 is our flagship water dispenser in the Premium Line.
    It features an advanced flavoring module that connects to the main unit via Bluetooth 5.0.
    The temperature sensor depends on the control board for proper operation.
    
    The Engineering team is responsible for firmware updates and maintaining the control systems.
    This model replaces the older X400 series, offering better performance and reliability.
    
    Common error code E501 indicates a pump failure, which can be troubleshooted by checking
    the water line connections. The Tech Support team collaborates with Engineering to resolve
    complex technical issues.
    
    The X500 targets enterprise customers and serves the premium market segment.
    All components are compatible with our mobile app version 2.0 or higher.
    """
    
    # Sample entities
    entities = [
        {"name": "Model X500", "type": "Product"},
        {"name": "Premium Line", "type": "Product"},
        {"name": "flavoring module", "type": "Component"},
        {"name": "Bluetooth 5.0", "type": "Technology"},
        {"name": "temperature sensor", "type": "Component"},
        {"name": "control board", "type": "Component"},
        {"name": "Engineering team", "type": "Department"},
        {"name": "firmware updates", "type": "Process"},
        {"name": "X400 series", "type": "Product"},
        {"name": "E501", "type": "Issue"},
        {"name": "pump failure", "type": "Issue"},
        {"name": "Tech Support team", "type": "Department"},
        {"name": "enterprise customers", "type": "Segment"},
        {"name": "premium market", "type": "Market"},
        {"name": "mobile app", "type": "Software"}
    ]
    
    # Extract relationships
    extractor = RelationshipExtractor()
    
    # Test rule-based only
    print("\n1. Rule-based Extraction:")
    relationships = await extractor.extract_relationships(
        text, entities, use_llm=False, use_rules=True
    )
    
    for rel in relationships[:5]:
        print(f"  {rel.source_entity} --{rel.relationship_type.label}--> {rel.target_entity}")
        print(f"    Confidence: {rel.properties.confidence:.2f}")
        print(f"    Context: {rel.properties.source_text[:50]}...")
    
    print(f"\n  Total relationships found: {len(relationships)}")
    
    # Test with department context
    print("\n2. Department-specific Extraction (Engineering):")
    eng_relationships = await extractor.extract_with_context(
        text, entities, "Engineering", "technical_spec"
    )
    
    tech_rels = [r for r in eng_relationships if r.relationship_type in [
        RelationshipType.DEPENDS_ON,
        RelationshipType.CONNECTS_TO,
        RelationshipType.COMPONENT_OF
    ]]
    
    print(f"  Technical relationships found: {len(tech_rels)}")
    for rel in tech_rels[:3]:
        print(f"    {rel.source_entity} --{rel.relationship_type.label}--> {rel.target_entity}")


if __name__ == "__main__":
    asyncio.run(test_relationship_extraction())
