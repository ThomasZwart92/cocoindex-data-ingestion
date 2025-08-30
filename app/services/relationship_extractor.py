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
from app.services.llm_service import LLMService


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
            "SERVES": [
                r"([\w\s]+?)\s+serves?\s+([\w\s]+)",
                r"([\w\s]+?)\s+(?:for|to)\s+([\w\s]+?)\s+(?:customers?|clients?|users?)"
            ],
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
                    
                    # Try partial matches if exact match fails
                    if not source_entity:
                        for key, entity in entity_lookup.items():
                            if source_text in key or key in source_text:
                                source_entity = entity
                                break
                    
                    if not target_entity:
                        for key, entity in entity_lookup.items():
                            if target_text in key or key in target_text:
                                target_entity = entity
                                break
                    
                    if source_entity and target_entity:
                        relationship = Relationship(
                            source_entity=source_entity["name"],
                            source_type=get_entity_type(source_entity["name"]),
                            relationship_type=rel_type,
                            target_entity=target_entity["name"],
                            target_type=get_entity_type(target_entity["name"]),
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
        """Extract relationships using LLM"""
        relationships = []
        
        # Build context about document type
        doc_context = ""
        if document_metadata:
            if isinstance(document_metadata, dict):
                doc_type = document_metadata.get("type", "document")
                department = document_metadata.get("department", "unknown")
            else:
                doc_type = getattr(document_metadata, 'type', 'document')
                department = getattr(document_metadata, 'department', 'unknown')
            doc_context = f"This is a {doc_type} from the {department} department."
        
        # Create prompt
        prompt = self._build_llm_prompt(text, entities, doc_context)
        
        try:
            # Call LLM
            response = await self.llm_service.extract_metadata(prompt)
            
            # Parse response
            if isinstance(response, str):
                response = json.loads(response)
            
            # Convert to Relationship objects
            for rel_data in response.get("relationships", []):
                try:
                    # Find the relationship type
                    rel_type = RelationshipType.get_by_label(
                        rel_data.get("relationship_type", "RELATES_TO")
                    )
                    if not rel_type:
                        rel_type = RelationshipType.RELATES_TO
                    
                    # Build properties
                    props = RelationshipProperties(
                        confidence=rel_data.get("confidence", 0.7),
                        source_text=rel_data.get("context", ""),
                        extracted_at=datetime.now(),
                        extraction_method="llm_gpt4",
                        additional_properties=rel_data.get("properties", {})
                    )
                    
                    relationship = Relationship(
                        source_entity=rel_data["source_entity"],
                        source_type=rel_data.get("source_type", "Entity"),
                        relationship_type=rel_type,
                        target_entity=rel_data["target_entity"],
                        target_type=rel_data.get("target_type", "Entity"),
                        properties=props
                    )
                    relationships.append(relationship)
                    
                except Exception as e:
                    print(f"Error parsing LLM relationship: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
        
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
        
        prompt = f"""
You are analyzing documentation for a smart water dispenser company.
{doc_context}

Extract ALL relationships between the following entities:
{json.dumps(entities, indent=2)}

Use these relationship types:
{chr(10).join(rel_descriptions)}

For each relationship, provide:
1. source_entity: Name of source entity
2. source_type: Type of source entity
3. relationship_type: One of the types above
4. target_entity: Name of target entity  
5. target_type: Type of target entity
6. confidence: 0.0 to 1.0
7. context: Quote from text supporting this relationship
8. properties: Additional properties relevant to the relationship type

Focus on:
- Technical dependencies and connections (for engineering docs)
- Troubleshooting relationships (for support docs)
- Target audiences and responsibilities (for sales/marketing)
- Component hierarchies and compatibility

Text to analyze:
{text[:3000]}  # Limit to avoid token limits

Return as JSON:
{{
    "relationships": [
        {{
            "source_entity": "Model X500",
            "source_type": "Product",
            "relationship_type": "COMPONENT_OF",
            "target_entity": "Premium Line",
            "target_type": "Product",
            "confidence": 0.95,
            "context": "The X500 is our flagship model in the Premium Line",
            "properties": {{
                "component_type": "product_model",
                "position": "flagship"
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
                        "first_mention": entity.get("first_position", 0)
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