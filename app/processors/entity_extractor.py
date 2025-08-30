"""Entity extractor using multi-model LLM support"""
import asyncio
import logging
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from app.models.chunk import Chunk
from app.models.entity import Entity, EntityRelationship, EntityType
from app.services.llm_service import LLMService, LLMProvider, DocumentMetadata

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract entities and relationships from text using LLMs"""
    
    def __init__(self):
        self.llm_service = LLMService()
    
    def extract(
        self, 
        chunks: List[Chunk], 
        document_id: str
    ) -> Tuple[List[Entity], List[EntityRelationship]]:
        """
        Extract entities and relationships from document chunks
        
        Args:
            chunks: List of document chunks
            document_id: Document ID
            
        Returns:
            Tuple of (entities, relationships)
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")
        
        # Check if an event loop is already running
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context
            # Create a task instead of trying to run a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._extract_async(chunks, document_id))
                return future.result()
        except RuntimeError:
            # No loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._extract_async(chunks, document_id)
                )
            finally:
                loop.close()
    
    async def _extract_async(
        self,
        chunks: List[Chunk],
        document_id: str
    ) -> Tuple[List[Entity], List[EntityRelationship]]:
        """Async entity extraction with multi-model comparison"""
        
        all_entities = []
        entity_map = {}  # Track unique entities for deduplication
        
        # Process chunks in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract entities from each chunk in parallel
            tasks = [
                self._extract_from_chunk(chunk, document_id)
                for chunk in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for chunk_entities in batch_results:
                if isinstance(chunk_entities, Exception):
                    logger.error(f"Entity extraction failed for chunk: {chunk_entities}")
                    continue
                
                for entity_data in chunk_entities:
                    # Deduplicate entities
                    entity_key = (
                        entity_data["name"].lower(),
                        entity_data["type"]
                    )
                    
                    if entity_key in entity_map:
                        # Update confidence if higher
                        existing = entity_map[entity_key]
                        if entity_data["confidence"] > existing.confidence:
                            existing.confidence = entity_data["confidence"]
                    else:
                        # Create new entity
                        entity = Entity(
                            document_id=document_id,
                            name=entity_data["name"],  # Use 'name' as required field
                            type=entity_data["type"],  # Use 'type' as string
                            entity_name=entity_data["name"],  # Also set entity_name for backwards compatibility
                            entity_type=self._map_entity_type(entity_data["type"]),
                            confidence=entity_data["confidence"],
                            metadata={
                                "context": entity_data.get("context", ""),
                                "chunk_id": entity_data.get("chunk_id"),
                                "description": self._generate_entity_description(entity_data),
                                "category": self._determine_subcategory(entity_data["type"], entity_data["name"]),
                                "attributes": self._extract_type_specific_attributes(entity_data)
                            }
                        )
                        entity_map[entity_key] = entity
                        all_entities.append(entity)
        
        # Extract relationships between entities
        relationships = await self._extract_relationships(
            all_entities, chunks, document_id
        )
        
        logger.info(f"Extracted {len(all_entities)} unique entities and {len(relationships)} relationships")
        
        return all_entities, relationships
    
    async def _extract_from_chunk(
        self,
        chunk: Chunk,
        document_id: str
    ) -> List[Dict]:
        """Extract entities from a single chunk"""
        
        try:
            # Try with primary provider (OpenAI)
            entities = await self.llm_service.extract_entities(
                text=chunk.chunk_text,
                provider=LLMProvider.OPENAI
            )
            
            # If OpenAI fails or returns empty, try Gemini
            if not entities and self.llm_service.gemini_client:
                entities = await self.llm_service.extract_entities(
                    text=chunk.chunk_text,
                    provider=LLMProvider.GEMINI
                )
            
            # Convert to dict format with chunk reference
            return [
                {
                    "name": e.name,
                    "type": e.type,
                    "confidence": e.confidence,
                    "context": e.context,
                    "chunk_id": chunk.id
                }
                for e in entities
            ]
            
        except Exception as e:
            logger.error(f"Failed to extract entities from chunk: {e}")
            return []
    
    async def _extract_relationships(
        self,
        entities: List[Entity],
        chunks: List[Chunk],
        document_id: str
    ) -> List[EntityRelationship]:
        """Extract relationships between entities"""
        
        relationships = []
        
        # Only extract relationships if we have multiple entities
        if len(entities) < 2:
            return relationships
        
        # Create entity name lookup
        entity_names = {e.entity_name.lower(): e for e in entities}
        
        # Analyze chunks for relationships
        for chunk in chunks[:10]:  # Limit to first 10 chunks for performance
            
            # Find entities mentioned in this chunk
            mentioned_entities = [
                e for e in entities
                if e.entity_name.lower() in chunk.chunk_text.lower()
            ]
            
            # If multiple entities in chunk, they might be related
            if len(mentioned_entities) >= 2:
                # Use LLM to identify relationships
                relationship_data = await self._identify_relationships(
                    chunk.chunk_text,
                    mentioned_entities
                )
                
                for rel in relationship_data:
                    source = entity_names.get(rel["source"].lower())
                    target = entity_names.get(rel["target"].lower())
                    
                    if source and target and source.id != target.id:
                        relationship = EntityRelationship(
                            source_entity_id=source.id,
                            target_entity_id=target.id,
                            relationship_type=rel["type"],
                            confidence_score=rel.get("confidence", 0.5),
                            metadata={
                                "context": rel.get("context", ""),
                                "chunk_id": chunk.id
                            }
                        )
                        relationships.append(relationship)
        
        # Deduplicate relationships
        unique_relationships = []
        seen = set()
        
        for rel in relationships:
            key = (rel.source_entity_id, rel.target_entity_id, rel.relationship_type)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    async def _identify_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Dict]:
        """Use LLM to identify relationships between entities in text"""
        
        entity_names = [e.entity_name for e in entities]
        
        prompt = f"""You are building a knowledge graph to represent how concepts interconnect in this text. Identify relationships that reveal the structure of knowledge and causal chains within the document.

Text: {text}

Entities: {', '.join(entity_names)}

AVAILABLE RELATIONSHIP TYPES:
Technical Relationships:
- COMPONENT_OF: Part of a larger system
- CONNECTS_TO: Physical or logical connection
- DEPENDS_ON: Requires for operation
- REPLACES: Substitutes or upgrades
- COMPATIBLE_WITH: Works together with
- TROUBLESHOOTS: Diagnoses or tests
- CAUSES: Direct causation
- PREVENTS: Stops from occurring
- REQUIRES: Necessary prerequisite
- RESOLVES: Fixes or addresses
- INDICATES: Symptom or sign of

Documentation Relationships:
- DEFINES: Provides definition
- DOCUMENTS: Records or describes
- REFERENCES: Points to
- TARGETS: Aimed at or affects

Business Relationships:
- RESPONSIBLE_FOR: Has duty or ownership
- SERVES: Provides service to
- IMPACTS: Affects or influences

General:
- RELATES_TO: General association

EXTRACTION APPROACH:
1. Identify the document's purpose (troubleshooting, specification, guide, analysis)
2. Based on purpose, prioritize finding:
   - Troubleshooting: problem→cause→solution chains
   - Specifications: component hierarchies and requirements
   - Guides: sequence of steps and their prerequisites
   - Analysis: evidence→conclusion relationships

3. Look for:
   - Causal chains: If X causes Y which requires Z, create:
     • X → CAUSES → Y
     • Y → REQUIRES → Z
   - Hierarchical structures: If A contains B which includes C, create:
     • A → COMPONENT_OF → B  
     • B → COMPONENT_OF → C
   - Problem-solution pairs: 
     • Solution → RESOLVES → Problem
   - Dependencies:
     • A → DEPENDS_ON → B
   - Symptoms and their sources:
     • Symptom → INDICATES → Root Cause
   
   Break complex relationships into multiple binary connections that together tell the complete story.

4. Include relationships even if implicit but clearly implied by context

Return a JSON array of relationships with:
- source: source entity name (exact match from list)
- target: target entity name (exact match from list)
- type: relationship type from above list
- confidence: confidence score (0-1)

Only include relationships where both entities are in the provided entity list.
Focus on relationships that add to understanding rather than trivial connections."""

        try:
            response = await self.llm_service.call_with_fallback(
                prompt=prompt,
                primary_provider=LLMProvider.OPENAI,
                temperature=0.3,
                max_tokens=1500  # Increased to avoid cutoff
            )
            
            import json
            
            # Log the raw response for debugging
            logger.debug(f"Raw relationship response: {response.content}")
            
            # Handle empty or None response
            if not response.content or response.content.strip() == "":
                logger.info("LLM returned empty response for relationships")
                return []
            
            # Try to parse JSON, handling potential markdown wrapping
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            
            content = content.strip()
            
            # If the response is "None" or similar, return empty list
            if content.lower() in ["none", "null", "[]"]:
                return []
            
            relationships = json.loads(content)
            return relationships if isinstance(relationships, list) else []
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationships JSON: {e}")
            logger.error(f"Raw content was: {response.content if 'response' in locals() else 'No response'}")
            return []
        except Exception as e:
            logger.error(f"Failed to identify relationships: {e}")
            return []
    
    def _map_entity_type(self, llm_type: str) -> EntityType:
        """Map LLM entity type to our EntityType enum"""
        
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "COMPANY": EntityType.ORGANIZATION,
            "LOCATION": EntityType.LOCATION,
            "PLACE": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "PRODUCT": EntityType.PRODUCT,
            "COMPONENT": EntityType.COMPONENT,
            "TECHNOLOGY": EntityType.TECHNOLOGY,
            "CHEMICAL": EntityType.CHEMICAL,
            "PROCEDURE": EntityType.PROCEDURE,
            "SPECIFICATION": EntityType.SPECIFICATION,
            "SYSTEM": EntityType.SYSTEM,
            "MEASUREMENT": EntityType.MEASUREMENT,
            "PROBLEM": EntityType.PROBLEM,
            "STATE": EntityType.STATE,
            "CONDITION": EntityType.CONDITION,
            "CONCEPT": EntityType.CONCEPT,
            "EVENT": EntityType.EVENT
        }
        
        return mapping.get(llm_type.upper(), EntityType.OTHER)
    
    def _generate_entity_description(self, entity_data: Dict) -> str:
        """Generate a brief description for the entity"""
        context = entity_data.get("context", "")
        entity_type = entity_data.get("type", "").upper()
        name = entity_data.get("name", "")
        
        # Generate type-specific descriptions
        if entity_type == "COMPONENT":
            return f"Hardware/software component: {context}" if context else f"Component named {name}"
        elif entity_type == "CHEMICAL":
            return f"Chemical substance: {context}" if context else f"Chemical compound {name}"
        elif entity_type == "PROCEDURE":
            return f"Process or method: {context}" if context else f"Procedure {name}"
        elif entity_type == "SPECIFICATION":
            return f"Standard or requirement: {context}" if context else f"Specification {name}"
        elif entity_type == "SYSTEM":
            return f"System or subsystem: {context}" if context else f"System {name}"
        elif entity_type == "MEASUREMENT":
            return f"Measurement or dimension: {context}" if context else f"Measurement {name}"
        elif entity_type == "PROBLEM":
            return f"Issue or symptom: {context}" if context else f"Problem: {name}"
        elif entity_type == "STATE":
            return f"Operational state: {context}" if context else f"State: {name}"
        elif entity_type == "CONDITION":
            return f"Physical condition: {context}" if context else f"Condition: {name}"
        else:
            return context or f"{entity_type.title()} entity"
    
    def _determine_subcategory(self, entity_type: str, name: str) -> str:
        """Determine a subcategory within the entity type"""
        entity_type = entity_type.upper()
        name_lower = name.lower()
        
        if entity_type == "COMPONENT":
            if any(term in name_lower for term in ["cable", "connector", "wire"]):
                return "electrical"
            elif any(term in name_lower for term in ["display", "screen", "monitor"]):
                return "display"
            elif any(term in name_lower for term in ["sensor", "detector"]):
                return "sensor"
            else:
                return "general"
        
        elif entity_type == "CHEMICAL":
            if "acid" in name_lower:
                return "acid"
            elif "alcohol" in name_lower:
                return "alcohol"
            elif any(term in name_lower for term in ["oxide", "hydroxide"]):
                return "compound"
            else:
                return "substance"
        
        elif entity_type == "PROCEDURE":
            if any(term in name_lower for term in ["clean", "wash", "wipe"]):
                return "cleaning"
            elif any(term in name_lower for term in ["test", "verify", "check"]):
                return "testing"
            elif any(term in name_lower for term in ["install", "setup", "configure"]):
                return "installation"
            else:
                return "process"
        
        elif entity_type == "PROBLEM":
            if any(term in name_lower for term in ["screen", "display", "visual"]):
                return "display_issue"
            elif any(term in name_lower for term in ["connect", "cable", "wire"]):
                return "connection_issue"
            elif any(term in name_lower for term in ["error", "fail", "crash"]):
                return "error"
            else:
                return "symptom"
        
        elif entity_type == "STATE":
            if any(term in name_lower for term in ["active", "running", "operational"]):
                return "active"
            elif any(term in name_lower for term in ["locked", "blocked", "frozen"]):
                return "blocked"
            elif any(term in name_lower for term in ["failed", "error", "broken"]):
                return "failed"
            else:
                return "operational"
        
        elif entity_type == "CONDITION":
            if any(term in name_lower for term in ["corrosion", "rust", "oxidation"]):
                return "corrosion"
            elif any(term in name_lower for term in ["contamination", "dirty", "debris"]):
                return "contamination"
            elif any(term in name_lower for term in ["wear", "worn", "degraded"]):
                return "wear"
            else:
                return "physical"
        
        return "general"
    
    def _extract_type_specific_attributes(self, entity_data: Dict) -> Dict:
        """Extract type-specific attributes for the entity"""
        entity_type = entity_data.get("type", "").upper()
        name = entity_data.get("name", "")
        context = entity_data.get("context", "")
        
        attributes = {}
        
        if entity_type == "MEASUREMENT":
            # Try to extract value and unit
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)', name)
            if match:
                attributes["value"] = match.group(1)
                attributes["unit"] = match.group(2)
        
        elif entity_type == "CHEMICAL":
            # Store chemical properties if identifiable
            if "isopropyl alcohol" in name.lower():
                attributes["formula"] = "C3H8O"
                attributes["common_name"] = "IPA"
            elif "ethanol" in name.lower():
                attributes["formula"] = "C2H6O"
                attributes["common_name"] = "ethyl alcohol"
        
        elif entity_type == "SPECIFICATION":
            # Extract standard numbers if present
            import re
            iso_match = re.search(r'ISO\s*(\d+)', name)
            if iso_match:
                attributes["standard_number"] = iso_match.group(1)
                attributes["standard_type"] = "ISO"
        
        return attributes