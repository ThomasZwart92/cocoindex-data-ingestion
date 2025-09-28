"""Entity extractor using multi-model LLM support (LEGACY)

DEPRECATED: This legacy extractor is disabled by default to prevent accidental
execution. Enable only if you explicitly need legacy behavior.

To enable temporarily, set environment variable:
  COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1

Prefer using the v2 CocoIndex-based entity pipeline.
"""
import os
import asyncio
import json
import logging
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from app.models.chunk import Chunk
from app.models.entity import Entity, EntityRelationship, EntityType
from app.services.llm_service import LLMService, LLMProvider, DocumentMetadata
from app.utils.entity_deduplication import EntityDeduplicator
from app.utils.entity_quality import EntityQualityValidator

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract entities and relationships from text using LLMs"""
    
    def __init__(self):
        if os.getenv("COCOINDEX_LEGACY_ENTITY_EXTRACTOR", "0") != "1":
            raise RuntimeError(
                "Legacy EntityExtractor is disabled. Set COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1 "
                "to allow temporary use, or migrate to the v2 CocoIndex pipeline."
            )
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
        """Async entity extraction using full document context"""
        
        # Check if we should use full document extraction (for higher quality)
        # Filter to only parent chunks (not semantic sub-chunks)
        # If no chunk_type metadata, assume chunks without 'parent_chunk_id' are parent chunks
        parent_chunks = []
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type') if chunk.metadata else None
            if chunk_type == 'parent':
                parent_chunks.append(chunk)
            elif chunk_type is None:
                # No chunk_type metadata, check if it's a sub-chunk
                if not (chunk.metadata and chunk.metadata.get('parent_chunk_id')):
                    parent_chunks.append(chunk)
        
        # If no parent chunks identified, use all chunks
        if not parent_chunks:
            logger.info("No parent chunks identified, using all chunks")
            parent_chunks = chunks
        
        # Combine parent chunks to get full document text
        full_text = "\n\n".join([chunk.chunk_text for chunk in parent_chunks])
        
        # Check document size - if reasonable, use full document extraction
        if len(full_text) < 100000:  # Under ~100k chars, use full document
            logger.info(f"Using full document extraction for {len(full_text)} chars")
            return await self._extract_full_document(full_text, document_id)
        else:
            logger.info(f"Document too large ({len(full_text)} chars), using chunked extraction")
            return await self._extract_chunked(chunks, document_id)
    
    async def _extract_full_document(
        self,
        full_text: str,
        document_id: str
    ) -> Tuple[List[Entity], List[EntityRelationship]]:
        """Extract entities and relationships from full document in single pass"""
        
        logger.info("Extracting entities and relationships from full document context")
        
        # Use GPT-5-Thinking or GPT-4o for comprehensive extraction
        # Check document size and split if needed
        MAX_CHARS_PER_CALL = 12000  # Safe limit to avoid token limits
        
        if len(full_text) <= MAX_CHARS_PER_CALL:
            # Process entire document at once
            logger.info(f"Document size ({len(full_text)} chars) fits in single call")
            text_to_process = full_text
        else:
            # For now, just take the first section and log a warning
            logger.warning(f"Document too large ({len(full_text)} chars), processing first {MAX_CHARS_PER_CALL} chars")
            text_to_process = full_text[:MAX_CHARS_PER_CALL]
        
        prompt = """You are extracting a knowledge graph from this document. Your goal is to identify meaningful entities and their relationships.

IMPORTANT PRINCIPLES:
1. Only extract entities that contribute to the relationship network
2. Every entity should either have relationships OR be a critical standalone concept
3. Focus on quality over quantity - fewer, more meaningful entities
4. Extract relationships that span the entire document

CRITICAL DEDUPLICATION RULES:
- Each unique concept should appear ONLY ONCE - choose the most specific and appropriate type
- If a term could fit multiple types, pick its PRIMARY role in this document
- DO NOT extract the same entity with different types (e.g., "firmware" as both component AND concept)
- Prefer specific compound terms over generic base terms (e.g., "firmware update" instead of separate "firmware" and "update")
- Treat singular and plural as the same entity (e.g., "update" and "updates" are one entity)

CRITICAL ENTITY QUALITY RULES:
- Entities MUST be noun phrases (not verb phrases or actions)
- Entities MUST have meaning outside their sentence context
- Entities MUST be at least 3 characters (except known acronyms like CEO, API, USB)
- Entities MUST be referenceable concepts (not fragments from sentences)

Document:
{document}

Extract entities and their relationships following these guidelines:

ENTITIES TO EXTRACT (with type selection priority):
- Components, systems, and parts → Use COMPONENT for physical/software components
- Problems, errors, and issues → Use PROBLEM for specific issues (not generic "issue")
- Solutions, procedures, and methods → Use PROCEDURE for action steps
- People, organizations, and roles → Use PERSON/ORGANIZATION for specific names
- Key specifications and measurements → Use SPECIFICATION only for formal specs
- Important concepts and states → Use CONCEPT only if no other type fits

NEVER EXTRACT THESE AS ENTITIES:
❌ Verb phrases: "did not work", "causing it", "was broken", "has been fixed"
❌ Generic single words: "issue", "problem", "last", "update", "system", "process"
❌ Pronouns/fragments: "it", "this", "that", "them"
❌ Questions: "where are they coming from", "how does it work"
❌ Temporal references: "yesterday", "last week", "recently"
❌ Sentence fragments: Any phrase with more than 4 words
❌ Generic descriptors: "new", "old", "current", "previous"

ENTITIES TO SKIP:
- Trivial measurements mentioned once
- Generic terms without specific meaning
- Base terms when a compound term exists (skip "firmware" if "firmware update" exists)
- Entities with no relationships unless they're critical concepts
- Duplicate entities with different types
- Any phrase starting with a verb
- Single letters or meaningless abbreviations (unless domain-specific like "USB")

EXAMPLES OF WHAT NOT TO DO:
❌ BAD: Extract "firmware" as component, "firmware" as concept, and "firmware issue" as problem
✅ GOOD: Extract "firmware" as component and "firmware issue" as problem

❌ BAD: Extract both "update" and "updates" as separate entities
✅ GOOD: Extract only "firmware update" as procedure

RELATIONSHIPS TO EXTRACT:
- Technical: COMPONENT_OF, CONNECTS_TO, DEPENDS_ON, COMPATIBLE_WITH
- Causal: CAUSES, PREVENTS, RESOLVES, INDICATES, REQUIRES
- Process: FOLLOWS, PRECEDES, TRIGGERS, RESULTS_IN
- Organizational: RESPONSIBLE_FOR, MANAGES, OWNS, SERVES
- Documentation: DEFINES, DOCUMENTS, REFERENCES, SPECIFIES

Return a JSON object with:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "COMPONENT/PROBLEM/PROCEDURE/etc",
      "confidence": 0.95,
      "context": "brief description or context",
      "importance": "high/medium/low"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name", 
      "type": "relationship type",
      "confidence": 0.9,
      "evidence": "quote from document supporting this relationship"
    }}
  ]
}}

Focus on creating a connected knowledge graph where most entities have relationships.""".format(
            document=text_to_process
        )
        
        try:
            # Try GPT-5-Thinking first if available, otherwise GPT-4o
            response = await self.llm_service.call_with_fallback(
                prompt=prompt,
                primary_provider=LLMProvider.OPENAI,
                temperature=0.2,
                max_tokens=4000,
                model="gpt-4o"  # Will be updated to gpt-5-thinking when available
            )
            
            content = response.content.strip()
            
            # Handle empty response
            if not content:
                logger.warning("Empty response from LLM for full document extraction")
                logger.info("Falling back to chunked extraction")
                # Fall back to chunked extraction with parent_chunks which should be passed
                from app.models import Chunk
                # Get parent chunks from database
                from app.database import get_db
                from sqlalchemy import select
                async for db in get_db():
                    chunk_result = await db.execute(
                        select(Chunk).where(
                            Chunk.document_id == document_id,
                            Chunk.hierarchy_level == "parent"
                        ).order_by(Chunk.chunk_number)
                    )
                    parent_chunks = chunk_result.scalars().all()
                    return await self._extract_chunked(parent_chunks, document_id)
            
            # Handle markdown-wrapped JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content.strip())
            
            # Convert to Entity and EntityRelationship objects
            entities = []
            entity_name_to_id = {}
            
            for entity_data in result.get("entities", []):
                entity = Entity(
                    document_id=document_id,
                    name=entity_data["name"],
                    type=entity_data.get("type", "OTHER"),
                    entity_name=entity_data["name"],
                    entity_type=self._map_entity_type(entity_data.get("type", "OTHER")),
                    confidence=entity_data.get("confidence", 0.8),
                    metadata={
                        "context": entity_data.get("context", ""),
                        "importance": entity_data.get("importance", "medium"),
                        "extracted_from": "full_document",
                        "normalized_name": EntityDeduplicator.normalize_name(entity_data["name"])
                    }
                )
                entities.append(entity)
                entity_name_to_id[entity.entity_name.lower()] = entity.id
            
            # Create relationships
            relationships = []
            for rel_data in result.get("relationships", []):
                source_id = entity_name_to_id.get(rel_data["source"].lower())
                target_id = entity_name_to_id.get(rel_data["target"].lower())
                
                if source_id and target_id:
                    relationship = EntityRelationship(
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relationship_type=rel_data.get("type", "RELATES_TO"),
                        confidence_score=rel_data.get("confidence", 0.7),
                        metadata={
                            "evidence": rel_data.get("evidence", ""),
                            "extracted_from": "full_document"
                        }
                    )
                    relationships.append(relationship)
                else:
                    logger.warning(f"Skipping relationship - entity not found: {rel_data['source']} -> {rel_data['target']}")
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from full document")
            
            # Calculate relationship density
            if entities:
                avg_relationships = len(relationships) * 2 / len(entities)  # *2 because each relationship involves 2 entities
                logger.info(f"Average relationships per entity: {avg_relationships:.2f}")
            
            # Apply quality filtering as final step (same as chunked extraction)
            logger.info(f"Applying quality filter to {len(entities)} entities from full document extraction")
            entities = await self._filter_low_quality_entities(entities, relationships)
            
            # Also filter relationships to only include those with valid entities after filtering
            if relationships and entities:
                valid_entity_ids = {e.id for e in entities}
                relationships = [r for r in relationships 
                               if r.source_entity_id in valid_entity_ids and r.target_entity_id in valid_entity_ids]
                logger.info(f"After filtering relationships: {len(relationships)} relationships remain")
            
            # Apply entity deduplication
            entities, relationships = self._deduplicate_entities_and_relationships(entities, relationships)
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Full document extraction failed: {e}")
            logger.info("Falling back to chunked extraction")
            # Fall back to chunked extraction - need to get parent chunks
            from app.models import Chunk
            from app.database import get_db
            from sqlalchemy import select
            async for db in get_db():
                chunk_result = await db.execute(
                    select(Chunk).where(
                        Chunk.document_id == document_id,
                        Chunk.hierarchy_level == "parent"
                    ).order_by(Chunk.chunk_number)
                )
                parent_chunks = chunk_result.scalars().all()
                return await self._extract_chunked(parent_chunks, document_id)
    
    async def _extract_chunked(
        self,
        chunks: List[Chunk],
        document_id: str
    ) -> Tuple[List[Entity], List[EntityRelationship]]:
        """Original chunked extraction method (fallback)"""
        
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
                    # Check for duplicates using fuzzy matching
                    entity_name = entity_data["name"]
                    entity_type = entity_data["type"]
                    normalized_name = EntityDeduplicator.normalize_name(entity_name)
                    
                    # Look for existing similar entities
                    duplicate_found = False
                    for existing_key, existing_entity in entity_map.items():
                        existing_type = existing_key[1]
                        
                        # Only compare entities of the same type
                        if entity_type.lower() == existing_type.lower():
                            existing_name = existing_entity.entity_name
                            similarity = EntityDeduplicator.calculate_similarity(entity_name, existing_name)
                            
                            # If highly similar, merge the entities
                            if similarity >= 0.9:
                                duplicate_found = True
                                
                                # Update confidence if higher
                                if entity_data["confidence"] > existing_entity.confidence:
                                    existing_entity.confidence = entity_data["confidence"]
                                
                                # Merge metadata
                                existing_metadata = existing_entity.metadata or {}
                                
                                # Add context to all_contexts list
                                if 'all_contexts' not in existing_metadata:
                                    existing_metadata['all_contexts'] = []
                                    if existing_metadata.get('context'):
                                        existing_metadata['all_contexts'].append(existing_metadata['context'])
                                
                                new_context = entity_data.get("context", "")
                                if new_context and new_context not in existing_metadata['all_contexts']:
                                    existing_metadata['all_contexts'].append(new_context)
                                
                                # Add chunk_id to chunk_ids list
                                if 'chunk_ids' not in existing_metadata:
                                    existing_metadata['chunk_ids'] = []
                                    if existing_metadata.get('chunk_id'):
                                        existing_metadata['chunk_ids'].append(existing_metadata['chunk_id'])
                                
                                new_chunk_id = entity_data.get("chunk_id")
                                if new_chunk_id and new_chunk_id not in existing_metadata['chunk_ids']:
                                    existing_metadata['chunk_ids'].append(new_chunk_id)
                                
                                # Track original name variations
                                if 'original_names' not in existing_metadata:
                                    existing_metadata['original_names'] = [existing_entity.entity_name]
                                if entity_name not in existing_metadata['original_names']:
                                    existing_metadata['original_names'].append(entity_name)
                                
                                existing_entity.metadata = existing_metadata
                                
                                logger.debug(f"Merged duplicate entity: '{entity_name}' with '{existing_name}' (similarity: {similarity:.2f})")
                                break
                    
                    if not duplicate_found:
                        # Create new entity with normalized name stored
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
                                "attributes": self._extract_type_specific_attributes(entity_data),
                                "normalized_name": normalized_name
                            }
                        )
                        # Use normalized name in key for better deduplication
                        entity_key = (normalized_name, entity_type.lower())
                        entity_map[entity_key] = entity
                        all_entities.append(entity)
        
        # Run post-processing deduplication
        all_entities = await self._post_process_deduplication(all_entities, document_id)
        
        # Extract relationships between entities
        relationships = await self._extract_relationships(
            all_entities, chunks, document_id
        )
        
        logger.info(f"Extracted {len(all_entities)} unique entities and {len(relationships)} relationships")
        
        # Apply quality filtering as final step
        all_entities = await self._filter_low_quality_entities(all_entities, relationships)
        
        return all_entities, relationships
    
    async def _post_process_deduplication(
        self,
        entities: List[Entity],
        document_id: str
    ) -> List[Entity]:
        """
        Post-process entities to find and merge remaining duplicates.
        
        Args:
            entities: List of extracted entities
            document_id: Document ID
            
        Returns:
            Deduplicated list of entities
        """
        if len(entities) <= 1:
            return entities
        
        logger.info(f"Running post-processing deduplication on {len(entities)} entities")
        
        # Convert entities to dict format for deduplicator
        entity_dicts = []
        for entity in entities:
            entity_dict = {
                'id': entity.id,
                'entity_name': entity.entity_name,
                'entity_type': entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                'confidence_score': entity.confidence,
                'metadata': entity.metadata
            }
            entity_dicts.append(entity_dict)
        
        # Run deduplication with auto-merge for very similar entities (cross-type enabled)
        deduplicated_dicts, review_groups = EntityDeduplicator.deduplicate_entities(
            entity_dicts,
            auto_merge_threshold=0.95,  # Auto-merge if 95% similar
            review_threshold=0.85,      # Flag for review if 85% similar
            cross_type=True            # Enable cross-type deduplication
        )
        
        # Log review groups for manual inspection
        if review_groups:
            logger.info(f"Found {len(review_groups)} groups of potential duplicates for review:")
            for group in review_groups:
                names = [e.get('entity_name', '') for e in group]
                logger.info(f"  - Potential duplicates: {', '.join(names)}")
        
        # Convert back to Entity objects
        deduplicated_entities = []
        entity_id_map = {e.id: e for e in entities}  # Map for preserving original Entity objects
        
        for entity_dict in deduplicated_dicts:
            entity_id = entity_dict.get('id')
            
            if entity_id and entity_id in entity_id_map:
                # Use original entity object, but update metadata if changed
                original_entity = entity_id_map[entity_id]
                if 'metadata' in entity_dict:
                    original_entity.metadata = entity_dict['metadata']
                deduplicated_entities.append(original_entity)
            else:
                # This is a merged entity, create new Entity object
                entity = Entity(
                    document_id=document_id,
                    name=entity_dict.get('entity_name', ''),
                    type=entity_dict.get('entity_type', 'other'),
                    entity_name=entity_dict.get('entity_name', ''),
                    entity_type=self._map_entity_type(entity_dict.get('entity_type', 'other')),
                    confidence=entity_dict.get('confidence_score', 0.5),
                    metadata=entity_dict.get('metadata', {})
                )
                deduplicated_entities.append(entity)
        
        logger.info(f"Post-processing reduced entities from {len(entities)} to {len(deduplicated_entities)}")
        
        return deduplicated_entities
    
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
    
    async def _ensure_entity_coverage(
        self,
        entities: List[Entity],
        relationships: List[EntityRelationship]
    ) -> List[EntityRelationship]:
        """Ensure all entities have at least one relationship"""
        
        # Find entities with no relationships
        connected_entities = set()
        for rel in relationships:
            connected_entities.add(rel.source_entity_id)
            connected_entities.add(rel.target_entity_id)
        
        unconnected = [e for e in entities if e.id not in connected_entities]
        
        if not unconnected:
            return relationships
        
        logger.info(f"Found {len(unconnected)} unconnected entities, creating relationships")
        
        # For each unconnected entity, find its best match
        new_relationships = []
        for entity in unconnected:
            # Find most similar connected entity
            best_match = None
            best_score = 0
            
            for other in entities:
                if other.id == entity.id or other.id not in connected_entities:
                    continue
                
                # Calculate similarity based on type and context
                score = 0
                if entity.entity_type == other.entity_type:
                    score += 0.3
                
                # Check for contextual relationships
                if entity.metadata and other.metadata:
                    entity_context = str(entity.metadata.get('context', '')).lower()
                    other_context = str(other.metadata.get('context', '')).lower()
                    
                    # Check for related terms
                    if any(word in other_context for word in entity.entity_name.lower().split()):
                        score += 0.5
                    if any(word in entity_context for word in other.entity_name.lower().split()):
                        score += 0.5
                
                if score > best_score:
                    best_score = score
                    best_match = other
            
            # Create a relationship with the best match
            if best_match:
                rel_type = self._determine_relationship_type(entity, best_match)
                new_rel = EntityRelationship(
                    source_entity_id=entity.id,
                    target_entity_id=best_match.id,
                    relationship_type=rel_type,
                    confidence_score=0.5,  # Lower confidence for inferred relationships
                    metadata={
                        "inferred": True,
                        "reason": "Connected orphaned entity"
                    }
                )
                new_relationships.append(new_rel)
                logger.debug(f"Connected orphaned entity '{entity.entity_name}' to '{best_match.entity_name}' via {rel_type}")
        
        return relationships + new_relationships
    
    def _determine_relationship_type(self, entity1: Entity, entity2: Entity) -> str:
        """Determine appropriate relationship type based on entity types"""
        type1 = str(entity1.entity_type).lower()
        type2 = str(entity2.entity_type).lower()
        
        # Problem -> Procedure relationships
        if 'problem' in type1 and 'procedure' in type2:
            return "RESOLVED_BY"
        if 'procedure' in type1 and 'problem' in type2:
            return "RESOLVES"
        
        # Component relationships
        if 'component' in type1 and 'component' in type2:
            # Check if one is likely part of another based on names
            if entity1.entity_name.lower() in entity2.entity_name.lower():
                return "PART_OF"
            elif entity2.entity_name.lower() in entity1.entity_name.lower():
                return "CONTAINS"
            else:
                return "RELATES_TO"
        
        # Component -> Problem relationships
        if 'component' in type1 and 'problem' in type2:
            return "CAN_HAVE"
        if 'problem' in type1 and 'component' in type2:
            return "AFFECTS"
        
        # Procedure -> Component relationships
        if 'procedure' in type1 and 'component' in type2:
            return "APPLIES_TO"
        if 'component' in type1 and 'procedure' in type2:
            return "MAINTAINED_BY"
        
        # Default relationship
        return "RELATES_TO"
    
    async def _extract_relationships(
        self,
        entities: List[Entity],
        chunks: List[Chunk],
        document_id: str
    ) -> List[EntityRelationship]:
        """Extract relationships between entities using aggregated context"""
        
        relationships = []
        
        # Only extract relationships if we have multiple entities
        if len(entities) < 2:
            return relationships
        
        # Create entity name lookup
        entity_names = {e.entity_name.lower(): e for e in entities}
        
        # Step 1: Collect entity co-occurrences across ALL chunks
        logger.info(f"Analyzing {len(chunks)} chunks for entity co-occurrences")
        entity_cooccurrences = {}  # (entity1_id, entity2_id) -> list of chunk contexts
        
        for chunk in chunks:  # Process ALL chunks, not just first 10
            # Find entities mentioned in this chunk
            mentioned_entities = [
                e for e in entities
                if e.entity_name.lower() in chunk.chunk_text.lower()
            ]
            
            # Record co-occurrences
            if len(mentioned_entities) >= 2:
                for i, e1 in enumerate(mentioned_entities):
                    for e2 in mentioned_entities[i+1:]:
                        # Create ordered pair to avoid duplicates
                        pair = tuple(sorted([e1.id, e2.id]))
                        if pair not in entity_cooccurrences:
                            entity_cooccurrences[pair] = []
                        entity_cooccurrences[pair].append({
                            'chunk_id': chunk.id,
                            'chunk_text': chunk.chunk_text[:500],  # Limit context size
                            'entities': [e1, e2]
                        })
        
        logger.info(f"Found {len(entity_cooccurrences)} entity pairs with co-occurrences")
        
        # Step 2: Process co-occurrences in batches for efficiency
        if entity_cooccurrences:
            # Batch process the most significant co-occurrences
            sorted_pairs = sorted(entity_cooccurrences.items(), 
                                key=lambda x: len(x[1]), reverse=True)[:50]  # Top 50 pairs
            
            for pair_key, contexts in sorted_pairs:
                entity1 = next(e for e in entities if e.id == pair_key[0])
                entity2 = next(e for e in entities if e.id == pair_key[1])
                
                # Aggregate context from all co-occurrences
                combined_context = "\n---\n".join([c['chunk_text'] for c in contexts[:3]])  # Use up to 3 contexts
                
                # Get relationships for this pair
                pair_relationships = await self._identify_relationships_batched(
                    combined_context,
                    [entity1, entity2],
                    num_contexts=len(contexts)
                )
                
                for rel in pair_relationships:
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
                                "evidence": rel.get("evidence", ""),
                                "num_occurrences": len(contexts),
                                "chunk_ids": [c['chunk_id'] for c in contexts[:3]]
                            }
                        )
                        relationships.append(relationship)
        
        # Step 3: Extract global relationships from full document context
        logger.info("Extracting global relationships from full document context")
        global_relationships = await self._extract_global_relationships(
            entities, chunks, document_id
        )
        relationships.extend(global_relationships)
        
        # Step 4: Deduplicate and merge relationships
        merged_relationships = self._merge_duplicate_relationships(relationships)
        
        # Step 5: Validate relationships
        validated_relationships = self._validate_relationships(merged_relationships, entities)
        
        # Step 6: Ensure all entities have at least one relationship
        complete_relationships = await self._ensure_entity_coverage(entities, validated_relationships)
        
        logger.info(f"Extracted {len(complete_relationships)} relationships after validation and coverage")
        logger.info(f"Relationship coverage improved: {len(validated_relationships)} -> {len(complete_relationships)}")
        return complete_relationships
    
    async def _identify_relationships_batched(
        self,
        combined_context: str,
        entities: List[Entity],
        num_contexts: int = 1
    ) -> List[Dict]:
        """Use LLM to identify relationships with aggregated context"""
        
        entity_names = [e.entity_name for e in entities]
        entity_types = {e.entity_name: e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type) for e in entities}
        
        prompt = f"""You are analyzing multiple contexts where these entities appear together.
Based on the aggregated evidence, identify the relationships between them.

Entities to analyze:
{chr(10).join([f"- {name} (type: {entity_types.get(name, 'unknown')})" for name in entity_names])}

Combined context from {num_contexts} occurrences:
{combined_context}

CRITICAL RULES FOR RELATIONSHIP DIRECTION:
1. Solutions/Procedures RESOLVE Problems (not vice versa)
   ✓ "factory reset" RESOLVES "black screen"
   ✗ "black screen" RESOLVES "factory reset"

2. Causes CREATE Problems/Effects (not vice versa)
   ✓ "firmware update" CAUSES "data corruption"
   ✗ "data corruption" CAUSES "firmware update"

3. Symptoms INDICATE Root Causes (not vice versa)
   ✓ "error message" INDICATES "configuration issue"
   ✗ "configuration issue" INDICATES "error message"

4. Components are PART_OF Systems (not vice versa)
   ✓ "display connector" COMPONENT_OF "display assembly"
   ✗ "display assembly" COMPONENT_OF "display connector"

AVAILABLE RELATIONSHIP TYPES:
- CAUSES: A directly causes B to occur
- RESOLVES: A fixes or addresses problem B
- PREVENTS: A stops B from occurring  
- INDICATES: A is a symptom/sign of B
- COMPONENT_OF: A is part of system B
- DEPENDS_ON: A requires B to function
- RELATES_TO: General association

Return a JSON array with relationships. For each relationship provide:
- source: entity that performs the action/is the subject
- target: entity that receives the action/is the object
- type: relationship type from above
- confidence: 0.0-1.0 based on evidence strength
- evidence: brief quote supporting this relationship

Only return relationships with strong evidence from the text.
Focus on the most important relationships that help understand the document."""
        
        try:
            response = await self.llm_service.call_with_fallback(
                prompt=prompt,
                primary_provider=LLMProvider.OPENAI,
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            relationships = json.loads(content.strip())
            return relationships if isinstance(relationships, list) else []
            
        except Exception as e:
            logger.error(f"Failed to identify batched relationships: {e}")
            return []
    
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
    
    async def _extract_global_relationships(
        self,
        entities: List[Entity],
        chunks: List[Chunk],
        document_id: str
    ) -> List[EntityRelationship]:
        """Extract relationships considering full document context"""
        
        if len(entities) < 2:
            return []
        
        # Create a summary of the document structure
        doc_summary = self._create_document_summary(chunks)
        
        # Create entity context summary
        entity_summary = self._create_entity_summary(entities)
        
        prompt = f"""Analyze this document holistically to identify important relationships between entities.

DOCUMENT SUMMARY:
{doc_summary}

ENTITIES IN DOCUMENT:
{entity_summary}

Identify the KEY relationships that explain:
1. Causal chains (what causes what)
2. Problem-solution pairs
3. Component hierarchies
4. Dependencies and requirements
5. Temporal sequences

CRITICAL: Follow these relationship direction rules:
- Problems are RESOLVED_BY solutions (not vice versa)
- Causes LEAD_TO effects (not vice versa)
- Systems CONTAIN components (not vice versa)
- Later events FOLLOW earlier events (not vice versa)

Return a JSON array of the most important relationships with:
- source: entity name (exact match)
- target: entity name (exact match)
- type: CAUSES, RESOLVES, COMPONENT_OF, DEPENDS_ON, PREVENTS, INDICATES, FOLLOWS, RELATES_TO
- confidence: 0.0-1.0
- reasoning: brief explanation of the relationship

Focus on relationships that span across different parts of the document."""
        
        try:
            response = await self.llm_service.call_with_fallback(
                prompt=prompt,
                primary_provider=LLMProvider.OPENAI,
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            rel_data = json.loads(content.strip())
            
            # Convert to EntityRelationship objects
            relationships = []
            entity_lookup = {e.entity_name.lower(): e for e in entities}
            
            for rel in rel_data:
                source = entity_lookup.get(rel["source"].lower())
                target = entity_lookup.get(rel["target"].lower())
                
                if source and target and source.id != target.id:
                    relationship = EntityRelationship(
                        source_entity_id=source.id,
                        target_entity_id=target.id,
                        relationship_type=rel.get("type", "RELATES_TO"),
                        confidence_score=rel.get("confidence", 0.6),
                        metadata={
                            "reasoning": rel.get("reasoning", ""),
                            "extracted_from": "global_analysis",
                            "document_id": document_id
                        }
                    )
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to extract global relationships: {e}")
            return []
    
    def _deduplicate_entities_and_relationships(
        self, 
        entities: List[Entity], 
        relationships: List[EntityRelationship]
    ) -> Tuple[List[Entity], List[EntityRelationship]]:
        """Deduplicate entities and update relationships"""
        
        if not entities:
            return entities, relationships
        
        logger.info(f"Starting deduplication for {len(entities)} entities")
        
        # Convert entities to dict format for deduplicator
        entity_dicts = []
        for entity in entities:
            entity_dict = {
                'id': entity.id,
                'entity_name': entity.entity_name,
                'entity_type': str(entity.entity_type),
                'confidence': entity.confidence,
                'metadata': entity.metadata or {}
            }
            entity_dicts.append(entity_dict)
        
        # Find duplicates with cross-type matching enabled
        deduplicator = EntityDeduplicator()
        duplicate_groups = deduplicator.find_duplicates(
            entity_dicts, 
            threshold=0.80,  # Lower threshold to catch more variations
            cross_type=True  # Allow matching across types (e.g., "firmware issue" problem vs "firmware" component)
        )
        
        if not duplicate_groups:
            logger.info("No duplicate entities found")
            return entities, relationships
        
        logger.info(f"Found {len(duplicate_groups)} groups of duplicate entities")
        
        # Create entity mapping (old ID -> new ID)
        entity_mapping = {}
        entities_to_keep = []
        processed_ids = set()
        
        for group in duplicate_groups:
            # Merge the group
            merged_dict = deduplicator.merge_entity_data(group)
            
            # Find the corresponding Entity object to keep
            primary_id = group[0]['id']  # Use first entity's ID as primary
            primary_entity = next((e for e in entities if e.id == primary_id), None)
            
            if primary_entity:
                # Update the primary entity with merged data
                primary_entity.entity_name = merged_dict.get('entity_name', primary_entity.entity_name)
                primary_entity.confidence = merged_dict.get('confidence_score', merged_dict.get('confidence', primary_entity.confidence))
                primary_entity.metadata = merged_dict.get('metadata', primary_entity.metadata)
                
                entities_to_keep.append(primary_entity)
                
                # Map all duplicate IDs to the primary ID
                for entity_dict in group:
                    entity_mapping[entity_dict['id']] = primary_id
                    processed_ids.add(entity_dict['id'])
        
        # Add non-duplicate entities
        for entity in entities:
            if entity.id not in processed_ids:
                entities_to_keep.append(entity)
        
        # Update relationships with new entity IDs
        updated_relationships = []
        seen_relationships = set()
        
        for rel in relationships:
            # Map to deduplicated IDs
            source_id = entity_mapping.get(rel.source_entity_id, rel.source_entity_id)
            target_id = entity_mapping.get(rel.target_entity_id, rel.target_entity_id)
            
            # Skip self-relationships
            if source_id == target_id:
                continue
            
            # Skip duplicate relationships
            rel_signature = (source_id, target_id, rel.relationship_type)
            if rel_signature in seen_relationships:
                continue
            seen_relationships.add(rel_signature)
            
            # Create updated relationship
            updated_rel = EntityRelationship(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=rel.relationship_type,
                confidence_score=rel.confidence_score,
                metadata=rel.metadata
            )
            updated_relationships.append(updated_rel)
        
        logger.info(f"Deduplication complete: {len(entities)} -> {len(entities_to_keep)} entities")
        logger.info(f"Relationships updated: {len(relationships)} -> {len(updated_relationships)}")
        
        return entities_to_keep, updated_relationships
    
    def _create_document_summary(self, chunks: List[Chunk]) -> str:
        """Create a brief summary of document structure"""
        if not chunks:
            return "No content available"
        
        # Take first and last chunks for context
        intro = chunks[0].chunk_text[:300] if chunks else ""
        conclusion = chunks[-1].chunk_text[:300] if len(chunks) > 1 else ""
        
        return f"""Document Introduction:
{intro}...

Document Conclusion:
...{conclusion}

Total sections: {len(chunks)}"""
    
    def _create_entity_summary(self, entities: List[Entity]) -> str:
        """Create a summary of entities by type"""
        by_type = {}
        for entity in entities:
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity.entity_name)
        
        summary_lines = []
        for entity_type, names in by_type.items():
            summary_lines.append(f"{entity_type}: {', '.join(names[:10])}")
        
        return "\n".join(summary_lines)
    
    def _merge_duplicate_relationships(self, relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Merge duplicate relationships and average confidence scores"""
        merged = {}
        
        for rel in relationships:
            key = (rel.source_entity_id, rel.target_entity_id, rel.relationship_type)
            
            if key not in merged:
                merged[key] = rel
            else:
                # Merge confidence scores and metadata
                existing = merged[key]
                
                # Average confidence scores
                existing.confidence_score = (existing.confidence_score + rel.confidence_score) / 2
                
                # Merge metadata
                if rel.metadata:
                    if not existing.metadata:
                        existing.metadata = {}
                    
                    # Combine evidence/context
                    if "evidence" in rel.metadata and rel.metadata["evidence"]:
                        existing.metadata["evidence"] = existing.metadata.get("evidence", "") + " | " + rel.metadata["evidence"]
                    
                    # Track all chunk IDs
                    if "chunk_ids" in rel.metadata:
                        existing_chunks = existing.metadata.get("chunk_ids", [])
                        new_chunks = rel.metadata["chunk_ids"]
                        existing.metadata["chunk_ids"] = list(set(existing_chunks + new_chunks))
                    
                    # Update occurrence count
                    existing.metadata["num_occurrences"] = existing.metadata.get("num_occurrences", 1) + rel.metadata.get("num_occurrences", 1)
        
        return list(merged.values())
    
    def _validate_relationships(self, relationships: List[EntityRelationship], entities: List[Entity]) -> List[EntityRelationship]:
        """Validate and filter relationships based on rules"""
        validated = []
        entity_types = {e.id: (e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)) for e in entities}
        entity_names = {e.id: e.entity_name for e in entities}
        
        for rel in relationships:
            # Skip self-relationships
            if rel.source_entity_id == rel.target_entity_id:
                logger.debug(f"Skipping self-relationship for {entity_names.get(rel.source_entity_id)}")
                continue
            
            # Skip low confidence relationships
            if rel.confidence_score < 0.3:
                logger.debug(f"Skipping low confidence relationship: {rel.confidence_score}")
                continue
            
            # Validate relationship direction based on entity types
            source_type = entity_types.get(rel.source_entity_id, "").upper()
            target_type = entity_types.get(rel.target_entity_id, "").upper()
            rel_type = rel.relationship_type.upper()
            
            # Check for invalid patterns
            invalid = False
            
            # Problems don't resolve procedures
            if source_type == "PROBLEM" and target_type == "PROCEDURE" and rel_type == "RESOLVES":
                logger.debug(f"Fixing: Problem cannot resolve procedure")
                # Swap source and target
                rel.source_entity_id, rel.target_entity_id = rel.target_entity_id, rel.source_entity_id
            
            # Problems don't cause solutions
            elif source_type == "PROBLEM" and target_type in ["PROCEDURE", "SOLUTION"] and rel_type == "CAUSES":
                logger.debug(f"Fixing: Problem cannot cause solution")
                # Change relationship type
                rel.relationship_type = "RESOLVED_BY"
                # Swap source and target
                rel.source_entity_id, rel.target_entity_id = rel.target_entity_id, rel.source_entity_id
            
            # Components aren't part of problems
            elif source_type == "COMPONENT" and target_type == "PROBLEM" and rel_type == "COMPONENT_OF":
                logger.debug(f"Skipping: Component cannot be part of problem")
                invalid = True
            
            if not invalid:
                validated.append(rel)
        
        logger.info(f"Validated {len(validated)} out of {len(relationships)} relationships")
        return validated
    
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
            "TOOL": EntityType.TOOL,
            "MATERIAL": EntityType.MATERIAL,
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
    
    async def _filter_low_quality_entities(
        self,
        entities: List[Entity],
        relationships: List[EntityRelationship]
    ) -> List[Entity]:
        """
        Filter out low-quality entities using the quality validator.
        
        Args:
            entities: List of entities to filter
            relationships: List of relationships for context
            
        Returns:
            Filtered list of high-quality entities
        """
        if not entities:
            return entities
        
        # Convert to dictionary format for validator
        entity_dicts = []
        for entity in entities:
            entity_dict = {
                'id': entity.id,
                'entity_name': entity.entity_name,
                'entity_type': entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                'confidence': entity.confidence,
                'metadata': entity.metadata
            }
            entity_dicts.append(entity_dict)
        
        # Convert relationships to dictionary format
        relationship_dicts = []
        for rel in relationships:
            rel_dict = {
                'source_entity_id': rel.source_entity_id,
                'target_entity_id': rel.target_entity_id,
                'relationship_type': rel.relationship_type
            }
            relationship_dicts.append(rel_dict)
        
        # Filter entities
        kept_entity_dicts, filtered_entity_dicts = EntityQualityValidator.filter_entities(
            entity_dicts,
            relationship_dicts,
            min_quality_score=0.4,
            preserve_with_relationships=3
        )
        
        # Log filtered entities
        if filtered_entity_dicts:
            logger.info(f"Quality filter removed {len(filtered_entity_dicts)} low-quality entities:")
            for entity_dict in filtered_entity_dicts[:10]:  # Log first 10
                logger.info(f"  - {entity_dict['entity_name']} ({entity_dict['entity_type']}): {entity_dict.get('quality_reason', 'low_quality')}")
        
        # Keep only the high-quality entities and update their metadata with quality scores
        kept_entity_ids = {e['id'] for e in kept_entity_dicts}
        
        # Create a map of entity id to quality info
        quality_info = {e['id']: {'quality_score': e['quality_score'], 'quality_reason': e['quality_reason']} 
                        for e in kept_entity_dicts}
        
        # Filter and update entities
        filtered_entities = []
        for entity in entities:
            if entity.id in kept_entity_ids:
                # Update metadata with quality info
                if entity.metadata is None:
                    entity.metadata = {}
                entity.metadata.update(quality_info[entity.id])
                filtered_entities.append(entity)
        
        logger.info(f"Quality filtering: {len(entities)} -> {len(filtered_entities)} entities")
        return filtered_entities
