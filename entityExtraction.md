# Entity Extraction Pipeline - Architectural Redesign

## Executive Summary

The current entity extraction pipeline conflates entity mentions with canonical entities, lacks versioning and caching, and maintains dual sources of truth (Supabase + Neo4j). This document presents a complete architectural redesign that separates concerns, introduces proper data modeling, and implements efficient caching and orchestration.

## Implementation Status (2025-09-01)

### Completed
- Legacy extractor archived and gated:
  - `app/processors/entity_extractor.py` marked LEGACY; disabled by default and only runs with `COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1`.
  - `docs/legacy/entity_extractor_legacy.md` created for reference.
- v2 schema added (non-destructive):
  - Tables: `extraction_runs`, `canonical_entities`, `entity_mentions`, `canonical_relationships`, `graph_projection_outbox`.
  - Order fixed so `canonical_entities` is created before `entity_mentions` (FK requirement).
  - `pg_trgm` enabled + indexes for fuzzy name/text search.
  - Migration file: `migrations/2025-09-01_entity_pipeline_v2.sql`.
- v2 typed models introduced:
  - `app/models/entity_v2.py` with `ExtractionRun`, `EntityMention`, `CanonicalEntity`, `CanonicalRelationship`.
- v2 extraction integrated into the pipeline (default):
  - Celery task `extract_entities` now runs v2 by default (controlled by `ENTITY_PIPELINE_VERSION`, default `v2`).
  - Runner flow: `app/flows/entity_extraction_runner_v2.py` uses CocoIndex `ExtractByLlm` to return typed mentions with offsets.
  - Persists `entity_mentions` with `canonical_entity_id` set after naive canonical upsert.
  - Records an `extraction_runs` row with counters and timing.
- Flags and docs:
  - Config flags in `app/config.py`: `ENTITY_PIPELINE_VERSION`, `COCOINDEX_LEGACY_ENTITY_EXTRACTOR`.
  - Migration guide: `docs/entity_pipeline_migration.md`.

### Next Up
- Neo4j projection worker:
  - Consume `graph_projection_outbox` and MERGE nodes/edges idempotently with a `projection_version`/`run_id` property.
- Relationship inference v2:
  - Infer document-wide relationships after canonicalization; persist to `canonical_relationships`.
  - Optional: add `mention_relationships` for provenance if needed.
- Governance and validation:
  - Add enums/reference tables for `entity_type` and `relationship_type` plus a type-compatibility matrix; enforce via DB constraints and code.
- Idempotency and caching:
  - Populate `input_hash` for `extraction_runs`, use advisory locks per document, and short-circuit identical re-runs.
- Observability and QA:
  - Metrics (counts, fragment rate, token/cost), structured logging, and golden tests on known documents.
- API/UI integration:
  - Read endpoints and review UI for mentions/canonical (merge suggestions, invalidation), plus graph preview before approval.
- Flow alignment (optional):
  - Expand `app/flows/entity_extraction_flow_v2.py` to export directly to Postgres/Neo4j targets if desired; current runner uses service writes which is acceptable short-term.

### How To Run Now
- Apply the migration in `migrations/2025-09-01_entity_pipeline_v2.sql` (ensures `pg_trgm`, tables, and indexes).
- Ensure `OPENAI_API_KEY` is set in `.env`.
- Trigger document processing; v2 will run by default. Legacy can be used only by setting `ENTITY_PIPELINE_VERSION=v1` and `COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1`.

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Architectural Vision](#architectural-vision)
3. [New Data Model](#new-data-model)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Implementation Components](#implementation-components)
6. [Migration Strategy](#migration-strategy)
7. [Success Metrics](#success-metrics)

## Current State Analysis

### Critical Issues

1. **Data Model Conflation**: The `entities` table mixes occurrences (mentions in text) with concepts (canonical entities), making deduplication a constant battle
2. **No Pipeline Versioning**: Can't track what changed between extraction runs or cache results
3. **Dual Source of Truth**: Supabase and Neo4j can drift without explicit sync guarantees
4. **Synchronous Processing**: API-driven extraction limits scalability and observability
5. **Poor Entity Quality**: Fragments like "last", "causing it", "did not work" pollute the graph
6. **Weak Relationships**: Batch-limited extraction produces arbitrary connections

### Current Database Schema Issues

```sql
-- Current problematic design
entities (
    entity_name TEXT,  -- Mixes "firmware update" mention #1, #2, #3
    entity_type TEXT,  -- All stored as separate rows
    document_id UUID   -- No way to track they're the same concept
)
```

## Architectural Vision

### Core Principles

1. **Separation of Concerns**: Entity mentions ≠ Canonical entities
2. **Single Source of Truth**: Supabase as primary, Neo4j as read-only projection
3. **Versioned & Cacheable**: Every extraction run is versioned and cacheable
4. **Observable Pipeline**: Background jobs with explicit steps and metrics
5. **Quality First**: Better to have fewer high-quality entities than many poor ones

### High-Level Architecture

```
Documents → Extract Mentions → Canonicalize → Infer Relationships → Project to Neo4j
                ↓                    ↓                ↓                      ↓
            [Cache Layer]      [Dedup Service]  [Validation]        [Sync Queue]
```

## New Data Model

### Core Tables

#### 1. Entity Mentions Table
Every occurrence of an entity in the text:

```sql
CREATE TABLE entity_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id),
    extraction_run_id UUID NOT NULL REFERENCES extraction_runs(id),
    
    -- Mention details
    text TEXT NOT NULL,                    -- "firmware update"
    type TEXT NOT NULL,                    -- "procedure"
    start_offset INTEGER NOT NULL,         -- 234
    end_offset INTEGER NOT NULL,           -- 249
    confidence FLOAT DEFAULT 0.5,
    
    -- Canonicalization
    canonical_entity_id UUID REFERENCES canonical_entities(id),
    canonicalization_score FLOAT,          -- Similarity to canonical
    
    -- Metadata
    context TEXT,                          -- Surrounding text
    attributes JSONB DEFAULT '{}',
    
    -- Tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Uniqueness constraint
    UNIQUE(document_id, chunk_id, start_offset, end_offset, extraction_run_id)
);

-- Indexes for performance
CREATE INDEX idx_mentions_document ON entity_mentions(document_id);
CREATE INDEX idx_mentions_canonical ON entity_mentions(canonical_entity_id);
CREATE INDEX idx_mentions_type ON entity_mentions(type);
CREATE INDEX idx_mentions_text_trgm ON entity_mentions USING gin(text gin_trgm_ops);
```

#### 2. Canonical Entities Table
The deduplicated, canonical version of entities:

```sql
CREATE TABLE canonical_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Canonical form
    name TEXT NOT NULL,                    -- "Firmware Update"
    type TEXT NOT NULL,                    -- "procedure"
    aliases TEXT[] DEFAULT '{}',           -- ["firmware updates", "FW update", "firmware upgrade"]
    
    -- Statistics
    mention_count INTEGER DEFAULT 0,       -- Total mentions across documents
    document_count INTEGER DEFAULT 0,      -- Documents containing this entity
    relationship_count INTEGER DEFAULT 0,  -- Number of relationships
    
    -- Quality metrics
    quality_score FLOAT DEFAULT 0.5,       -- Based on frequency, relationships, validation
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by UUID,
    validated_at TIMESTAMPTZ,
    
    -- Embedding for similarity
    embedding VECTOR(1536),                -- For semantic similarity
    
    -- Metadata
    definition TEXT,                       -- Human-provided or LLM-generated definition
    category TEXT,                         -- High-level categorization
    attributes JSONB DEFAULT '{}',
    
    -- Tracking
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Uniqueness
    UNIQUE(name, type)
);

-- Indexes
CREATE INDEX idx_canonical_name_trgm ON canonical_entities USING gin(name gin_trgm_ops);
CREATE INDEX idx_canonical_type ON canonical_entities(type);
CREATE INDEX idx_canonical_quality ON canonical_entities(quality_score DESC);
CREATE INDEX idx_canonical_embedding ON canonical_entities USING ivfflat(embedding vector_cosine_ops);
```

#### 3. Extraction Runs Table
Track every extraction with versioning:

```sql
CREATE TABLE extraction_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id),
    
    -- Versioning
    pipeline_version TEXT NOT NULL,        -- "v2.1.0"
    prompt_version TEXT NOT NULL,          -- "2024-01-15-v3"
    model TEXT NOT NULL,                   -- "gpt-4-turbo"
    temperature FLOAT DEFAULT 0.1,
    
    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    
    -- Metrics
    chunks_processed INTEGER DEFAULT 0,
    mentions_extracted INTEGER DEFAULT 0,
    entities_canonicalized INTEGER DEFAULT 0,
    relationships_inferred INTEGER DEFAULT 0,
    
    -- Cost tracking
    total_tokens INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 4) DEFAULT 0,
    
    -- Cache stats
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    
    -- Metadata
    parameters JSONB DEFAULT '{}',         -- Run-specific parameters
    metrics JSONB DEFAULT '{}',            -- Performance metrics
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate runs
    UNIQUE(document_id, pipeline_version, prompt_version, model)
);

CREATE INDEX idx_runs_document ON extraction_runs(document_id);
CREATE INDEX idx_runs_status ON extraction_runs(status);
CREATE INDEX idx_runs_created ON extraction_runs(created_at DESC);
```

#### 4. Canonical Relationships Table
Relationships at the canonical level:

```sql
CREATE TABLE canonical_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Canonical entities
    source_entity_id UUID NOT NULL REFERENCES canonical_entities(id),
    target_entity_id UUID NOT NULL REFERENCES canonical_entities(id),
    relationship_type TEXT NOT NULL,
    
    -- Validation
    confidence FLOAT DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,      -- Number of mentions supporting this
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by UUID,
    validated_at TIMESTAMPTZ,
    
    -- Provenance
    first_seen_run_id UUID REFERENCES extraction_runs(id),
    last_seen_run_id UUID REFERENCES extraction_runs(id),
    
    -- Metadata
    attributes JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicates
    UNIQUE(source_entity_id, target_entity_id, relationship_type),
    -- Prevent self-relationships
    CHECK(source_entity_id != target_entity_id)
);

CREATE INDEX idx_canonical_rel_source ON canonical_relationships(source_entity_id);
CREATE INDEX idx_canonical_rel_target ON canonical_relationships(target_entity_id);
CREATE INDEX idx_canonical_rel_type ON canonical_relationships(relationship_type);
```

#### 5. Extraction Cache Table
Cache LLM responses for efficiency:

```sql
CREATE TABLE extraction_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Cache key components
    content_hash TEXT NOT NULL,            -- SHA256 of input text
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    temperature FLOAT NOT NULL,
    
    -- Cached response
    response JSONB NOT NULL,
    
    -- Metadata
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '7 days',
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique cache key
    UNIQUE(content_hash, model, prompt_version, temperature)
);

CREATE INDEX idx_cache_key ON extraction_cache(content_hash, model, prompt_version);
CREATE INDEX idx_cache_expires ON extraction_cache(expires_at);
```

#### 6. Entity Type Constraints
Define allowed entity types and relationship patterns:

```sql
CREATE TABLE entity_types (
    type TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT,
    validation_rules JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO entity_types (type, category, description) VALUES
    ('person', 'entity', 'Individual person with a name'),
    ('organization', 'entity', 'Company, institution, or agency'),
    ('location', 'entity', 'Physical or geographical location'),
    ('date', 'temporal', 'Specific date or time period'),
    ('product', 'entity', 'Commercial product or service'),
    ('component', 'technical', 'Part of a larger system'),
    ('procedure', 'action', 'Method, process, or technique'),
    ('problem', 'issue', 'Issue, failure, or error'),
    ('technology', 'technical', 'Platform, framework, or standard'),
    ('measurement', 'value', 'Quantity with units');

CREATE TABLE relationship_types (
    type TEXT PRIMARY KEY,
    source_types TEXT[] NOT NULL,          -- Allowed source entity types
    target_types TEXT[] NOT NULL,          -- Allowed target entity types
    is_directional BOOLEAN DEFAULT TRUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO relationship_types (type, source_types, target_types, description) VALUES
    ('RESOLVES', ARRAY['procedure'], ARRAY['problem'], 'Procedure fixes problem'),
    ('CAUSES', ARRAY['component', 'event'], ARRAY['problem'], 'Source causes problem'),
    ('PART_OF', ARRAY['component'], ARRAY['system', 'product'], 'Component part of system'),
    ('DEPENDS_ON', ARRAY['component', 'procedure'], ARRAY['component', 'technology'], 'Requires to function'),
    ('INDICATES', ARRAY['measurement', 'problem'], ARRAY['problem', 'condition'], 'Symptom of issue');
```

### Compatibility Views
For backward compatibility during migration:

```sql
-- View to mimic old entities table
CREATE VIEW entities_compat AS
SELECT 
    em.id,
    em.document_id,
    ce.name as entity_name,
    ce.name as name,  -- Support both field names
    ce.type as entity_type,
    ce.type as type,   -- Support both field names
    em.confidence as confidence_score,
    em.chunk_id as source_chunk_id,
    em.extraction_run_id,
    ce.is_validated as is_verified,
    ce.validated_by as verified_by,
    ce.validated_at as verified_at,
    em.canonical_entity_id,
    jsonb_build_object(
        'context', em.context,
        'attributes', em.attributes,
        'quality_score', ce.quality_score
    ) as metadata,
    em.created_at,
    em.updated_at
FROM entity_mentions em
LEFT JOIN canonical_entities ce ON em.canonical_entity_id = ce.id;

-- View for relationships
CREATE VIEW entity_relationships_compat AS
SELECT 
    cr.id,
    cr.source_entity_id,
    cr.target_entity_id,
    cr.relationship_type,
    cr.confidence as confidence_score,
    cr.is_validated as is_verified,
    cr.validated_by as verified_by,
    cr.validated_at as verified_at,
    cr.attributes as metadata,
    cr.created_at,
    cr.updated_at
FROM canonical_relationships cr;
```

## Pipeline Architecture

### Extraction Pipeline

```python
class ExtractionPipeline:
    """
    Orchestrates the complete extraction pipeline with caching,
    versioning, and quality checks at each step.
    """
    
    def __init__(self):
        self.cache = ExtractionCache()
        self.canonicalizer = EntityCanonicalizer()
        self.relationship_extractor = RelationshipExtractor()
        self.neo4j_projector = Neo4jProjector()
        
    async def run(self, document_id: str, force_refresh: bool = False) -> ExtractionRun:
        """
        Execute complete extraction pipeline for a document.
        
        Args:
            document_id: Document to process
            force_refresh: Bypass cache if True
            
        Returns:
            ExtractionRun record with metrics
        """
        # Create extraction run record
        run = await self.create_extraction_run(document_id)
        
        try:
            # Step 1: Extract mentions with caching
            mentions = await self.extract_mentions(document_id, run.id, force_refresh)
            
            # Step 2: Canonicalize mentions
            canonical_mappings = await self.canonicalize_mentions(mentions, run.id)
            
            # Step 3: Infer relationships at canonical level
            relationships = await self.infer_relationships(
                canonical_mappings, document_id, run.id
            )
            
            # Step 4: Update quality scores
            await self.update_quality_scores(canonical_mappings, relationships)
            
            # Step 5: Project to Neo4j
            await self.project_to_neo4j(canonical_mappings, relationships, run.id)
            
            # Update run status
            await self.complete_extraction_run(run, "completed")
            
        except Exception as e:
            await self.complete_extraction_run(run, "failed", str(e))
            raise
            
        return run
```

### Mention Extraction

```python
class MentionExtractor:
    """
    Extracts entity mentions from text with offsets and context.
    Uses function calling for structured output with caching.
    """
    
    async def extract_mentions(
        self, 
        chunk: Chunk, 
        run_id: str,
        force_refresh: bool = False
    ) -> List[EntityMention]:
        """
        Extract entity mentions from a text chunk.
        """
        # Generate cache key
        cache_key = self.generate_cache_key(chunk.text)
        
        # Check cache unless force refresh
        if not force_refresh:
            cached = await self.cache.get(cache_key)
            if cached:
                return self.parse_cached_mentions(cached, chunk, run_id)
        
        # Prepare function calling schema
        schema = {
            "name": "extract_entities",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"type": "string", "enum": ALLOWED_ENTITY_TYPES},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["text", "type", "start", "end"]
                        }
                    }
                },
                "required": ["entities"]
            }
        }
        
        # Call LLM with function calling
        response = await self.llm.call_with_function(
            prompt=self.build_extraction_prompt(chunk.text),
            functions=[schema],
            temperature=0.1
        )
        
        # Cache response
        await self.cache.set(cache_key, response)
        
        # Parse and validate mentions
        mentions = self.parse_mentions(response, chunk, run_id)
        
        # Apply quality filters
        mentions = self.filter_low_quality_mentions(mentions)
        
        return mentions
    
    def filter_low_quality_mentions(self, mentions: List[EntityMention]) -> List[EntityMention]:
        """
        Filter out poor quality entity mentions.
        """
        filtered = []
        
        for mention in mentions:
            # Skip fragments and stopwords
            if self.is_fragment(mention.text):
                continue
                
            # Skip generic terms based on type
            if self.is_generic_for_type(mention.text, mention.type):
                continue
                
            # Skip low confidence
            if mention.confidence < 0.3:
                continue
                
            filtered.append(mention)
            
        return filtered
```

### Entity Canonicalization

```python
class EntityCanonicalizer:
    """
    Maps entity mentions to canonical entities using embeddings,
    fuzzy matching, and type-specific rules.
    """
    
    async def canonicalize_mentions(
        self,
        mentions: List[EntityMention],
        run_id: str
    ) -> Dict[str, str]:
        """
        Map mentions to canonical entities.
        
        Returns:
            Dict mapping mention_id -> canonical_entity_id
        """
        mappings = {}
        
        # Group mentions by type for efficient processing
        mentions_by_type = self.group_by_type(mentions)
        
        for entity_type, typed_mentions in mentions_by_type.items():
            # Get existing canonical entities of this type
            canonical = await self.get_canonical_entities(entity_type)
            
            for mention in typed_mentions:
                # Find best matching canonical entity
                best_match, score = await self.find_best_match(
                    mention, canonical, entity_type
                )
                
                if best_match and score > self.get_threshold(entity_type):
                    # Map to existing canonical
                    mappings[mention.id] = best_match.id
                    await self.update_mention_mapping(mention, best_match, score)
                else:
                    # Create new canonical entity
                    new_canonical = await self.create_canonical_entity(mention)
                    mappings[mention.id] = new_canonical.id
                    
        return mappings
    
    async def find_best_match(
        self,
        mention: EntityMention,
        candidates: List[CanonicalEntity],
        entity_type: str
    ) -> Tuple[Optional[CanonicalEntity], float]:
        """
        Find best matching canonical entity using multiple strategies.
        """
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            # Strategy 1: Exact match
            if self.normalize(mention.text) == self.normalize(candidate.name):
                return candidate, 1.0
            
            # Strategy 2: Alias match
            if mention.text.lower() in [a.lower() for a in candidate.aliases]:
                return candidate, 0.95
            
            # Strategy 3: Fuzzy match
            fuzzy_score = self.fuzzy_match(mention.text, candidate.name)
            
            # Strategy 4: Embedding similarity
            if mention.embedding and candidate.embedding:
                embedding_score = self.cosine_similarity(
                    mention.embedding, candidate.embedding
                )
                # Weighted average of fuzzy and embedding
                score = 0.6 * fuzzy_score + 0.4 * embedding_score
            else:
                score = fuzzy_score
            
            # Strategy 5: Type-specific matching
            type_score = self.type_specific_match(
                mention.text, candidate.name, entity_type
            )
            score = max(score, type_score)
            
            if score > best_score:
                best_score = score
                best_match = candidate
                
        return best_match, best_score
```

### Relationship Inference

```python
class RelationshipExtractor:
    """
    Infers relationships between canonical entities using
    document-wide context and validation rules.
    """
    
    async def infer_relationships(
        self,
        canonical_mappings: Dict[str, str],
        document_id: str,
        run_id: str
    ) -> List[CanonicalRelationship]:
        """
        Infer relationships between canonical entities.
        """
        # Get unique canonical entities
        canonical_ids = set(canonical_mappings.values())
        canonical_entities = await self.get_canonical_entities(canonical_ids)
        
        # Get full document text for context
        document_text = await self.get_document_text(document_id)
        
        # Generate relationship candidates
        candidates = self.generate_relationship_candidates(canonical_entities)
        
        # Validate relationships using LLM with full context
        relationships = []
        
        for batch in self.batch_candidates(candidates, batch_size=20):
            # Use LLM to identify valid relationships
            validated = await self.validate_relationships_with_llm(
                batch, document_text
            )
            
            for rel in validated:
                # Check if relationship type is valid for entity types
                if self.is_valid_relationship_pattern(rel):
                    relationships.append(rel)
        
        # Deduplicate and merge evidence
        relationships = self.merge_duplicate_relationships(relationships)
        
        return relationships
    
    def is_valid_relationship_pattern(
        self,
        relationship: CanonicalRelationship
    ) -> bool:
        """
        Validate relationship against allowed patterns.
        """
        # Get allowed patterns from database
        pattern = (
            relationship.source.type,
            relationship.target.type,
            relationship.type
        )
        
        return pattern in self.allowed_patterns
```

### Neo4j Projection

```python
class Neo4jProjector:
    """
    Projects canonical entities and relationships to Neo4j
    as a read-only view for visualization and graph queries.
    """
    
    async def project_to_neo4j(
        self,
        entities: List[CanonicalEntity],
        relationships: List[CanonicalRelationship],
        run_id: str
    ):
        """
        Sync entities and relationships to Neo4j.
        """
        async with self.neo4j.session() as session:
            # Clear previous projection for this document
            await self.clear_document_projection(session, document_id)
            
            # Project entities as nodes
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e += $properties
                    SET e.last_synced = datetime()
                    SET e.projection_version = $version
                    """,
                    id=entity.id,
                    properties={
                        'name': entity.name,
                        'type': entity.type,
                        'quality_score': entity.quality_score,
                        'mention_count': entity.mention_count
                    },
                    version=run_id
                )
            
            # Project relationships as edges
            for rel in relationships:
                await session.run(
                    """
                    MATCH (s:Entity {id: $source_id})
                    MATCH (t:Entity {id: $target_id})
                    MERGE (s)-[r:RELATES {type: $type}]->(t)
                    SET r.confidence = $confidence
                    SET r.evidence_count = $evidence_count
                    SET r.last_synced = datetime()
                    """,
                    source_id=rel.source_entity_id,
                    target_id=rel.target_entity_id,
                    type=rel.relationship_type,
                    confidence=rel.confidence,
                    evidence_count=rel.evidence_count
                )
```

## Implementation Components

### Quality Validation

```python
class EntityQualityValidator:
    """
    Validates entity quality using multiple criteria.
    """
    
    # Fragments and stopwords to filter
    FRAGMENTS = {
        'causing it', 'did not work', 'last', 'first', 
        'online', 'offline', 'test', 'field', 'case'
    }
    
    GENERIC_BY_TYPE = {
        'location': {'field', 'area', 'place', 'site'},
        'date': {'last', 'first', 'recent', 'old'},
        'problem': {'issue', 'problem', 'error'},
        'procedure': {'process', 'method', 'approach'}
    }
    
    def validate_entity(self, text: str, entity_type: str) -> bool:
        """
        Validate if an entity is meaningful.
        """
        # Check if it's a fragment
        if text.lower() in self.FRAGMENTS:
            return False
        
        # Check for verb phrases that aren't procedures
        if entity_type != 'procedure':
            if any(text.startswith(v) for v in ['did', 'was', 'causing', 'making']):
                return False
        
        # Check for generic terms by type
        if entity_type in self.GENERIC_BY_TYPE:
            if text.lower() in self.GENERIC_BY_TYPE[entity_type]:
                return False
        
        # Check minimum length
        if len(text) < 2:
            return False
            
        # Type-specific validation
        validators = {
            'date': self.is_valid_date,
            'measurement': self.has_units,
            'person': self.is_proper_name,
            'email': self.is_valid_email
        }
        
        if entity_type in validators:
            return validators[entity_type](text)
            
        return True
```

### Caching Layer

```python
class ExtractionCache:
    """
    Caches LLM responses with content-based keys.
    """
    
    def generate_cache_key(
        self,
        text: str,
        model: str,
        prompt_version: str,
        temperature: float
    ) -> str:
        """
        Generate deterministic cache key.
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        key_parts = [content_hash, model, prompt_version, str(temperature)]
        return ":".join(key_parts)
    
    async def get(self, cache_key: str) -> Optional[dict]:
        """
        Retrieve cached response if valid.
        """
        result = await self.db.fetch_one(
            """
            SELECT response, expires_at
            FROM extraction_cache
            WHERE content_hash = $1
              AND model = $2
              AND prompt_version = $3
              AND temperature = $4
              AND expires_at > NOW()
            """,
            *cache_key.split(":")
        )
        
        if result:
            # Update hit count and last accessed
            await self.db.execute(
                """
                UPDATE extraction_cache
                SET hit_count = hit_count + 1,
                    last_accessed = NOW()
                WHERE content_hash = $1
                """,
                cache_key.split(":")[0]
            )
            
            return result['response']
            
        return None
```

### Background Job Orchestration

```python
from celery import Celery, chain, group

class ExtractionJobs:
    """
    Background job definitions for extraction pipeline.
    """
    
    @celery.task(bind=True, max_retries=3)
    def extract_document(self, document_id: str, force_refresh: bool = False):
        """
        Main job to extract entities from a document.
        """
        # Create job chain
        workflow = chain(
            extract_mentions.s(document_id, force_refresh),
            canonicalize_entities.s(),
            infer_relationships.s(),
            update_quality_scores.s(),
            project_to_neo4j.s(),
            notify_completion.s()
        )
        
        return workflow.apply_async()
    
    @celery.task
    def extract_mentions(document_id: str, force_refresh: bool):
        """Extract entity mentions from all chunks."""
        # Implementation
        pass
    
    @celery.task
    def canonicalize_entities(mentions: List[dict]):
        """Map mentions to canonical entities."""
        # Implementation
        pass
    
    @celery.task
    def infer_relationships(canonical_mappings: dict):
        """Infer relationships between canonical entities."""
        # Implementation
        pass
```

## Migration Strategy

### Phase 1: Schema Addition (Non-Breaking)
Add new tables alongside existing ones:

```sql
-- Run all CREATE TABLE statements from New Data Model section
-- Add compatibility views
-- No changes to existing code yet
```

### Phase 2: Dual Write
Update extraction pipeline to write to both old and new schemas:

```python
class DualWriteExtractor:
    async def save_entities(self, entities, document_id):
        # Write to old schema
        await self.save_to_old_schema(entities, document_id)
        
        # Write to new schema
        mentions = self.convert_to_mentions(entities)
        canonical = await self.canonicalize_mentions(mentions)
        await self.save_mentions(mentions)
        await self.save_canonical(canonical)
```

### Phase 3: Migration of Historical Data

```sql
-- Migrate existing entities to mentions and canonical
INSERT INTO entity_mentions (
    document_id, text, type, canonical_entity_id, created_at
)
SELECT 
    document_id,
    entity_name as text,
    entity_type as type,
    canonical_entity_id,
    created_at
FROM entities;

-- Create canonical entities from unique entities
INSERT INTO canonical_entities (name, type, mention_count)
SELECT 
    entity_name as name,
    entity_type as type,
    COUNT(*) as mention_count
FROM entities
GROUP BY entity_name, entity_type;
```

### Phase 4: Cutover
Switch reads to new schema via compatibility views, then update code to use new tables directly.

### Phase 5: Cleanup
Remove old tables and compatibility views after verification.

## Success Metrics

### Quality Metrics
- **Entity Quality Score**: Average quality_score > 0.7
- **No Fragments**: Zero entities matching fragment patterns
- **Deduplication Rate**: < 5% duplicate canonical entities
- **Relationship Validity**: 100% relationships match allowed patterns

### Performance Metrics
- **Cache Hit Rate**: > 60% for repeated documents
- **Extraction Time**: < 20s per document (with cache)
- **Canonicalization Time**: < 5s per 100 mentions
- **Neo4j Sync Time**: < 10s per document

### Business Metrics
- **Entity Precision**: > 90% of entities are meaningful
- **Relationship Recall**: > 80% of true relationships captured
- **Cost Reduction**: 50% reduction in LLM costs via caching
- **User Satisfaction**: Reduced manual correction time by 70%

## Testing Strategy

### Unit Tests
```python
def test_mention_extraction():
    """Test entity mention extraction with offsets."""
    text = "The firmware update resolved the black screen issue."
    mentions = extractor.extract_mentions(text)
    
    assert len(mentions) == 3
    assert mentions[0].text == "firmware update"
    assert mentions[0].start_offset == 4
    assert mentions[0].end_offset == 19
    assert mentions[0].type == "procedure"

def test_canonicalization():
    """Test mention to canonical mapping."""
    mentions = [
        EntityMention(text="firmware update", type="procedure"),
        EntityMention(text="Firmware updates", type="procedure"),
        EntityMention(text="FW update", type="procedure")
    ]
    
    mappings = canonicalizer.canonicalize_mentions(mentions)
    
    # All should map to same canonical entity
    assert len(set(mappings.values())) == 1

def test_relationship_validation():
    """Test relationship pattern validation."""
    valid = CanonicalRelationship(
        source_type="procedure",
        target_type="problem",
        relationship_type="RESOLVES"
    )
    
    invalid = CanonicalRelationship(
        source_type="problem",
        target_type="procedure",
        relationship_type="RESOLVES"  # Wrong direction
    )
    
    assert validator.is_valid_pattern(valid) == True
    assert validator.is_valid_pattern(invalid) == False
```

### Integration Tests
```python
async def test_full_pipeline():
    """Test complete extraction pipeline."""
    document_id = "test-doc-123"
    
    # Run extraction
    run = await pipeline.run(document_id)
    
    # Verify mentions created
    mentions = await db.fetch_all(
        "SELECT * FROM entity_mentions WHERE document_id = $1",
        document_id
    )
    assert len(mentions) > 0
    
    # Verify canonicalization
    canonical = await db.fetch_all(
        """
        SELECT DISTINCT ce.* 
        FROM canonical_entities ce
        JOIN entity_mentions em ON em.canonical_entity_id = ce.id
        WHERE em.document_id = $1
        """,
        document_id
    )
    assert len(canonical) < len(mentions)  # Deduplication occurred
    
    # Verify Neo4j projection
    with neo4j.session() as session:
        result = session.run(
            "MATCH (e:Entity) WHERE e.projection_version = $run_id RETURN count(e)",
            run_id=run.id
        )
        assert result.single()[0] == len(canonical)
```

### Golden Tests
Maintain a set of documents with known correct extractions:

```python
GOLDEN_DOCUMENTS = {
    "NC2056": {
        "expected_mentions": 95,
        "expected_canonical": 43,
        "expected_relationships": 67,
        "key_entities": ["firmware update", "black screen", "factory reset"],
        "key_relationships": [
            ("factory reset", "RESOLVES", "black screen"),
            ("firmware update", "CAUSES", "data corruption")
        ]
    }
}

async def test_golden_document():
    """Test against known correct extraction."""
    for doc_id, expected in GOLDEN_DOCUMENTS.items():
        run = await pipeline.run(doc_id)
        
        assert run.mentions_extracted == expected["expected_mentions"]
        assert run.entities_canonicalized == expected["expected_canonical"]
        assert run.relationships_inferred == expected["expected_relationships"]
        
        # Verify key entities exist
        for entity_name in expected["key_entities"]:
            result = await db.fetch_one(
                "SELECT * FROM canonical_entities WHERE name = $1",
                entity_name
            )
            assert result is not None
```

## Monitoring & Observability

### Metrics to Track

```python
# Prometheus metrics
extraction_duration = Histogram('extraction_duration_seconds', 'Time to extract entities')
cache_hit_rate = Counter('cache_hits_total', 'Number of cache hits')
entity_quality = Histogram('entity_quality_score', 'Distribution of entity quality scores')
llm_tokens_used = Counter('llm_tokens_total', 'Total LLM tokens consumed')
extraction_errors = Counter('extraction_errors_total', 'Number of extraction failures')

@extraction_duration.time()
async def extract_with_metrics(document_id):
    # Extraction logic
    pass
```

### Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "extraction_completed",
    document_id=document_id,
    mentions_count=len(mentions),
    canonical_count=len(canonical),
    cache_hits=cache_hits,
    duration_seconds=duration,
    cost_usd=cost
)
```

### Dashboards
- **Extraction Overview**: Success rate, processing time, queue depth
- **Quality Metrics**: Entity quality distribution, deduplication rate
- **Cost Analysis**: Token usage, cache hit rate, cost per document
- **Error Analysis**: Failure reasons, retry patterns

## Conclusion

This architectural redesign addresses all critical issues in the current entity extraction pipeline:

1. **Separates mentions from canonical entities** - Solves the deduplication problem at its root
2. **Introduces versioning and caching** - Reduces costs and enables experimentation
3. **Single source of truth** - Supabase as primary, Neo4j as projection
4. **Quality-first approach** - Multiple validation layers ensure high-quality entities
5. **Observable and scalable** - Background jobs with metrics and monitoring

The new architecture is more complex but provides the foundation for a production-grade entity extraction system that can scale, evolve, and maintain high quality over time.
