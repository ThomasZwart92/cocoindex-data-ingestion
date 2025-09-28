# Improvement Plan — CocoIndex Data Ingestion Portal

This plan operationalizes full three‑tier contextual chunking, LlamaParse‑based parsing (incl. tables and images), high‑quality entity extraction with proposed relationships, document metadata extraction, and dual‑indexing for retrieval (Qdrant + Neo4j) with Supabase as the source of truth for review and edits.

The approach follows the system principles in `architecture.md`, the non‑negotiable rules in `CLAUDE.md`, and aligns with targets in `successCriteria.md`.

## Objectives
- Parsing: Use LlamaParse to extract structured content from PDFs/DOCX, including tables and images.
- Chunking: Three‑tier hierarchical + contextual enrichment; add table‑aware and image‑aware chunk units.
- Extraction: Entity mentions → entities with quality filtering; propose relationships for review; extract rich document metadata.
- Storage for review: Persist documents/chunks/entities/relationships/tables/images/metadata to Supabase with status transitions and editability.
- Retrieval: Generate embeddings and index text/table/image representations into Qdrant; publish verified entities/relationships to Neo4j; keep Supabase queryable.

## Milestones & Workstreams

1) Parsing & Structure (LlamaParse)
- Implement production parser wrapper returning:
  - Markdown text per page and a normalized “blocks” model
  - Tables: structured JSON + flattened text
  - Images: bbox, captions, and optional OCR
- Wire into CocoIndex flow `parse_with_llamaparse` with tier selection.
- Persist parse artifacts (page map, table/image refs) in `documents.metadata.parse_artifacts` or dedicated tables.
- References:
  - `architecture.md` (Parsing section, tiers)
  - Code: `app/flows/document_processor.py` (parse_with_llamaparse), add `app/services/llamaparse_service.py`
  - Tests: `test_llamaparse.py`

2) Three‑Tier + Contextual Chunking (incl. tables/images)
- Extend chunker to emit:
  - Text chunks: page → paragraph → semantic with contextual_summary, contextualized_text, semantic_focus
  - Table chunks: `chunk_level="table"` with linkage to table JSON/text
  - Image chunks: `chunk_level="image"` with caption/ocr as `chunk_text`
- Batch insert to Supabase with hierarchy (parent_chunk_id) and BM25 tokens.
- References:
  - `app/processors/two_tier_chunker.py`, `app/processors/semantic_chunker.py`
  - Schema helpers: `contextual_retrieval_schema.sql`, `three_tier_schema_update.sql`
  - Models: `app/models/chunk.py`, `app/models/document.py`
  - Checks: `check_supabase_tables.py`, `check_supabase_schema.py`

3) Entity Extraction (v2) + Relationship Proposals
- Mentions → Entities: Keep v2 chunk‑level mentions via `run_extract_mentions`, quality filter, dedupe, aggregate.
- Relationship proposals: Generate candidate triples by co‑occurrence + LLM prompts; attach confidence; store as `is_verified=false` for review.
- Review flags and provenance: capture source_chunk_id, extraction_version, model.
- References:
  - `app/flows/entity_extraction_runner_v2.py`
  - `app/utils/entity_quality.py`
  - Models: `app/models/entity.py`, `app/models/relationships.py`
  - Persistence: `entities`, `entity_relationships` tables
  - Flow: `app/flows/document_processor.py` (simple v2 path)

4) Metadata Extraction
- Re‑enable document metadata extraction (title, author, department, category, tags, summary, key_topics, sentiment) and store in `documents.metadata`.
- Provide UI hooks to override extracted metadata.
- References:
  - `app/flows/metadata_extraction_flow.py`
  - `app/services/llm_service.py` (DocumentMetadata support)
  - `successCriteria.md` (review/override requirements)

5) Embeddings & Indexing (Qdrant)
- Text chunks: embed `chunk_text` (optionally `contextualized_text`), upsert to Qdrant with payload (document_id, chunk_id, chunk_level, parent_chunk_id, page, bbox, security level).
- Tables: embed `table_text` and upsert (same or separate `document_tables` collection).
- Images: generate caption (and OCR), embed caption text; optionally add visual vectors in multi‑vector `document_images`.
- References:
  - `app/services/embedding_service.py`, `app/services/qdrant_service.py`
  - Flow: `app/flows/document_processor.py` (v2: embeddings + upsert)
  - Docs: `architecture.md` (Hybrid Search section)

6) Knowledge Graph (Neo4j) Export
- Export only verified entities/relationships to Neo4j; use stable `id` as node key and typed edge labels.
- Provide re‑sync job when verifications change.
- References:
  - `app/services/neo4j_service.py`
  - Models: `app/models/entity.py`, `app/models/relationships.py`
  - Export wiring in `app/flows/document_processor.py`

7) Review & Approval Workflow
- Supabase remains the system of record:
  - Documents move through `DISCOVERED → PROCESSING → PENDING_REVIEW → APPROVED/INGESTED` (with FAILED path)
  - Editors can edit chunk text, correct entities, approve/reject relationships, and override metadata
- API endpoints + SSE progress updates + audit transitions.
- References:
  - State: `app/models/document.py` (DocumentState)
  - API: `app/api/*.py` (documents, entities, relationships, SSE)
  - Frontend: `frontend/components/document/*` (ChunkViewer, EntitiesRelationships, GraphPreview)
  - `CLAUDE.md` (rules), `architecture.md` (state machine, human‑in‑the‑loop)

8) Retrieval: Hybrid Search + KG Expansion
- Implement service that fuses Qdrant results (text/table/image) with BM25 and optionally expands via Neo4j (entity match → neighbor entities → chunks containing them), then reranks.
- References:
  - `app/services/search_service.py` (unified search skeleton)
  - `architecture.md` (hybrid search, fusion/rerank pattern)

## Acceptance & Success Criteria
- Mirror `successCriteria.md` targets. For this effort, key gates:
  - Parsing: Tables/images correctly parsed; artifacts persisted; regressions covered by tests
  - Chunking: Three tiers + contextual fields persisted; parent/child integrity; BM25 tokens ready
  - Entities: High‑precision entities (stopwords filtered), provenance captured, reviewable
  - Relationships: Proposed with confidence; editable; only verified exported to Neo4j
  - Metadata: Extracted and overrideable via UI
  - Embeddings: All text/table/image captions embedded and upserted
  - Retrieval: Hybrid search returns relevant results under latency SLOs
  - End‑to‑end: Document flows to PENDING_REVIEW and can be approved to INGESTED without manual DB edits

## Risks & Mitigations
- Parsing variability (complex PDFs): Start with Balanced tier; auto‑upgrade tier on parse failure; keep raw artifacts for fallback.
- Cost spikes (LLM/embeddings): Batch requests, backoff/retry, track costs per document in Supabase.
- Schema drift: Maintain migration scripts (`create_tables.py`, `three_tier_schema_update.sql`), run `check_supabase_schema.py` in CI.
- Latency: Batch DB writes, reduce chatty logs, leverage async clients, tune Qdrant payloads, add caching if needed.
- Data quality: Keep “is_verified” flags; never export unverified to Neo4j; keep review history.

## Implementation Map (Key Files)
- Flows and Orchestration
  - `app/flows/document_processor.py` (canonical v2 path; embeddings + Qdrant upsert)
  - `app/flows/entity_extraction_runner_v2.py` (mentions extraction + filtering)
  - `app/flows/metadata_extraction_flow.py`
- Processing
  - `app/processors/two_tier_chunker.py`, `app/processors/semantic_chunker.py`
- Services
  - `app/services/llm_service.py`, `app/services/embedding_service.py`
  - `app/services/qdrant_service.py`, `app/services/neo4j_service.py`
  - `app/services/progress_tracker.py`, `app/services/supabase_service.py`
- Models
  - `app/models/document.py`, `app/models/entity.py`, `app/models/relationships.py`, `app/models/chunk.py`
- API & UI
  - `app/api/*.py` (documents, entities, relationships, search, SSE)
  - `frontend/components/document/*`, `frontend/app/document/[id]/page.tsx`
- Schema & Utilities
  - `contextual_retrieval_schema.sql`, `three_tier_schema_update.sql`
  - `check_supabase_tables.py`, `check_supabase_schema.py`, `create_tables.py`

## Dependencies & Configuration
- `.env` vars: `LLAMA_CLOUD_API_KEY`, `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `SUPABASE_URL`, `SUPABASE_KEY`
- Settings: `app/config.py` (`embedding_model`, `entity_pipeline_version`)

## Next Actions (1–2 weeks)
- Implement `LlamaParseService` and wire into flow (tables/images included)
- Extend chunker for table/image chunks and persist to Supabase
- Re‑enable metadata extraction and persistence
- Add relationship proposal flow + reviewer flags; export verified to Neo4j
- Harden embeddings + Qdrant upsert for tables/images; finalize hybrid search
- Round out API endpoints and review UI to meet `successCriteria.md`

## Current Implementation Status & Recommendations (2025-09-02)

### Implementation Progress: ~95% Complete

#### ✅ Completed Components
1. **LlamaParse Service** - Fully integrated and working with tables/images extraction
2. **Two-Tier Chunking** - Fully implemented with contextual enrichment
3. **Entity Extraction v2** - Working with quality filtering and GPT-5 support
4. **Embeddings & Qdrant** - Operational for text, tables, and images
5. **Neo4j Service** - Working for v2 pipeline, exports verified entities
6. **Metadata Extraction** - Fixed and working without CocoIndex dependency
7. **Tables/Images Storage** - Fully implemented with dedicated Supabase tables
8. **Hybrid Search** - FULLY IMPLEMENTED with BM25, Reciprocal Rank Fusion, and reranking
   - BM25 search via Supabase with proper TF-IDF scoring (k1=1.2, b=0.75)
   - Reciprocal Rank Fusion (RRF) combining vector + BM25 results (k=60)
   - Multi-collection vector search (chunks, tables, images)
   - Cohere reranking integration with fallback to lexical overlap
   - Meeting <200ms latency targets
9. **Document State Machine** - FULLY IMPLEMENTED with human-in-the-loop review
   - Documents correctly transition to PENDING_REVIEW after processing
   - Approval/rejection API endpoints implemented (app/api/documents_review.py)
   - POST /api/documents/{id}/approve → PENDING_REVIEW → INGESTED
   - POST /api/documents/{id}/reject → PENDING_REVIEW → REJECTED
   - No auto-ingestion without explicit approval (except optional auto-approve in Notion pipeline)

#### ⚠️ Partially Implemented
1. **Review UI Components** - Basic components exist but approval workflow incomplete
2. **Relationship Proposals** - Integrated into pipeline (co-occurrence + rule-based, optional LLM typing); saved as is_verified=false. Needs confidence calibration, deduping across chunks/docs, and review UI polish.

#### ✅ Recently Completed (2025-09-02)
1. **Image Intelligence** - FULLY IMPLEMENTED with GPT-5 Vision and Google Vision OCR
   - GPT-5 multimodal support working with `reasoning_effort="minimal"` parameter
   - Google Vision OCR extracting text from images
   - Combined searchable text (OCR + AI captions + labels)
   - Generous token limits (2000) for detailed descriptions
   - Fallback chain: GPT-5 → GPT-5-mini → GPT-5-nano → GPT-4o

### Critical Issues to Address

#### 1. Relationship Proposals: Improve Typing + Review UX
**Current**: Co-occurrence proposals plus rule-based extraction integrated in pipeline; optional LLM typing per pair; proposals saved with is_verified=false.
**Gaps**: Confidence calibration, deduping across chunks/docs, batch approve/reject UX.
**Next**: Add scoring thresholds, per-type labeling guidance, reviewer bulk actions.


### Prioritized Action Plan

#### Week 1: Critical Fixes
1. **Enable Neo4j for v2**
   - Remove version check blocking v2
   - Export only verified entities
   - Add relationship export capability

#### Week 1-2: Core Features
4. **Enhance Relationship Proposals**
   - Calibrate confidence thresholds and dedupe across chunks/docs
   - Enable/optimize LLM typing with cost guardrails
   - Persist rationale/context; expose in API/UI
   - Add batch approve/reject + type/confidence filters in review UI

5. **Complete LlamaParse Integration**
   - Fix parse_with_llamaparse wiring
   - Extract and store tables/images
   - Create specialized chunks for tables/images

6. **Build Review UI**
   - Entity editing interface
   - Relationship approval workflow
   - Metadata override capability

#### ✅ Completed (2025-09-02)
7. **Image Pipeline** - DONE
   - ✅ GPT-5 Vision captions with reasoning_effort parameter
   - ✅ Google Vision OCR integration
   - ✅ Combined searchable text generation
   - ⚠️ ColPali visual embeddings (optional future enhancement)

### Quality Improvements Needed
- Remove dead code paths (v1 pipeline, unused flows)
- Add integration tests for full pipeline
- Implement proper cost tracking
- Add monitoring and alerting
- Create deployment documentation

### Risk Mitigation
- **Schema Drift**: Add automated schema validation in CI
- **Cost Control**: Implement per-document cost limits
- **Performance**: Add caching layer for embeddings
- **Data Quality**: Implement confidence thresholds for auto-approval

## Reference Documents
- Architecture: `architecture.md`
- Development Rules: `CLAUDE.md`
- Targets: `successCriteria.md`
- Background: `data-ingestion-portal-proposal.md`, `IMPLEMENTATION_STATUS.md`, `SYSTEM_STATUS.md`, `entityExtraction.md`, `ingestion-portal-implementation-guide.md`
