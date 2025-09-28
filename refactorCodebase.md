# CocoIndex Codebase Refactoring Plan

## Overview
Systematic refactoring to fix critical gaps, reduce file sizes, and improve maintainability.

## Critical Issues to Fix
1. ❌ Missing `publish_approved_document` task (approval fails)
2. ❌ No real embedding generation (using placeholders)
3. ❌ 3 files >1,500 lines (unmaintainable)
4. ❌ Disconnected Notion pipeline
5. ❌ Empty Qdrant/Neo4j stores

---

## Phase 1: Fix Publishing Pipeline (Week 1)
**Goal:** Complete the approval → publishing → storage flow

### Task 1.1: Implement Publishing Task
**File:** `app/tasks/document_tasks.py`

- [ ] Add `publish_approved_document` function at line ~1600
- [ ] Update document state to `publishing`
- [ ] Call embedding generation
- [ ] Store in Qdrant
- [ ] Store in Neo4j
- [ ] Update state to `published` or `publish_failed`

```python
@celery_app.task(bind=True, name="publish_approved_document")
def publish_approved_document(self, document_id: str):
    # Implementation here
```

### Task 1.2: Fix Embedding Generation
**File:** `app/tasks/document_tasks.py` (line 531)

- [ ] Replace placeholder in `generate_embeddings`
- [ ] Use `EmbeddingService` from `app/services/embedding_service.py`
- [ ] Batch process chunks (100 per API call)
- [ ] Store vectors in chunk metadata

### Task 1.3: Add Storage Methods
**File:** `app/services/qdrant_service.py`

- [ ] Add `store_document_embeddings` method
- [ ] Implement batch upsert to collection

**File:** `app/services/neo4j_service.py`

- [ ] Add `store_document_graph` method
- [ ] Store canonical entities and relationships

---

## Phase 2: Refactor Giant Files (Week 2)
**Goal:** Break files >1,000 lines into manageable modules

### Task 2.1: Split entity_extraction_runner_v2.py (2,130 lines)
**Current:** `app/flows/entity_extraction_runner_v2.py`

**New Structure:**
```
app/flows/entity_extraction/
├── __init__.py
├── mention_extractor.py      (400 lines)
├── canonicalizer.py          (400 lines)
├── relationship_builder.py   (400 lines)
├── quality_filters.py        (300 lines)
└── runner.py                 (600 lines)
```

- [ ] Extract mention extraction logic
- [ ] Extract canonicalization logic
- [ ] Extract relationship building
- [ ] Extract quality filtering
- [ ] Keep main orchestration in runner.py
- [ ] Update imports in `document_tasks.py`

### Task 2.2: Split document_tasks.py (1,604 lines)
**Current:** `app/tasks/document_tasks.py`

**New Structure:**
```
app/tasks/
├── __init__.py
├── processing_tasks.py     (600 lines)
├── publishing_tasks.py     (400 lines)
├── state_tasks.py          (300 lines)
└── helpers.py              (300 lines)
```

- [ ] Move parse/chunk/embed to `processing_tasks.py`
- [ ] Move publish logic to `publishing_tasks.py`
- [ ] Move state management to `state_tasks.py`
- [ ] Extract shared utilities to `helpers.py`
- [ ] Update Celery app imports

### Task 2.3: Split entity_extractor.py (1,539 lines)
**Current:** `app/processors/entity_extractor.py`

- [ ] Check if still used (v1 legacy?)
- [ ] If used: split similar to runner_v2
- [ ] If unused: mark deprecated and schedule removal

---

## Phase 3: Consolidate & Clean (Week 3)
**Goal:** Remove redundancy and standardize patterns

### Task 3.1: Unify Document Ingestion Flows
**Current Files:**
- `document_ingestion_flow.py` (451 lines)
- `document_ingestion_flow_v2.py` (500 lines)
- `document_ingestion_dual_write.py` (718 lines)
- `document_ingestion_by_id.py` (520 lines)

- [ ] Identify which flow is actually used
- [ ] Consolidate into single `document_ingestion.py`
- [ ] Remove unused flows
- [ ] Update references

### Task 3.2: Fix Notion Pipeline
**File:** `app/pipelines/notion_ingestion.py`

- [ ] Replace SQLAlchemy with SupabaseService
- [ ] Mount in `app/main.py`
- [ ] Test with real Notion workspace

### Task 3.3: Remove Legacy Code
- [ ] Remove unused imports
- [ ] Delete commented code blocks
- [ ] Remove v1 entity extractor if unused
- [ ] Clean up experimental flows

---

## Phase 4: Performance Optimization (Week 4)
**Goal:** Improve processing speed and reliability

### Task 4.1: Add Batch Processing
**Files to Update:**
- [ ] `app/services/embedding_service.py` - Batch embed API calls
- [ ] `app/services/supabase_service.py` - Batch DB operations
- [ ] `app/services/qdrant_service.py` - Batch vector upserts

### Task 4.2: Implement Caching
- [ ] Add Redis caching for LLM responses
- [ ] Cache entity canonicalization
- [ ] Cache frequently accessed documents

### Task 4.3: Optimize Database Queries
- [ ] Add indexes to Supabase tables
- [ ] Use connection pooling
- [ ] Implement query result pagination

---

## Testing Checklist

### After Phase 1:
- [ ] Document approval triggers publishing task
- [ ] Embeddings are real OpenAI vectors (1536 dims)
- [ ] Vectors stored in Qdrant successfully
- [ ] Entities/relationships in Neo4j
- [ ] Search returns results

### After Phase 2:
- [ ] All imports still work
- [ ] Celery tasks execute correctly
- [ ] No circular dependencies
- [ ] Tests pass

### After Phase 3:
- [ ] Single ingestion flow works
- [ ] Notion pipeline processes documents
- [ ] No duplicate code remains

### After Phase 4:
- [ ] Processing time <2 min per document
- [ ] API response time <200ms
- [ ] Memory usage stable
- [ ] No timeout errors

---

## File Size Targets

| File | Current | Target | Priority |
|------|---------|--------|----------|
| entity_extraction_runner_v2.py | 2,130 | <600 | 🔴 Critical |
| document_tasks.py | 1,604 | <600 | 🔴 Critical |
| entity_extractor.py | 1,539 | Remove | 🟡 High |
| llm_service.py | 915 | <500 | 🟡 High |
| supabase_service.py | 860 | <500 | 🟡 High |
| notion_connector.py | 757 | <500 | 🟡 High |
| neo4j_service.py | 757 | <500 | 🟡 High |

---

## Quick Wins (Do First)
1. **Fix import error:** Add `publish_approved_document` stub to prevent crashes
2. **Enable real embeddings:** Switch from placeholder to OpenAI
3. **Initialize collections:** Create Qdrant/Neo4j schemas
4. **Add fallback search:** Use Supabase when stores empty

---

## Success Metrics
- ✅ All approved documents reach published state
- ✅ No files >1,000 lines
- ✅ All tests pass
- ✅ <2 minute processing per document
- ✅ Zero import errors
- ✅ Search returns relevant results

---

## Commands for Validation

```bash
# Check file sizes
find app -name "*.py" -exec wc -l {} + | sort -rn | head -10

# Find missing implementations
grep -r "publish_approved_document" app/

# Test imports
python -c "from app.tasks.document_tasks import *"

# Run tests
pytest tests/

# Check Celery tasks
celery -A app.celery_app inspect registered
```

---

*Start Date: ____________*
*Target Completion: 4 weeks*
*Daily Standup: Track progress using task checkboxes*