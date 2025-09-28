# Simple v2 Processing Flow (Deprecated)

This document preserves the high-level structure of the deprecated "simple v2" document processing flow that previously lived in `app/flows/document_processor.py`.

The application no longer calls this code — the Celery-based pipeline in `app/tasks/document_tasks.py` is the sole path for processing and reprocessing. The API endpoints have been updated to enqueue Celery jobs and return `job_id` and `celery_task_id` for progress tracking.

If you need to review or restore exact prior code, refer to your git history for `app/flows/document_processor.py` before this change.

## What It Did

- Fetched a document from Supabase
- Optionally parsed with LlamaParse (PDF/DOCX)
- Three-tier chunking with contextual summaries and BM25 tokens
- Embedded chunks and upserted to Qdrant
- Extracted entity mentions via v2 runner and proposed relationships
- Saved chunks/entities/relationships to Supabase
- Exported verified items to Neo4j
- Sent live progress via `ProgressTracker`

## Key Deprecated Functions

```python
# DEPRECATED: Use Celery pipeline instead
async def process_document_simple(document_id: str, progress_callback=None) -> Dict[str, Any]:
    """
    Processes a single document through:
      - LlamaParse (optional)
      - Three-tier chunking (hierarchical, contextual)
      - Table/image extraction + embeddings
      - v2 entity mentions via LLM + quality filtering
      - Relationship proposals (co-occurrence + optional LLM typing)
      - Supabase persistence (chunks, tables, images, entities, relationships)
      - Optional Neo4j export of verified items
    Sends SSE progress via ProgressTracker.
    """
    ...


# DEPRECATED: Use POST /api/documents/{id}/process (Celery)
async def create_and_run_flow(document_id: str) -> Dict[str, Any]:
    """
    Previously initialized CocoIndex and invoked the simple pipeline.
    Now deprecated; Celery pipeline replaces this for reliability and observability.
    """
    ...
```

## Why Deprecated

- Celery provides better retry semantics, isolation, and observability.
- Unified v2 entity pipeline (mentions + canonical + relationships) persists to Supabase with `extraction_runs` for caching/metrics.
- API endpoints return `job_id`/`celery_task_id` and SSE now polls job status.

## Where To Look Now

- Orchestration: `app/tasks/document_tasks.py`
- Entity v2 runner: `app/flows/entity_extraction_runner_v2.py`
- Supabase helpers: `app/services/supabase_service.py`
- Progress SSE: `app/api/documents.py` (progress endpoint) + `app/services/progress_tracker.py`

## Notes on GPT‑5 Configuration

- All GPT‑5 calls should use `model="gpt-5"` with `reasoning_effort="minimal"` and `temperature=1.0`.
- The `LLMService` enforces these when `model` starts with `"gpt-5"`.

