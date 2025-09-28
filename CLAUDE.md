# CocoIndex Data Ingestion Portal - Engineering Reference

## 1. Purpose & Scope
- Single source of truth for how the ingestion portal is wired today.
- Use alongside `architecture.md` for diagrams and `improvementPlan.md` for the backlog.
- Audience: backend engineers, ingestion operators, and frontend contributors.

## 2. System Snapshot (2025-09-25 - Updated)
### Working well
- Celery pipeline in `app/tasks/document_tasks.py` drives parse -> chunk -> real embeddings -> entity extraction, writing results and state updates to Supabase.
- **NEW**: Post-approval publishing pipeline (`publish_approved_document` task) automatically pushes approved documents to Qdrant and Neo4j.
- **NEW**: Real OpenAI embeddings generation using `EmbeddingService` with text-embedding-3-small model (1536 dimensions).
- **NEW**: Batch embedding processing (100 chunks per API call) for cost efficiency.
- **NEW**: Publishing state machine with new states: `publishing`, `published`, `publish_failed`.
- Two-tier contextual chunking with summaries and semantic focus is active via `app/processors/two_tier_chunker.py`.
- Entity extraction v2 (`app/flows/entity_extraction_runner_v2.py`) stores mentions, canonical entities, and curated relationships through `SupabaseService`; the legacy extractor only runs when `ENTITY_PIPELINE_VERSION=v1` **and** `COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1`.
- FastAPI exposes document CRUD, review actions, SSE progress, search endpoints, and job utilities under `app/api/`.
- Next.js dashboard (`frontend/app/page.tsx`) lets reviewers scan Notion/Drive, monitor jobs, review results, and reprocess documents.
- Production connectors for Notion and Google Drive enforce credential-backed access (`app/connectors`).
- **NEW**: QdrantService includes `store_document_embeddings` method for bulk vector storage.
- **NEW**: Neo4jService includes `store_document_graph` method for entity/relationship storage.

### Known gaps
- The async Notion ingestion pipeline (`app/pipelines/notion_ingestion.py`) uses legacy SQLAlchemy helpers and is not mounted in `app/main.py`.
- Celery `process_single_document` in `app/tasks/ingestion_tasks.py` returns mocked counts and should not be used outside tests.
- Some bridge endpoints expect Qdrant/Neo4j collections that may be empty in fresh environments (though graceful fallbacks are now in place).

### Near-term priorities
- ✅ ~~Swap `EmbeddingGenerator` with `EmbeddingService.embed_batch` and upsert vectors through `QdrantService`.~~ **COMPLETED**
- ✅ ~~Backfill Neo4j writes from approved canonical relationships~~ **COMPLETED via publishing pipeline**
- Reconcile Notion pipeline storage with Supabase, then re-enable `/api/ingestion`.
- Add integration coverage that exercises the full Celery pipeline against staging services.

## 3. Core Principles
1. Fetch real content: connectors in `app/connectors` require valid API tokens or service accounts; only `/api/ingestion/test` uses synthetic text for smoke checks.
2. Persist through Supabase: every pipeline stage must go through `app/services/supabase_service.py` so dashboards and state machines stay consistent.
3. Honor the state machine: transitions must follow `DocumentState` rules in `app/models/document.py`.
4. Run processing through Celery: the supported orchestration path is `process_document` in `app/tasks/document_tasks.py`; ad-hoc scripts should enqueue jobs rather than bypassing Celery.
5. Centralize LLM work: use `app/services/llm_service.py` and `app/services/embedding_service.py` for model calls so retry, cost tracking, and fallbacks remain uniform.
6. Keep progress observable: emit step updates with `_emit_progress` (SSE) and update `ProcessingJob` records to avoid blind background work.

## 4. Architecture Overview
- **Sources** - Notion databases/pages and Google Drive folders via async connectors.
- **Ingestion orchestration** - Celery workers (see `app/celery_app.py`) backed by Redis manage scanning jobs and per-document processing.
- **Processing** - LlamaParse-driven parsing, hierarchical chunking, entity extraction, and Supabase persistence (`app/tasks/document_tasks.py` plus `app/processors` and `app/flows`).
- **Storage** - Supabase (documents, chunks, mentions, canonical tables), optional Qdrant for vectors, optional Neo4j for graph exploration.
- **API layer** - FastAPI app in `app/main.py` exposes documents, review, search, bridge, and streaming endpoints.
- **Frontend** - Next.js dashboard in `frontend/` consumes REST + SSE to run reviews and manual scans.

## 5. Data Sources & Acquisition
- **Notion**: `NotionConnector` (`app/connectors/notion_connector.py`) supports per-security-level tokens, change detection, and workspace scanning. Celery tasks in `app/tasks/ingestion_tasks.py` call `run_notion_ingestion` when ingestion is enabled.
- **Google Drive**: `GoogleDriveConnector` (`app/connectors/google_drive_connector.py`) reads via service accounts, handles export of Google-native formats, and flags binary docs for LlamaParse.
- **Local/CocoIndex flows**: CocoIndex example flows (`app/flows/document_ingestion_flow*.py`) remain for experimentation but are not part of the production path; enable them only when explicitly testing the declarative engine.

## 6. Document Processing Pipeline
1. **Job kickoff** - `/api/documents/{id}/process` (see `app/api/documents.py`) creates a `ProcessingJob` row and enqueues `process_document`.
2. **Parse** - `parse_document` loads the source file and runs `DocumentParser` (`app/processors/parser.py`), using LlamaParse for complex docs and inline readers for plain text.
3. **Chunk** - `chunk_document` clears prior chunks, then runs `TwoTierChunker.process_and_save`, producing parent + semantic chunks with contextual summaries stored in `chunks`.
4. **Embeddings** - `generate_embeddings` now uses `EmbeddingService.embed_batch` to create real OpenAI embeddings (text-embedding-3-small, 1536 dimensions) and stores them in chunk metadata.
5. **Entity extraction** - `extract_entities` builds `ChunkInput` records and calls `run_extract_mentions` (CocoIndex transform). Quality filters, canonicalization, relationship creation, and description refreshes operate through `SupabaseService`.
6. **State transition** - Successful runs move the document to `pending_review`; failures capture the error, increment retries, and emit `failed` SSE events.
7. **Progress reporting** - `_emit_progress` sends SSE updates, while job rows capture `current_step` and `progress` for dashboards.
8. **Publishing** (NEW) - After approval, `publish_approved_document` task:
   - Transitions document to `publishing` state
   - Generates embeddings if not already done (using `EmbeddingService`)
   - Stores vectors in Qdrant via `QdrantService.store_document_embeddings`
   - Stores entities/relationships in Neo4j via `Neo4jService.store_document_graph`
   - Updates document to `published` or `publish_failed` state
   - Tracks publish attempts and errors for debugging

## 7. Review Workflow & State Management
- Supabase tables (`documents`, `document_state_history`, `chunks`, `entity_mentions`, `canonical_entities`, `canonical_relationships`) are the source of truth; see `app/services/supabase_service.py`.
- Review endpoints in `app/api/documents_review.py` approve or reject documents, update audit history, and **now trigger the publishing pipeline automatically on approval**.
- Document states now include: `discovered`, `processing`, `pending_review`, `approved`, `rejected`, `ingested`, `failed`, `deleted`, **`publishing`, `published`, `publish_failed`**.
- `DocumentStateManager` (`app/services/state_manager.py`) validates transitions, exposes metrics, and powers the SSE layer.
- The Next.js dashboard surfaces chunk/entity counts and allows manual reprocessing, invoking the same Celery pipeline.

## 8. APIs & Background Services
- **Documents** (`app/api/documents.py`) - list, detail, edit metadata, trigger processing, approve/reject, download chunks/entities.
- **Processing** (`app/api/processing.py`) - legacy scan endpoints, SSE status streams, and job polling helpers (ingestion router currently detached in `app/main.py`).
- **Search** (`app/api/search.py`) - wraps `SearchService` for vector, graph, and hybrid queries (expects embeddings/Qdrant data).
- **Bridge** (`app/api/bridge.py`) - debugging utilities for Qdrant/Neo4j synchronization.
- **Celery** - `app/tasks/document_tasks.py` for per-document processing and `app/tasks/ingestion_tasks.py` for source scans; `app/celery_app.py` configures the worker used by the API.

## 9. Frontend Overview
- Built with Next.js 15 and React Query (`frontend/app/page.tsx`), providing:
  - Live health badges for backend, Redis, Neo4j, and Qdrant.
  - Notion/Google Drive scan triggers that poll `/api/process/jobs/{id}/status`.
  - Document table with selection, hard delete, inline status, and reprocess actions.
  - SSE progress feeds for active documents.
- Uses `.env.local` for API base URLs and expects the backend on `http://localhost:8001` during development.

## 10. Configuration & Environment
`app/config.py` loads all runtime settings. Windows dev scripts expect the `.venv312` virtual environment (dot prefix) so the interpreter resolves to `.venv312\Scripts\python.exe`; update the scripts if you keep Python elsewhere. Required keys:

```
# Runtime & infrastructure
ENVIRONMENT=development
DATABASE_URL=postgresql://...
REDIS_URL=redis://localhost:6379/0

# Supabase
SUPABASE_URL=https://...
SUPABASE_KEY=...
SUPABASE_HOST=...
SUPABASE_DB_PASSWORD=...

# Vector / graph stores
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

# LLMs & parsing
LLAMA_CLOUD_API_KEY=...
LLAMA_PARSE_BASE_URL=https://api.cloud.llamaindex.ai/api/v1
OPENAI_API_KEY=...
GOOGLE_AI_API_KEY=...

# Sources
NOTION_API_KEY (legacy fallback)
NOTION_API_KEY_[PUBLIC|CLIENT|PARTNER|EMPLOYEE|MANAGEMENT]_ACCESS=...
GOOGLE_DRIVE_CREDENTIALS_PATH=...
GOOGLE_DRIVE_FOLDER_IDS=[...]

# Optional
COHERE_API_KEY=...
ENTITY_PIPELINE_VERSION=v2
CORS_ORIGINS=["http://localhost:3000", ...]
```

Default builds run with `ENTITY_PIPELINE_VERSION=v2`; set `ENTITY_PIPELINE_VERSION=v1` together with `COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1` only when you intentionally need the legacy extractor.

Keep secrets in `.env` locally and inject via the deployment platform in other environments.

## 11. Local Development Workflow

### Quick Start (Recommended)
We now have simplified start/stop scripts that handle everything consistently:

**Windows (PowerShell):**
```powershell
# Start infrastructure only (recommended for development)
.\start.ps1

# Start everything in Docker containers
.\start.ps1 -Full

# Stop everything
.\stop.ps1

# Stop and clean volumes
.\stop.ps1 -Full -Clean
```

**Linux/Mac (Bash):**
```bash
# Start infrastructure only (recommended for development)
./start.sh

# Start everything in Docker containers
./start.sh --full

# Stop everything
./stop.sh

# Stop and clean volumes
./stop.sh --full --clean
```

The scripts will:
1. Start infrastructure services (Redis, PostgreSQL, Neo4j, Qdrant) in Docker
2. Detect your Python virtual environment (.venv312, venv312, .venv, or venv)
3. Start the backend API on port 8005
4. Start the Celery worker
5. Start the frontend on port 3000
6. Show combined logs (Ctrl+C to stop viewing, services continue running)

### Docker Compose Options
For more control, use Docker Compose directly:

```bash
# Start infrastructure only
docker-compose -f docker-compose.dev.yml up -d

# Start everything in containers
docker-compose -f docker-compose.full.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes
docker-compose -f docker-compose.dev.yml down -v
```

### Manual Startup (Alternative)
If you prefer manual control:

1. **Dependencies** - start infrastructure containers:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. **Backend API** - in your virtual environment:
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8005
   ```

3. **Celery worker** - in your virtual environment:
   ```bash
   celery -A app.celery_app worker --loglevel=info --pool=solo
   ```

4. **Frontend** - in the frontend directory:
   ```bash
   cd frontend && npm run dev
   ```

5. **Smoke check** - verify services:
   - Backend health: `http://localhost:8005/health`
   - Frontend: `http://localhost:3000`
   - Neo4j Browser: `http://localhost:7474`
   - Qdrant Dashboard: `http://localhost:6333/dashboard`

### Environment Setup
1. Copy `.env.example` to `.env`
2. Fill in your API keys and credentials
3. Ensure your virtual environment is created: `python -m venv .venv312`
4. Install dependencies: `pip install -r requirements.txt`
5. Install frontend dependencies: `cd frontend && npm install`

### Legacy PowerShell Scripts
The old scripts in `scripts/` directory (`start-dev.ps1`, `stop-dev.ps1`) are being replaced. If you still need them:
```powershell
# Old startup
pwsh scripts/start-dev.ps1

# Old shutdown
pwsh scripts/stop-dev.ps1 -StopContainers
```

## 12. Observability & Troubleshooting
- **Celery worker recovery**: `restartCelery.md` captures PowerShell commands to kill stray workers and restart the `.venv312` task with `--pool=solo`; use it if the dashboard warns about missing workers or the wrong virtual environment.
- Server-Sent Events: `/api/documents/{id}/progress` (documents) and `/api/documents/{id}/stream` (processing) provide real-time updates.
- Job status: `/api/process/jobs/{job_id}/status` surfaces Celery progress; logs stream to the `logs/` directory.
- `app/services/progress_tracker.py` centralizes programmatic progress pushes.
- For ingestion scans, inspect Celery logs (`app/tasks/ingestion_tasks.py`) and the in-memory `JobTracker` exposed via `/api/process/sources/scan` responses.

## 13. Outstanding Work & Risks
- ✅ ~~Replace placeholder embeddings and wire Qdrant/Neo4j updates~~ **COMPLETED - Real embeddings and publishing pipeline implemented**
- Align Notion ingestion (`app/pipelines/notion_ingestion.py`) with Supabase-first storage to avoid divergent SQLAlchemy models.
- Audit bridge endpoints so destructive helpers are gated in shared environments.
- Expand automated tests to cover canonicalization, relationship quality filters, and end-to-end retries across real documents.
- Monitor OpenAI API costs for embedding generation (currently using text-embedding-3-small at $0.020 per 1M tokens).

## 14. Reference Index
- `app/tasks/document_tasks.py` - Celery pipeline, state updates, SSE emission, **publishing task**.
- `app/processors/two_tier_chunker.py` - contextual chunk generation and Supabase persistence.
- `app/flows/entity_extraction_runner_v2.py` - CocoIndex transform powering entity extraction v2.
- `app/services/llm_service.py` - OpenAI/Gemini orchestration, caching, and fallbacks.
- `app/services/embedding_service.py` - **NEW: OpenAI embedding generation with batch processing**.
- `app/services/supabase_service.py` - Supabase CRUD and caching helpers, **canonical entity/relationship retrieval**.
- `app/services/qdrant_service.py` - Vector storage, **store_document_embeddings method**.
- `app/services/neo4j_service.py` - Graph storage, **store_document_graph method**.
- `app/api/documents.py` / `app/api/documents_review.py` - REST surface for review workflow, **publishing trigger**.
- `app/api/search.py` & `app/services/search_service.py` - hybrid search implementation.
- `frontend/app/page.tsx` - reviewer dashboard logic.
- `architecture.md` & `improvementPlan.md` - complementary roadmap context.
- `approvalImplementation.md` - **NEW: Detailed implementation plan for post-approval publishing pipeline**.

## Port binding troubleshooting
- 2025-09-26: Local Windows host reserved port 8001 via HTTP.sys (System process), blocking uvicorn. Either delete the reservation (
etsh http delete urlacl url=http://+:8001/) or run backend on an alternate port (currently 8005).
- Update scripts/start-dev.ps1 to bind to 8005 and scripts/stop-dev.ps1 to release port 8005. Ensure frontend .env.local sets NEXT_PUBLIC_API_URL=http://localhost:8005. 

