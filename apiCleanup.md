# API Cleanup Plan

This document lists concrete issues and recommended fixes to professionalize and stabilize the API used by the frontend and tests. It focuses on consistency, clarity, and removal of ambiguity so the UI has a single, predictable contract.

## Goals
- Single canonical path per capability (no path collisions).
- Consistent payload shapes across endpoints for the same resource.
- Clear, typed OpenAPI with real examples for success and error.
- Stable naming for fields and states; avoid legacy drift.
- Minimal breaking changes with migration notes where needed.

## Status Summary (2025-09-02)
- Bridge router moved to `/api/bridge` to avoid collisions (Done).
- Deprecated duplicates: `/api/documents/{id}/stream` and `/api/process/jobs/{job_id}/status` hidden from schema (Done).
- Canonical chunks endpoint remains `/api/documents/{id}/chunks` (Done).
- Standardized document detail payloads for chunks and entities; added `/api/documents/{id}/entities` with standardized shape and a confidence alias (Done).
- Relationships CRUD returns raw records (no envelope) consistently (Done).
- Added relationship proposals endpoints: list proposals and bulk approve/reject (Done).
- Expanded default CORS to include common dev origins (Done).
- SSE docs include concrete event examples (Done).
- Operation IDs and full 4xx/5xx error schemas: Pending.
- Versioning (`/api/v1`): Pending.

## Issues & Fixes

### 1) Route Collisions and Path Layout
- Duplicate chunks route:
  - Status: Completed. Canonical stays at `/api/documents/{id}/chunks`. Bridge variant now at `/api/bridge/documents/{id}/chunks`.
  - Files: `app/api/bridge.py`, `app/api/documents.py`, `app/main.py`.

- Streaming endpoints duplication:
  - Status: Completed. `/api/documents/{id}/progress` remains public. `/api/documents/{id}/stream` deprecated (hidden from schema). SSE examples added.
  - Files: `app/api/documents.py`, `app/api/processing.py`, `app/api/sse.py`.

- Jobs endpoints inconsistency:
  - Status: Completed. `/api/process/jobs/{job_id}/status` deprecated (hidden from schema). `/api/jobs/{job_id}/status` remains.
  - Files: `app/api/processing.py`.

### 2) Payload Shape Inconsistencies
- Chunks shape across endpoints:
  - Standardize on: `chunk_index`, `chunk_number` (alias), `chunk_text`, `chunk_size`, `start_position`, `end_position`, `metadata`.
  - Status: Completed for document detail and `/api/documents/{id}/chunks`. Bridge outputs remain for Qdrant view; consider normalizing or documenting as separate shape.
  - Files: `app/api/documents.py`, `app/api/bridge.py`.

- Entities shape:
  - Standardize on: `entity_name`, `entity_type`, `confidence_score`, `metadata`.
  - Current: Some endpoints (bridge, legacy) expose `name`, `type`, `confidence`.
  - Status: Document detail and `/api/documents/{id}/entities` return standardized keys with a `confidence` alias for compatibility. Bridge may continue returning its native shape; optionally normalize later.
  - Files: `app/api/documents.py`, `app/api/bridge.py`, `app/api/entities.py`.

- Relationships shape:
  - Status: Completed. Relationships CRUD now returns raw objects consistently.
  - Files: `app/api/relationships.py`.

### 3) OpenAPI & Documentation
- Operation metadata:
  - Status: Tags, summaries, response models, and real examples added for core endpoints.
  - Next: Add `operation_id` for stable client generation; optionally add per-route 422 examples.

- SSE documentation:
  - Status: Completed. Examples added to `app/api/sse.py`.
  - Files: `app/api/sse.py`, `app/api/documents.py`, `app/api/processing.py`.

- Versioning:
  - Recommendation: Introduce `/api/v1` prefix to stabilize future changes. Optional now; plan for v1.1 later.

### 4) Review Workflow & State Semantics
- Approve/Reject flow:
  - Status: Implemented under `/api/documents/{id}/approve|reject` (moves to `ingested|rejected`).
  - Optional enhancement: Consider explicit `approved` intermediate state before `ingested` if the frontend UX requires a two-step ingest, else keep current.

- Timestamps:
  - Consistency between `processed_at` and `ingested_at`. Ensure they are set/output consistently and documented in doc detail.

### 5) Relationship Proposals & Management
- Proposals in pipeline:
  - Status: Co-occurrence + rule-based inference integrated; optional LLM typing; saved as `is_verified=false`.
  - API Enhancements:
    - Status: Completed. Added list endpoint and bulk approve/reject.
  - Export:
    - Status: Confirmed. Neo4j export in v2 flow filters `is_verified=true`.
  - Files: `app/api/relationships.py`, `app/api/bridge.py`, `app/flows/document_processor.py`.

### 6) Processing & Ingestion
- Title vs Name mapping:
  - Current: Update uses `title` and `name`. The DB uses `name`.
  - Status: Completed. Title is mapped to name during update.
  - Files: `app/api/documents.py`.

- List endpoints shape:
  - Current: `GET /api/documents/` returns a raw list of dicts with counts; some list endpoints return an envelope.
  - Fix: Choose a consistent pattern (recommend raw list for collection GETs; envelopes for multi-field paginated responses). If envelope, return `{ items, total, limit, offset }`.

### 7) Error Handling & CORS
- Error schema:
  - Status: Partially applied on core endpoints; extend across remaining routes.

- CORS origins:
  - Status: Completed. Defaults expanded to include `http://127.0.0.1:3000` and `http://localhost:5173`.
  - Files: `.env`, `app/config.py`, `app/main.py`.

### 8) Bridge Endpoints Scope
- Pathing:
  - Status: Completed. Bridge mounted at `/api/bridge`.
  - Files: `app/main.py`, `app/api/bridge.py`.

### 9) Testing & Contracts
- Update test harness expectations:
  - `test_frontend_backend_connection.py` expects fields: chunks require `chunk_number`, `chunk_text`, `chunk_size`; entities require `entity_type`, `entity_name`, `confidence`. Align with standardized keys or update test accordingly.
  - Ensure tests call the canonical routes (avoid bridge duplicates).

### 10) Security (Future)
- Add authentication/authorization and document requirements in OpenAPI:
  - Security schemes (e.g., bearer token), per-route protection, and role rules for approve/reject.
  - Not required for current cleanup but important for production.

## Action Plan (Prioritized)
1) OpenAPI completeness
   - Add `operation_id` across routes.
   - Ensure all 4xx/5xx responses use `ErrorResponse` models with examples.
   - Add SSE examples to route descriptions (module-level examples added).

2) Normalize remaining payloads
   - Consider normalizing bridge payloads to standardized keys or documenting them explicitly.
   - Keep legacy aliases for one release and mark deprecated in docs.

3) Versioning
   - Plan `/api/v1` migration banner; prepare redirect/aliases.

## Acceptance Criteria
- No duplicate paths across routers.
- All chunk/entity/relationship payloads use standardized field names; legacy aliases accepted for one release and marked deprecated.
- /docs shows clear, grouped endpoints with real examples and unique `operation_id`s.
- Streaming endpoint documented with event examples; clients can consume without inspecting code.
- Relationship proposals fetchable and reviewable in bulk; only verified exported to Neo4j.
- CORS permits active frontend origins in development.

## Migration Notes
- Deprecate old paths with `include_in_schema=False` and add warnings in logs for 2 releases before removal.
- For entity/relationship field renames, return both new and legacy keys during deprecation window.
- Communicate changes to frontend maintainers with example payload diffs.

## QA Checklist
- [x] GET /api/documents returns list with chunk/entity counts; pagination consistent.
- [x] GET /api/documents/{id} returns standardized chunk and entity shapes.
- [x] GET /api/documents/{id}/chunks matches standardized keys and ordering.
- [x] GET /api/documents/{id}/entities (proxy) returns standardized entity keys (with confidence alias).
- [x] POST /api/documents/{id}/approve|reject handle invalid state with clear errors.
- [x] POST /api/relationships, PUT /api/relationships/{id}, DELETE /api/relationships/{id} return consistent shapes.
- [x] SSE endpoints stream events with documented structure.
- [ ] No 404s for expected frontend routes; no route collisions in logs (verify in integration).

## Open Questions
- Should we introduce an explicit `approved` intermediate state before `ingested`, or is direct `pending_review → ingested` sufficient for the UI?
- Should the relationships CRUD adopt an envelope everywhere or return raw objects? (Recommend raw objects for simplicity.)
- Do we want to expose bridge queries under `/api/bridge` only, and offer a proxy under `/api/documents/{id}/entities` to avoid frontend dependency on multiple stores?
