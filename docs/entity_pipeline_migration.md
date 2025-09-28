# Entity Pipeline Migration (v2)

This document outlines how to adopt the mentions/canonical/v2 flow while preventing accidental use of the legacy extractor.

- Legacy extractor class `app/processors/entity_extractor.py` is now guarded. It will raise at instantiation unless `COCOINDEX_LEGACY_ENTITY_EXTRACTOR=1`.
- New schema is provided in `migrations/2025-09-01_entity_pipeline_v2.sql` (non-destructive). Apply via Supabase SQL or `execute_supabase_sql.py`.
- New typed models live in `app/models/entity_v2.py`.
- New CocoIndex flow scaffold: `app/flows/entity_extraction_flow_v2.py`.

## Feature Flags
- `ENTITY_PIPELINE_VERSION` (default `v2`)
- `COCOINDEX_LEGACY_ENTITY_EXTRACTOR`: set to `1` only if you must run the legacy extractor temporarily.

## Suggested Integration Path
1. Run the migration SQL to create `extraction_runs`, `entity_mentions`, `canonical_entities`, `canonical_relationships`, and `graph_projection_outbox`.
2. Wire the ingestion flow to provide a `chunks` dataset with fields `id`, `document_id`, and `text` to the v2 flow.
3. Export mentions and canonical entities to Postgres. Add a projection worker to consume `graph_projection_outbox` (to be added next).
4. Keep Neo4j as a projection target via the outbox pattern; Supabase remains the system of record.

See `entityExtraction.md` for the architectural rationale.

