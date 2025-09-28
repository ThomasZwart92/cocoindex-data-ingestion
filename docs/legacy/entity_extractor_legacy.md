# Legacy Entity Extractor (Archived)

This document archives the previous implementation of the entity extraction pipeline for reference.

- Source file: `app/processors/entity_extractor.py`
- Status: Deprecated. Disabled by default via feature flag to prevent accidental execution.
- Note: Do not modify the legacy implementation. New work should target the v2 pipeline and data model.

If you need to review the old logic, open `app/processors/entity_extractor.py`. The class is guarded so it will not run unless explicitly enabled.

