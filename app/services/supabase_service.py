"""Supabase service for database operations"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from supabase import create_client, Client
from app.config import settings
from app.models.document import Document, DocumentState
from app.models.chunk import Chunk
from app.models.entity import Entity, EntityRelationship
from app.models.entity_v2 import EntityMention, CanonicalEntity, CanonicalRelationship
from app.models.job import ProcessingJob
import time

# Align gotrue's legacy 'proxy' keyword with current httpx signature
try:
    from gotrue.http_clients import SyncClient as _GoTrueSyncClient
    from httpx import Client as _HttpxClient
except ImportError:
    _GoTrueSyncClient = None
else:
    if _GoTrueSyncClient is not None and not getattr(_GoTrueSyncClient, '_proxy_kw_fix', False):
        _orig_sync_init = _HttpxClient.__init__

        def _patched_sync_init(self, *args, proxy=None, **kwargs):
            if proxy is not None:
                kwargs.setdefault('proxies', proxy)
            _orig_sync_init(self, *args, **kwargs)

        _GoTrueSyncClient.__init__ = _patched_sync_init
        _GoTrueSyncClient._proxy_kw_fix = True

# Create a singleton Supabase client
_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get or create the singleton Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        # Create client with default settings for now
        # The httpx client options were causing issues
        _supabase_client = create_client(
            settings.supabase_url, 
            settings.supabase_key
        )
    return _supabase_client

logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for Supabase operations - Singleton pattern"""
    
    _instance = None
    _cache: Dict[str, tuple[Any, float]] = {}  # Shared cache across all instances
    _cache_ttl = 5.0  # seconds
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseService, cls).__new__(cls)
            cls._instance.client = get_supabase_client()
        return cls._instance
    
    def __init__(self):
        # Client is already initialized in __new__
        pass
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set cached value with current timestamp"""
        self._cache[key] = (value, time.time())
    
    def _clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern or all if no pattern"""
        if pattern:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
        else:
            self._cache.clear()
    
    # Document operations
    def create_document(self, document: Document) -> Document:
        """Create a new document"""
        data = document.to_supabase_dict()
        result = self.client.table("documents").insert(data).execute()
        if result.data:
            return Document(**result.data[0])
        raise Exception("Failed to create document")
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        result = self.client.table("documents").select("*").eq("id", document_id).execute()
        if result.data:
            return Document(**result.data[0])
        return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> Document:
        """Update document"""
        # Handle enum conversions
        if 'status' in updates and isinstance(updates['status'], DocumentState):
            updates['status'] = updates['status'].value
        
        result = self.client.table("documents").update(updates).eq("id", document_id).execute()
        if result.data:
            # Clear document list cache when a document is updated
            self._clear_cache("docs:")
            return Document(**result.data[0])
        raise Exception("Failed to update document")
    
    def update_document_status(self, document_id: str, status: DocumentState, error: Optional[str] = None) -> Document:
        """Update document status"""
        updates = {"status": status.value}
        if error:
            updates["processing_error"] = error
        if status == DocumentState.PROCESSING:
            updates["processing_error"] = None
        return self.update_document(document_id, updates)
    
    def list_documents(self, status: Optional[DocumentState] = None, limit: int = 100, source_type: Optional[str] = None, source_id: Optional[str] = None) -> List[Document]:
        """List documents with optional filtering and caching"""
        # Create cache key from parameters
        cache_key = f"docs:{status}:{limit}:{source_type}:{source_id}"
        
        # Check cache first
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        query = self.client.table("documents").select("*")
        
        if status:
            query = query.eq("status", status.value)
        # No longer filtering out rejected documents since we're using hard delete
        
        if source_type:
            query = query.eq("source_type", source_type)
        
        if source_id:
            query = query.eq("source_id", source_id)
        
        result = query.limit(limit).order("created_at", desc=True).execute()
        documents = [Document(**doc) for doc in result.data]
        
        # Cache the result
        self._set_cached(cache_key, documents)
        
        return documents
    
    # Chunk operations
    def create_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Create multiple chunks"""
        data = [chunk.to_supabase_dict() for chunk in chunks]
        result = self.client.table("chunks").insert(data).execute()
        if result.data:
            return [Chunk(**chunk) for chunk in result.data]
        raise Exception("Failed to create chunks")
    
    def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document"""
        result = self.client.table("chunks").select("*").eq("document_id", document_id).order("chunk_index").execute()
        return [Chunk(**chunk) for chunk in result.data]
    
    def update_chunk(self, chunk_id: str, updates: Dict[str, Any]) -> Chunk:
        """Update a chunk"""
        result = self.client.table("chunks").update(updates).eq("id", chunk_id).execute()
        if result.data:
            return Chunk(**result.data[0])
        raise Exception("Failed to update chunk")
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        self.client.table("chunks").delete().eq("document_id", document_id).execute()
        return True
    
    def delete_document_entity_mentions(self, document_id: str) -> int:
        """Delete all entity mentions for a document."""
        try:
            resp = (
                self.client
                .table("entity_mentions")
                .delete(count="exact")
                .eq("document_id", document_id)
                .execute()
            )
        except Exception as exc:
            logger.warning("Failed to delete entity mentions for document %s: %s", document_id, exc)
            return 0

        if hasattr(resp, "count") and resp.count is not None:
            return int(resp.count)
        return len(resp.data or [])

    # Entity operations
    def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple entities"""
        data = [entity.to_supabase_dict() for entity in entities]
        result = self.client.table("entities").insert(data).execute()
        if result.data:
            return [Entity(**entity) for entity in result.data]
        raise Exception("Failed to create entities")
    
    def get_document_entities(self, document_id: str) -> List[Entity]:
        """Get all entities for a document"""
        result = self.client.table("entities").select("*").eq("document_id", document_id).execute()
        return [Entity(**entity) for entity in result.data]
    
    def create_entity_relationships(self, relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Create entity relationships"""
        data = [rel.to_supabase_dict() for rel in relationships]
        result = self.client.table("entity_relationships").insert(data).execute()
        if result.data:
            return [EntityRelationship(**rel) for rel in result.data]
        raise Exception("Failed to create relationships")
    
    # Job operations
    def create_job(self, job: ProcessingJob) -> ProcessingJob:
        """Create a processing job"""
        data = job.to_supabase_dict()
        result = self.client.table("processing_jobs").insert(data).execute()
        if result.data:
            return ProcessingJob(**result.data[0])
        raise Exception("Failed to create job")
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        result = self.client.table("processing_jobs").select("*").eq("id", job_id).execute()
        if result.data:
            return ProcessingJob(**result.data[0])
        return None
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> ProcessingJob:
        """Update a job"""
        # Handle enum conversions
        if 'job_status' in updates:
            from app.models.job import JobStatus
            if isinstance(updates['job_status'], JobStatus):
                updates['job_status'] = updates['job_status'].value
        
        result = self.client.table("processing_jobs").update(updates).eq("id", job_id).execute()
        if result.data:
            return ProcessingJob(**result.data[0])
        raise Exception("Failed to update job")
    
    def get_document_jobs(self, document_id: str) -> List[ProcessingJob]:
        """Get all jobs for a document"""
        result = self.client.table("processing_jobs").select("*").eq("document_id", document_id).order("created_at", desc=True).execute()
        return [ProcessingJob(**job) for job in result.data]

    # -------------------------
    # Entity Pipeline v2 (mentions/canonical)
    # -------------------------

    def create_extraction_run(self, document_id: str, pipeline_version: str, prompt_version: str | None = None, model: str | None = None, input_hash: str | None = None) -> str:
        """Create a new extraction_runs row and return its id."""
        payload = {
            "document_id": document_id,
            "pipeline_version": pipeline_version,
            "prompt_version": prompt_version,
            "model": model,
            "status": "running",
            "input_hash": input_hash,
            "started_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
        result = self.client.table("extraction_runs").insert(payload).execute()
        if not result.data:
            raise Exception("Failed to create extraction run")
        return result.data[0]["id"]

    def complete_extraction_run(self, run_id: str, mentions: int, canonical: int, relationships: int, status: str = "completed") -> None:
        """Mark extraction run completed with counters."""
        updates = {
            "status": status,
            "mentions_extracted": mentions,
            "entities_canonicalized": canonical,
            "relationships_inferred": relationships,
            "completed_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self.client.table("extraction_runs").update(updates).eq("id", run_id).execute()

    def insert_entity_mentions(self, mentions: list[EntityMention], extraction_run_id: str) -> int:
        """Batch insert entity mentions. Returns number inserted."""
        rows = []
        for m in mentions:
            row = {
                "document_id": m.document_id,
                "chunk_id": m.chunk_id,
                "extraction_run_id": extraction_run_id,
                "text": m.text,
                "type": m.type,
                "start_offset": m.start_offset,
                "end_offset": m.end_offset,
                "confidence": m.confidence,
                "context": m.context,
                "attributes": m.attributes or {},
                "canonical_entity_id": m.canonical_entity_id,
                "canonicalization_score": m.canonicalization_score,
            }
            rows.append(row)
        if not rows:
            return 0
        result = self.client.table("entity_mentions").insert(rows).execute()
        return len(result.data or [])

    def get_canonical_entity(self, name: str, type_: str) -> dict | None:
        """Fetch a canonical entity by name/type, tolerating differences in type casing."""
        type_normalized = (type_ or "concept").strip()
        candidates: list[str] = []
        if type_normalized:
            candidates.append(type_normalized)
            upper_candidate = type_normalized.upper()
            lower_candidate = type_normalized.lower()
            if upper_candidate not in candidates:
                candidates.append(upper_candidate)
            if lower_candidate not in candidates:
                candidates.append(lower_candidate)
        else:
            candidates.append("concept")

        for candidate in dict.fromkeys(candidates):
            res = (
                self.client.table("canonical_entities")
                .select("*")
                .eq("name", name)
                .eq("type", candidate)
                .limit(1)
                .execute()
            )
            if res.data:
                return res.data[0]
        return None


    def get_canonical_entities_by_ids(self, canonical_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch canonical entity records for the provided ids."""
        ids = sorted({cid for cid in canonical_ids if cid})
        if not ids:
            return []
        try:
            response = (
                self.client
                .table("canonical_entities")
                .select("id,name,type,metadata")
                .in_("id", ids)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network call safeguard
            logger.warning("Failed to fetch canonical entities by ids: %s", exc)
            return []
        return response.data or []

    def upsert_canonical_entities(self, entities: list[CanonicalEntity]) -> list[str]:
        """Upsert canonical entities by (name,type). Returns list of ids in same order."""
        ids: list[str] = []
        for ce in entities:
            # Guard against invalid rows
            if not ce or not isinstance(ce.name, str) or not ce.name.strip():
                continue
            ce_type = ce.type or "concept"
            existing = self.get_canonical_entity(ce.name, ce.type)
            if existing:
                ids.append(existing["id"])
                continue
            payload = {
                "name": ce.name.strip(),
                "type": ce_type,
                "aliases": ce.aliases,
                "quality_score": ce.quality_score,
                "is_validated": ce.is_validated,
                "definition": ce.definition,
                "category": ce.category,
                "metadata": ce.metadata,
            }
            res = self.client.table("canonical_entities").insert(payload).execute()
            if res.data:
                ids.append(res.data[0]["id"])
            else:
                # Fallback: query again (another process may have created it)
                existing2 = self.get_canonical_entity(ce.name, ce.type)
                if not existing2:
                    raise Exception(f"Failed to upsert canonical entity: {ce.name} ({ce.type})")
                ids.append(existing2["id"])
        return ids

    def upsert_canonical_entities_map(self, entities: list[CanonicalEntity]) -> dict[tuple[str, str], str]:
        """Upsert canonical entities and return a mapping with case-stable type keys."""
        mapping: dict[tuple[str, str], str] = {}
        for ce in entities:
            if not ce or not isinstance(ce.name, str) or not ce.name.strip():
                continue

            ce_name = ce.name.strip()
            ce_type_raw = ce.type or "concept"
            ce_type = ce_type_raw.strip().upper()
            key_name = ce_name.lower()

            existing = self.get_canonical_entity(ce_name, ce_type)
            if existing:
                cid = existing["id"]
            else:
                payload = {
                    "name": ce_name,
                    "type": ce_type,
                    "aliases": ce.aliases,
                    "quality_score": ce.quality_score,
                    "is_validated": ce.is_validated,
                    "definition": ce.definition,
                    "category": ce.category,
                    "metadata": ce.metadata,
                }
                res = self.client.table("canonical_entities").insert(payload).execute()
                if res.data:
                    cid = res.data[0]["id"]
                else:
                    existing2 = self.get_canonical_entity(ce_name, ce_type)
                    if not existing2:
                        raise Exception(f"Failed to upsert canonical entity: {ce.name} ({ce.type})")
                    cid = existing2["id"]

            base_keys = {
                (key_name, ce_type),
                (key_name, ce_type.lower()),
            }
            for mapping_key in base_keys:
                mapping[mapping_key] = cid

            typed_alias = f"{ce_name} ({ce_type})".lower()
            mapping[(typed_alias, ce_type)] = cid
            mapping[(typed_alias, ce_type.lower())] = cid

            if ce.aliases:
                for alias in ce.aliases:
                    if isinstance(alias, str) and alias.strip():
                        alias_norm = alias.strip().lower()
                        mapping[(alias_norm, ce_type)] = cid
                        mapping[(alias_norm, ce_type.lower())] = cid
        return mapping


    def get_cached_extraction_run(self, document_id: str, input_hash: str) -> Optional[str]:
        """Return the latest completed extraction_run id for a given document and input_hash, if any."""
        res = (
            self.client.table("extraction_runs")
            .select("id,status,created_at")
            .eq("document_id", document_id)
            .eq("input_hash", input_hash)
            .eq("status", "completed")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["id"]
        return None

    def get_mentions_by_run(self, extraction_run_id: str) -> list[dict]:
        """Fetch mentions persisted for a given extraction_run_id."""
        res = (
            self.client.table("entity_mentions")
            .select("id,document_id,chunk_id,text,type,start_offset,end_offset,confidence,canonical_entity_id")
            .eq("extraction_run_id", extraction_run_id)
            .execute()
        )
        return res.data or []

    def insert_canonical_relationships(self, rels: list[CanonicalRelationship]) -> int:
        """Insert canonical relationships (idempotency ensured by unique constraint)."""
        if not rels:
            return 0
        rows = []
        for r in rels:
            rows.append({
                "source_entity_id": r.source_entity_id,
                "target_entity_id": r.target_entity_id,
                "relationship_type": r.relationship_type,
                "confidence_score": r.confidence_score,
                "metadata": r.metadata or {},
            })
        res = self.client.table("canonical_relationships").insert(rows, count="exact").execute()
        return len(res.data or [])

    def get_document_relationships(self, document_id: str) -> list[Dict[str, Any]]:
        """Fetch canonical relationships associated with a specific document."""
        try:
            response = (
                self.client
                .table("canonical_relationships")
                .select("*")
                .contains("metadata", {"document_id": str(document_id)})
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network call safeguard
            logger.error("Failed to fetch relationships for document %s: %s", document_id, exc)
            raise
        return response.data or []

    def create_document_relationship(
        self,
        document_id: str,
        *,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        confidence_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        is_manual: bool = True,  # Default to manual for API calls
    ) -> Dict[str, Any]:
        """Create a single canonical relationship.

        For manually created relationships (is_manual=True), we store document_context
        instead of document_id to preserve them during reprocessing.
        For auto-generated relationships (is_manual=False), we store document_id
        so they get cleaned up on reprocessing.
        """
        payload_metadata: Dict[str, Any] = dict(metadata or {})

        if is_manual:
            # Manual relationships: preserve during reprocessing
            payload_metadata["manual"] = True
            payload_metadata["document_context"] = str(document_id)
            # Remove document_id if it was set
            payload_metadata.pop("document_id", None)
        else:
            # Auto-generated relationships: clean up on reprocessing
            payload_metadata["document_id"] = str(document_id)

        record: Dict[str, Any] = {
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "relationship_type": relationship_type,
            "confidence_score": confidence_score,
            "metadata": payload_metadata,
            "is_verified": is_manual,  # Manual relationships are pre-verified
        }

        try:
            response = (
                self.client
                .table("canonical_relationships")
                .insert(record)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network call safeguard
            logger.error("Failed to create relationship for document %s: %s", document_id, exc)
            raise

        data = response.data or []
        if not data:
            raise RuntimeError("Supabase insert returned no data for canonical_relationships")
        return data[0]

    def delete_document_relationships(self, document_id: str, canonical_ids: Optional[list[str]] = None) -> int:
        """Remove canonical relationships generated for a specific document during reprocessing.

        This will delete ALL relationships that were auto-generated for this document
        (identified by document_id in metadata), regardless of verification status.
        This ensures clean reprocessing without accumulating duplicates.

        Manually created relationships (without document_id in metadata) are preserved.
        """
        ids_to_delete: list[str] = []

        # First priority: Delete all relationships that were generated for this document
        # (These have document_id in their metadata)
        try:
            doc_relationships = (
                self.client
                .table("canonical_relationships")
                .select("id,metadata")
                .contains("metadata", {"document_id": document_id})
                .execute()
            )

            for rel in (doc_relationships.data or []):
                rel_id = rel.get("id")
                if rel_id:
                    ids_to_delete.append(rel_id)

            if ids_to_delete:
                logger.info(
                    "Removing %d existing relationships for document %s before regenerating",
                    len(ids_to_delete),
                    document_id
                )
        except Exception as exc:
            logger.warning("Failed to fetch document relationships for cleanup: %s", exc)

        # Optionally also clean up relationships involving canonical IDs that no longer exist
        # This handles the case where entities changed between processings
        if canonical_ids:
            canonical_filter = set(filter(None, canonical_ids))
            if canonical_filter:
                try:
                    # Find relationships involving entities from this document
                    # but only delete if they're not manually created (no document_id means manual)
                    source_rels = (
                        self.client
                        .table("canonical_relationships")
                        .select("id,metadata,source_entity_id,target_entity_id")
                        .in_("source_entity_id", list(canonical_filter))
                        .execute()
                    )

                    target_rels = (
                        self.client
                        .table("canonical_relationships")
                        .select("id,metadata,source_entity_id,target_entity_id")
                        .in_("target_entity_id", list(canonical_filter))
                        .execute()
                    )

                    # Only delete if both endpoints are in our canonical set
                    # AND it's not a manually created relationship
                    for rel in (source_rels.data or []) + (target_rels.data or []):
                        rel_id = rel.get("id")
                        if rel_id and rel_id not in ids_to_delete:
                            metadata = rel.get("metadata") or {}
                            # Skip manually created relationships (those without document_id)
                            if not metadata.get("document_id"):
                                continue
                            # Only delete if both entities are from this document's processing
                            if (rel.get("source_entity_id") in canonical_filter and
                                rel.get("target_entity_id") in canonical_filter):
                                ids_to_delete.append(rel_id)

                except Exception as exc:
                    logger.warning("Failed to fetch canonical relationships for cleanup: %s", exc)

        if not ids_to_delete:
            return 0

        # Remove duplicates from the deletion list
        ids_to_delete = list(set(ids_to_delete))

        try:
            self.client.table("canonical_relationships").delete().in_("id", ids_to_delete).execute()
            logger.info(f"Successfully deleted {len(ids_to_delete)} relationships for document {document_id}")
        except Exception as exc:  # pragma: no cover - network call safeguard
            logger.warning("Failed to delete document relationships: %s", exc)
            return 0

        return len(ids_to_delete)

    def refresh_canonical_entity_metrics(
        self,
        canonical_ids: list[str],
        description_updates: dict[str, str] | None = None,
        existing_metadata: dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        """Recompute mention/document/relationship counts and metadata for canonical entities."""
        ids = sorted({cid for cid in canonical_ids if cid})
        if not ids:
            return

        description_updates = description_updates or {}

        if existing_metadata is not None:
            meta_map: Dict[str, Dict[str, Any]] = {
                cid: existing_metadata.get(cid, {}) or {}
                for cid in ids
            }
        else:
            # Fetch existing metadata
            try:
                entity_resp = (
                    self.client
                    .table("canonical_entities")
                    .select("id,metadata")
                    .in_("id", ids)
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - network call safeguard
                logger.warning("Failed to fetch canonical entities for refresh: %s", exc)
                return

            meta_map = {}
            for row in entity_resp.data or []:
                cid = row.get("id")
                if cid:
                    meta_map[cid] = row.get("metadata") or {}

        metrics: Dict[str, Dict[str, Any]] = {
            cid: {
                "mention_count": 0,
                "documents": set(),
                "confidence_total": 0.0,
                "confidence_samples": 0,
                "relationship_count": 0,
                "relationship_documents": set(),
            }
            for cid in ids
        }

        # Mentions aggregation
        try:
            mentions_resp = (
                self.client
                .table("entity_mentions")
                .select("canonical_entity_id,document_id,confidence")
                .in_("canonical_entity_id", ids)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network call safeguard
            logger.warning("Failed to fetch mentions for canonical refresh: %s", exc)
            mentions_resp = None

        if mentions_resp and mentions_resp.data:
            for row in mentions_resp.data:
                cid = row.get("canonical_entity_id")
                data = metrics.get(cid)
                if not data:
                    continue
                data["mention_count"] += 1
                doc_id = row.get("document_id")
                if doc_id:
                    data["documents"].add(str(doc_id))
                conf = row.get("confidence")
                if isinstance(conf, (int, float)):
                    data["confidence_total"] += float(conf)
                    data["confidence_samples"] += 1

        # Relationships aggregation
        def _collect_relationships(column: str):
            try:
                resp = (
                    self.client
                    .table("canonical_relationships")
                    .select("id,source_entity_id,target_entity_id,metadata")
                    .in_(column, ids)
                    .execute()
                )
            except Exception as exc:  # pragma: no cover - network call safeguard
                logger.warning("Failed to fetch relationships for %s aggregation: %s", column, exc)
                return []
            return resp.data or []

        rel_rows = _collect_relationships("source_entity_id") + _collect_relationships("target_entity_id")

        for row in rel_rows:
            source_id = row.get("source_entity_id")
            target_id = row.get("target_entity_id")
            metadata = row.get("metadata") or {}
            doc_id = metadata.get("document_id")

            for cid in (source_id, target_id):
                data = metrics.get(cid)
                if not data:
                    continue
                data["relationship_count"] += 1
                if doc_id:
                    data["relationship_documents"].add(str(doc_id))

        # Apply updates
        for cid, data in metrics.items():
            current_metadata = meta_map.get(cid, {}) or {}
            doc_ids = sorted(data["documents"])
            rel_doc_ids = sorted(data["relationship_documents"])

            avg_confidence = (
                data["confidence_total"] / data["confidence_samples"]
                if data["confidence_samples"]
                else current_metadata.get("quality_score", 0.5)
            )

            metadata_update = dict(current_metadata)
            if doc_ids:
                metadata_update["document_ids"] = doc_ids
            if rel_doc_ids:
                metadata_update["relationship_document_ids"] = rel_doc_ids
            metadata_update["last_refreshed_at"] = datetime.utcnow().isoformat()
            if cid in description_updates:
                metadata_update["description"] = description_updates[cid]

            update_payload = {
                "mention_count": data["mention_count"],
                "document_count": len(doc_ids),
                "relationship_count": data["relationship_count"],
                "quality_score": max(0.0, min(avg_confidence, 1.0)),
                "metadata": metadata_update,
            }

            try:
                self.client.table("canonical_entities").update(update_payload).eq("id", cid).execute()
            except Exception as exc:  # pragma: no cover - network call safeguard
                logger.warning("Failed to update canonical entity %s metrics: %s", cid, exc)

    def get_canonical_entities_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all canonical entities associated with a document

        Args:
            document_id: Document ID

        Returns:
            List of canonical entity dictionaries
        """
        try:
            # Get entity mentions for the document
            mentions_response = self.client.table("entity_mentions") \
                .select("canonical_entity_id") \
                .eq("document_id", document_id) \
                .execute()

            if not mentions_response.data:
                return []

            # Get unique canonical entity IDs
            canonical_ids = list(set(m['canonical_entity_id'] for m in mentions_response.data if m.get('canonical_entity_id')))

            if not canonical_ids:
                return []

            # Get canonical entities
            entities_response = self.client.table("canonical_entities") \
                .select("*") \
                .in_("id", canonical_ids) \
                .execute()

            return entities_response.data if entities_response.data else []

        except Exception as e:
            logger.error(f"Failed to get canonical entities for document {document_id}: {e}")
            return []

    def get_canonical_relationships_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all canonical relationships associated with a document

        Args:
            document_id: Document ID

        Returns:
            List of canonical relationship dictionaries
        """
        try:
            # Get relationships where document_id is in metadata
            relationships_response = self.client.table("canonical_relationships") \
                .select("*") \
                .execute()

            # Filter relationships that contain this document_id in metadata
            document_relationships = []
            for rel in relationships_response.data:
                metadata = rel.get('metadata', {})
                if isinstance(metadata, dict) and metadata.get('document_id') == document_id:
                    document_relationships.append(rel)

            return document_relationships

        except Exception as e:
            logger.error(f"Failed to get canonical relationships for document {document_id}: {e}")
            return []

