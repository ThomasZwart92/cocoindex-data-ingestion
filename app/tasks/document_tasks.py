"""Document processing tasks"""

import asyncio
import json

import logging
from uuid import uuid4

from collections import defaultdict
from dataclasses import asdict

from datetime import datetime, timezone

from typing import List, Dict, Any, Optional

from celery import Task, chain, group

from app.celery_app import celery_app

from app.models.document import Document, DocumentState, ParseTier

from app.models.chunk import Chunk, ChunkingStrategy

from app.models.job import ProcessingJob, JobType, JobStatus

from app.services.supabase_service import SupabaseService
from app.services.embedding_service import EmbeddingService, EmbeddingResult
from app.services.qdrant_service import QdrantService
from app.services.neo4j_service import Neo4jService

from app.processors.parser import DocumentParser

from app.processors.two_tier_chunker import TwoTierChunker


from app.config import settings

from app.flows.entity_extraction_runner_v2 import run_extract_mentions, ChunkInput

from app.models.entity_v2 import EntityMention, CanonicalEntity, CanonicalRelationship

from app.services.canonical_description_service import CanonicalEntityDescriptionService

from app.services.relationship_extractor import RelationshipExtractor

from app.utils.relationship_types import canonicalize_relationship_type



logger = logging.getLogger(__name__)



class DocumentTask(Task):

    """Base task with database connections"""

    _supabase = None

    

    @property

    def supabase(self):

        if self._supabase is None:

            self._supabase = SupabaseService()

        return self._supabase

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        document_id = kwargs.get('document_id')
        job_id = kwargs.get('job_id')

        if not document_id and args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                document_id = first_arg.get('document_id')
            elif isinstance(first_arg, (str, bytes)):
                document_id = first_arg.decode() if isinstance(first_arg, bytes) else first_arg

        if not job_id and len(args) > 1:
            second_arg = args[1]
            if isinstance(second_arg, dict):
                job_id = second_arg.get('job_id') or second_arg.get('id')
            elif isinstance(second_arg, (str, bytes)):
                job_id = second_arg.decode() if isinstance(second_arg, bytes) else second_arg

        if not job_id:
            job_id = kwargs.get('job_id')

        try:
            if document_id:
                self.supabase.update_document_status(str(document_id), DocumentState.FAILED, error=str(exc))
        except Exception:
            logger.exception('Failed to update document status to FAILED during task failure')

        try:
            if job_id:
                job = self.supabase.get_job(str(job_id))
                if job:
                    job.fail(str(exc))
                    self.supabase.update_job(str(job_id), job.to_supabase_dict())
        except Exception:
            logger.exception('Failed to update processing job during task failure')

        super().on_failure(exc, task_id, args, kwargs, einfo)



@celery_app.task(

    bind=True, 

    base=DocumentTask, 

    name="process_document",

    autoretry_for=(Exception,),

    retry_kwargs={'max_retries': 3, 'countdown': 5},

    retry_backoff=True,

    retry_backoff_max=600,

    retry_jitter=True

)

def process_document(self, document_id: str, job_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Main document processing pipeline
    Orchestrates: parse -> chunk -> embed -> extract entities

    force_reprocess is currently used by the API to signal a clean re-run;
    the Celery pipeline always performs the full sequence regardless of the flag
    but logs it for observability.

    Retry policy:
    - Max 3 retries with exponential backoff
    - Initial delay: 5 seconds
    - Max delay: 600 seconds (10 minutes)
    - Adds jitter to prevent thundering herd
    """
    logger.info(
        f"Starting document processing for {document_id} (attempt {self.request.retries + 1}, "
        f"force_reprocess={force_reprocess})"
    )

    job = None

    try:
        document = self.supabase.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        job = self.supabase.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.start()
        self.supabase.update_job(job_id, job.to_supabase_dict())

        self.supabase.update_document_status(document_id, DocumentState.PROCESSING)

        processing_chain = chain(
            parse_document.s(document_id, job_id),
            chunk_document.s(job_id),
            generate_embeddings.s(job_id),
            extract_entities.s(job_id),
            finalize_document_processing.s(job_id, document_id)
        )

        async_result = processing_chain.apply_async()

        self.supabase.update_job(job_id, {
            'celery_task_id': async_result.id,
            'job_status': JobStatus.RUNNING,
            'progress': 1,
            'current_step': 'Queued for processing'
        })

        logger.info(
            f"Queued document processing chain for {document_id} (job {job_id}, task {async_result.id})"
        )
        return {
            'document_id': document_id,
            'job_id': job_id,
            'status': 'processing',
            'celery_task_id': async_result.id
        }

    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {str(e)}")

        try:
            self.supabase.update_document_status(
                document_id,
                DocumentState.FAILED,
                error=str(e)
            )
        except Exception:
            logger.exception('Failed to mark document as FAILED after orchestration error')

        try:
            if job:
                job.fail(str(e))
                self.supabase.update_job(job_id, job.to_supabase_dict())
        except Exception:
            logger.exception('Failed to mark processing job as failed after orchestration error')

        raise
@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="parse_document",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 10},
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def parse_document(self, document_id: str, job_id: str) -> Dict[str, Any]:
    """Parse document using LlamaParse with automatic retry"""
    logger.info(f"Parsing document {document_id} (attempt {self.request.retries + 1})")

    try:
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 10,
            "current_step": "Parsing document with LlamaParse"
        })

        # Get document
        document = self.supabase.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        metadata = dict(document.metadata or {})
        source_url = str(getattr(document, 'source_url', '') or '').strip()
        inline_content = getattr(document, 'content', None)

        if not source_url:
            for key in ("source_url", "download_url", "file_path", "local_path"):
                candidate = metadata.get(key)
                if candidate:
                    source_url = str(candidate).strip()
                    if source_url:
                        break

        if not source_url and inline_content:
            text_content = inline_content if isinstance(inline_content, str) else str(inline_content)
            parse_confidence = 0.9

            parse_metadata = dict(metadata.get("parse_metadata") or {})
            strategy_label = getattr(document.source_type, 'value', str(document.source_type))
            parse_metadata.update({
                "parser": "inline_content",
                "strategy": strategy_label,
                "confidence": parse_confidence,
            })

            metadata["parse_metadata"] = parse_metadata
            metadata["parsed_text_length"] = len(text_content)

            self.supabase.update_document(document_id, {
                "parse_confidence": parse_confidence,
                "metadata": metadata
            })

            self.supabase.update_job(job_id, {
                "progress": 25,
                "current_step": "Document parsed using inline content"
            })

            logger.info("Using inline content for %s (%d chars)", document_id, len(text_content))
            return {
                "document_id": document_id,
                "text": text_content,
                "metadata": metadata,
                "images": [],
                "pages": []
            }

        if not source_url:
            raise ValueError("Document has no source_url or inline content to parse")

        # Parse document via LlamaParse
        parser = DocumentParser()
        parse_result = parser.parse(
            document_path=source_url,
            document_name=document.name,
            parse_tier=ParseTier.BALANCED  # Start with balanced
        )

        metadata["parsed_text_length"] = len(parse_result["text"])
        metadata["parse_metadata"] = parse_result.get("metadata", {})

        # Update document with parse results
        self.supabase.update_document(document_id, {
            "parse_tier": ParseTier.BALANCED.value,
            "parse_confidence": parse_result.get("confidence", 0.8),
            "metadata": metadata
        })

        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 25,
            "current_step": "Document parsed successfully"
        })

        return {
            "document_id": document_id,
            "text": parse_result["text"],
            "metadata": parse_result.get("metadata", {}),
            "images": parse_result.get("images", []),
            "pages": parse_result.get("pages", [])
        }

    except Exception as e:
        logger.error(f"Document parsing failed: {str(e)}")
        raise


@celery_app.task(

    bind=True,

    base=DocumentTask,

    name="chunk_document",

    autoretry_for=(Exception,),

    retry_kwargs={'max_retries': 2, 'countdown': 5},

    retry_backoff=True

)

def chunk_document(self, parse_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Chunk parsed document text with automatic retry"""

    document_id = parse_result["document_id"]
    text = parse_result["text"]
    pages = parse_result.get("pages")

    logger.info(f"Chunking document {document_id} (attempt {self.request.retries + 1})")

    try:
        # Update job progress
        self.supabase.update_job(job_id, {
            "progress": 35,
            "current_step": "Chunking document (two-tier)"
        })

        document = self.supabase.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        # Start fresh for this document
        self.supabase.delete_document_chunks(document_id)

        chunk_ids: List[str] = []
        chunk_count = 0
        semantic_count = 0
        chunking_method = "two_tier"
        job_step_message = ""

        document_name = document.name or (document.metadata or {}).get("title") or document_id
        source_type_value = getattr(document.source_type, "value", document.source_type)

        metadata_template: Dict[str, Any] = {
            "document_id": document_id,
            "document_name": document_name,
            "source_type": source_type_value,
        }
        if document.security_level:
            metadata_template["security_level"] = document.security_level
        if document.access_level is not None:
            metadata_template["access_level"] = document.access_level
        if document.metadata:
            metadata_template["document_metadata"] = json.loads(json.dumps(document.metadata, default=str))
        if getattr(document, "doc_metadata", None):
            metadata_template["doc_metadata"] = json.loads(json.dumps(document.doc_metadata, default=str))

        parse_metadata = parse_result.get("metadata") or {}
        if parse_metadata:
            metadata_template["parse_metadata"] = json.loads(json.dumps(parse_metadata, default=str))

        metadata_template = {k: v for k, v in metadata_template.items() if v is not None}

        chunker = TwoTierChunker()

        def _run_chunker():
            return chunker.process_document(
                document_id=document_id,
                content=text,
                title=document_name,
                metadata=metadata_template or None,
                pages=pages,
            )

        try:
            chunk_data_list = asyncio.run(_run_chunker())
        except RuntimeError as runtime_err:
            if "asyncio.run()" in str(runtime_err):
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    chunk_data_list = loop.run_until_complete(_run_chunker())
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
            else:
                raise

        if not chunk_data_list:
            raise ValueError("Two-tier chunker returned no chunks")

        parent_summaries: Dict[str, Optional[str]] = {}
        semantic_positions: Dict[str, int] = defaultdict(int)
        chunk_payloads: List[Dict[str, Any]] = []

        for index, chunk_data in enumerate(chunk_data_list):
            tier = (chunk_data.metadata or {}).get("tier")
            chunk_data.chunk_index = index

            if tier == "parent":
                chunk_data.chunk_level = "page"
                chunk_data.parent_chunk_id = None
                chunk_data.chunking_strategy = "semantic"  # Use valid DB enum value
                parent_summaries[chunk_data.id] = chunk_data.contextual_summary
                position_in_parent: Optional[int] = None
            else:
                chunk_data.chunk_level = "semantic"
                chunk_data.chunking_strategy = "semantic"  # Use valid DB enum value
                parent_identifier = chunk_data.parent_chunk_id or ""
                position_in_parent = semantic_positions[parent_identifier]
                semantic_positions[parent_identifier] = position_in_parent + 1
                semantic_count += 1

            chunk_dict = asdict(chunk_data)
            chunk_dict["chunk_size"] = len(chunk_dict.get("chunk_text") or "")
            chunk_dict["chunking_strategy"] = "semantic"  # Use valid DB enum value
            # Remove chunk_level as it doesn't exist in DB schema
            chunk_dict.pop("chunk_level", None)
            # Store chunk_level in metadata instead
            if not chunk_dict.get("metadata"):
                chunk_dict["metadata"] = {}
            chunk_dict["metadata"]["chunk_level"] = "page" if tier == "parent" else "semantic"
            chunk_dict["bm25_tokens"] = chunk_dict.get("bm25_tokens") or []
            chunk_dict["position_in_parent"] = position_in_parent
            if tier != "parent":
                chunk_dict["parent_context"] = parent_summaries.get(chunk_dict.get("parent_chunk_id"))
            else:
                chunk_dict["parent_context"] = None

            chunk_dict.pop("sentence_count", None)
            semantic_focus = chunk_dict.pop("semantic_focus", None)
            if semantic_focus:
                metadata_value = chunk_dict.get("metadata") or {}
                metadata_value["semantic_focus"] = semantic_focus
                chunk_dict["metadata"] = metadata_value

            for dt_field in ("created_at", "updated_at"):
                if chunk_dict.get(dt_field):
                    chunk_dict[dt_field] = chunk_dict[dt_field].isoformat()

            metadata_value = chunk_dict.get("metadata") or {}
            chunk_dict["metadata"] = json.loads(json.dumps(metadata_value, default=str))

            chunk_payloads.append(chunk_dict)

        if not chunk_payloads:
            raise ValueError("Two-tier chunker produced no payloads")

        response = self.supabase.client.table("chunks").insert(chunk_payloads).execute()
        saved_rows = response.data or []
        if not saved_rows:
            raise ValueError("Two-tier chunk insert returned no rows")

        chunk_ids = [row.get("id") for row in saved_rows if row.get("id")]
        if len(chunk_ids) != len(saved_rows):
            raise ValueError("Two-tier chunk insert missing chunk IDs")

        chunk_count = len(saved_rows)
        job_step_message = f"Created {chunk_count} two-tier chunks"

        # Update job progress after chunk creation
        self.supabase.update_job(job_id, {
            "progress": 50,
            "current_step": job_step_message
        })

        logger.info(f"{job_step_message} for document {document_id}")

        result_metadata: Dict[str, Any] = {}
        if isinstance(parse_result.get("metadata"), dict):
            result_metadata.update(parse_result["metadata"])
        result_metadata["chunking_method"] = chunking_method
        result_metadata["chunk_count_total"] = chunk_count
        result_metadata["semantic_chunk_count"] = semantic_count

        # Do NOT include raw images in Celery results; they can be large/non-serializable
        return {
            "document_id": document_id,
            "chunk_ids": chunk_ids,
            "chunk_count": chunk_count,
            "metadata": result_metadata,
        }

    except Exception as e:
        logger.error(f"Document chunking failed: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="generate_embeddings",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 10},
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True
)
def generate_embeddings(self, chunk_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Generate embeddings for chunks with automatic retry"""

    document_id = chunk_result["document_id"]
    chunk_ids = chunk_result["chunk_ids"]

    logger.info(f"Generating embeddings for {len(chunk_ids)} chunks (attempt {self.request.retries + 1})")

    try:
        self.supabase.update_job(job_id, {
            "progress": 60,
            "current_step": "Generating embeddings"
        })

        chunks = self.supabase.get_document_chunks(document_id)

        embedding_results: List[EmbeddingResult] = []
        if settings.openai_api_key:
            try:
                embedding_service = EmbeddingService()

                async def _embed_chunks() -> List[EmbeddingResult]:
                    texts = [chunk.chunk_text for chunk in chunks]
                    return await embedding_service.embed_batch(texts, batch_size=100)

                embedding_results = asyncio.run(_embed_chunks())
                logger.info("Generated %d embedding vectors via EmbeddingService", len(embedding_results))
            except Exception as embed_err:
                logger.error("Failed to generate embeddings via EmbeddingService: %s", embed_err)
                embedding_results = []
        else:
            logger.warning("OPENAI_API_KEY not configured; skipping embedding vector generation")

        now_iso = datetime.utcnow().isoformat()
        embeddings_generated = 0

        for idx, chunk in enumerate(chunks):
            updates: Dict[str, Any] = {
                "embedding_id": chunk.embedding_id or str(uuid4())
            }

            if embedding_results and idx < len(embedding_results):
                result = embedding_results[idx]
                updates.update({
                    "embedding_vector": result.embedding,
                    "embedding_model": result.model,
                    "embedding_dimensions": result.dimensions,
                    "embedded_at": now_iso
                })
                embeddings_generated += 1
            else:
                updates["embedding_model"] = chunk.embedding_model or "text-embedding-3-small"

            self.supabase.update_chunk(chunk.id, updates)

        self.supabase.update_job(job_id, {
            "progress": 75,
            "current_step": f"Generated {embeddings_generated} embeddings"
        })

        logger.info(
            "Prepared %d embeddings (%d vectors generated) for document %s",
            len(chunks),
            embeddings_generated,
            document_id,
        )

        # Ensure we do not propagate heavy/non-serializable fields
        sanitized = {k: v for k, v in (chunk_result or {}).items() if k != "images"}
        return {
            **sanitized,
            "embeddings_generated": embeddings_generated,
        }

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise


@celery_app.task(

    bind=True,

    base=DocumentTask,

    name="extract_entities",

    autoretry_for=(Exception,),

    retry_kwargs={'max_retries': 3, 'countdown': 15},

    retry_backoff=True,

    retry_backoff_max=300

)

def extract_entities(self, embedding_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:

    """Extract entities from document with automatic retry"""

    document_id = str(embedding_result["document_id"])



    logger.info(f"Extracting entities for document {document_id} (attempt {self.request.retries + 1})")



    try:

        # Update job progress

        self.supabase.update_job(job_id, {

            "progress": 85,

            "current_step": "Extracting entities"

        })



        chunks = self.supabase.get_document_chunks(document_id)



        document: Optional[Document] = None

        try:

            document = self.supabase.get_document(document_id)

        except Exception as doc_err:

            logger.warning(f"Unable to fetch document {document_id} for metadata enrichment: {doc_err}")



        pipeline_version = (settings.entity_pipeline_version or "v2").lower()

        if pipeline_version == "v1":
            raise RuntimeError(
                "Legacy entity pipeline v1 has been removed. "
                "Please use ENTITY_PIPELINE_VERSION=v2 (default)."
            )



            self.supabase.update_job(job_id, {

                "progress": 95,

                "current_step": f"Extracted {len(entities)} entities and {len(relationships)} relationships (legacy)"

            })



            return {

                **embedding_result,

                "entities_extracted": len(entities),

                "relationships_extracted": len(relationships)

            }



        return _extract_entities_v2(

            task=self,

            document_id=document_id,

            job_id=job_id,

            chunks=chunks,

            embedding_result=embedding_result,

            document=document,

        )



    except Exception as e:

        logger.error(f"Entity extraction failed: {str(e)}")

        raise






@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="finalize_document_processing",
)
def finalize_document_processing(self, extraction_result: Dict[str, Any], job_id: str, document_id: str) -> Dict[str, Any]:
    """Finalize processing by marking document and job complete."""
    logger.info(f"Finalizing document processing for {document_id} (job {job_id})")

    job = None
    try:
        self.supabase.update_document_status(str(document_id), DocumentState.PENDING_REVIEW)

        job = self.supabase.get_job(str(job_id))
        if job:
            job.complete()
            self.supabase.update_job(str(job_id), job.to_supabase_dict())
        else:
            logger.warning(f"Processing job {job_id} not found during finalization")

        return {
            "document_id": str(document_id),
            "job_id": str(job_id),
            "status": "completed",
            "result": extraction_result
        }

    except Exception as e:
        logger.error(f"Finalization failed for document {document_id}: {e}")

        try:
            self.supabase.update_document_status(str(document_id), DocumentState.FAILED, error=str(e))
        except Exception:
            logger.exception("Failed to mark document as FAILED during finalization")

        try:
            if job:
                job.fail(str(e))
                self.supabase.update_job(str(job_id), job.to_supabase_dict())
        except Exception:
            logger.exception("Failed to mark job as FAILED during finalization")

        raise
def _extract_entities_v2(

    task: DocumentTask,

    document_id: str,

    job_id: str,

    chunks: List[Chunk],

    embedding_result: Dict[str, Any],

    document: Optional[Document],

) -> Dict[str, Any]:

    supabase = task.supabase



    supabase.update_job(job_id, {

        "progress": 88,

        "current_step": "Extracting entity mentions (v2 pipeline)"

    })



    deleted_mentions = supabase.delete_document_entity_mentions(document_id)

    if deleted_mentions:

        logger.info(f"Removed {deleted_mentions} previous entity mentions for document {document_id}")



    deleted_relationships = supabase.delete_document_relationships(document_id)

    if deleted_relationships:

        logger.info(f"Removed {deleted_relationships} previous canonical relationships for document {document_id}")



    extraction_run_id = supabase.create_extraction_run(

        document_id=document_id,

        pipeline_version="v2",

        model="gpt-4o-mini",

    )



    chunk_inputs = [

        ChunkInput(

            id=chunk.id,

            document_id=chunk.document_id,

            text=chunk.chunk_text or "",

        )

        for chunk in chunks

    ]



    def _normalize_type(type_value: str | None) -> str:

        return (type_value or "CONCEPT").strip().upper()



    def _canonical_key(name: str, type_value: str | None) -> tuple[str, str] | None:

        label = (name or "").strip()

        if not label:

            return None

        return (label.lower(), _normalize_type(type_value))



    def _context_window(text: str, start: int, end: int, mention_text: str, window: int = 160) -> str:

        if not text:

            return ""

        try:

            start_idx = max(0, int(start))

        except Exception:

            start_idx = 0

        try:

            end_idx = int(end)

        except Exception:

            end_idx = start_idx + len(mention_text or "")

        if end_idx <= start_idx:

            end_idx = min(len(text), start_idx + max(len(mention_text or ""), 1))

        start_idx = max(0, start_idx - window)

        end_idx = min(len(text), end_idx + window)

        snippet = text[start_idx:end_idx]

        return " ".join(snippet.split())



    mention_batches = run_extract_mentions(chunk_inputs) if chunk_inputs else []

    mentions: List[EntityMention] = []

    evidence_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)



    for chunk, mention_list in zip(chunks, mention_batches):

        if not mention_list:

            continue

        chunk_text = chunk.chunk_text or ""

        strategy_value = chunk.chunking_strategy.value if hasattr(chunk.chunking_strategy, "value") else str(chunk.chunking_strategy)

        for mention in mention_list:

            mention.document_id = document_id

            mention.chunk_id = chunk.id

            mention.type = _normalize_type(getattr(mention, "type", None))

            start_offset = getattr(mention, "start_offset", 0) or 0

            end_offset = getattr(mention, "end_offset", 0) or 0

            mention.context = _context_window(chunk_text, start_offset, end_offset, mention.text or "")

            attributes = dict(getattr(mention, "attributes", {}) or {})

            attributes.update({

                "chunk_id": chunk.id,

                "chunk_index": chunk.chunk_index,

                "chunking_strategy": strategy_value,

            })

            if chunk.metadata:

                for key in ("section", "title", "heading", "subheading", "page", "page_number"):

                    if key in chunk.metadata and chunk.metadata[key] is not None:

                        attributes.setdefault(key, chunk.metadata[key])

            mention.attributes = attributes

            mentions.append(mention)



    mention_count = len(mentions)

    logger.info(f"V2 pipeline extracted {mention_count} raw mentions for document {document_id}")



    seen_keys: set[tuple[str, str]] = set()

    canonical_candidates: List[CanonicalEntity] = []

    for mention in mentions:

        key = _canonical_key(mention.text, mention.type)

        if not key or key in seen_keys:

            continue

        seen_keys.add(key)

        canonical_candidates.append(

            CanonicalEntity(

                name=(mention.text or "").strip(),

                type=key[1],

                metadata={"seed_document_id": document_id},

            )

        )



    canonical_map = supabase.upsert_canonical_entities_map(canonical_candidates) if canonical_candidates else {}

    canonical_ids: set[str] = set()

    for mention in mentions:

        key = _canonical_key(mention.text, mention.type)

        if not key:

            continue

        canonical_id = canonical_map.get(key) or canonical_map.get((key[0], key[1].lower()))

        if canonical_id:

            mention.canonical_entity_id = canonical_id

            mention.canonicalization_score = 1.0

            canonical_ids.add(canonical_id)

            evidence_map[canonical_id].append({

                "mention": mention.text,

                "context": mention.context,

                "chunk_index": mention.attributes.get("chunk_index") if mention.attributes else None,

                "chunk_id": mention.chunk_id,

                "confidence": mention.confidence,

                "attributes": mention.attributes,

            })



    canonical_id_list = sorted(canonical_ids)

    canonical_count = len(canonical_id_list)

    logger.info(f"Document {document_id} has {canonical_count} canonical entities after normalization")



    supabase.update_job(job_id, {

        "progress": 90,

        "current_step": f"Persisting {mention_count} mentions / {canonical_count} canonical entities"

    })



    inserted_mentions = supabase.insert_entity_mentions(mentions, extraction_run_id=extraction_run_id) if mentions else 0

    logger.info(f"Inserted {inserted_mentions} entity mentions for document {document_id}")



    canonical_entities_rows = supabase.get_canonical_entities_by_ids(canonical_id_list) if canonical_id_list else []

    existing_metadata_map: Dict[str, Dict[str, Any]] = {

        row["id"]: row.get("metadata") or {}

        for row in canonical_entities_rows

        if row.get("id")

    }



    description_updates: Dict[str, str] = {}

    if canonical_entities_rows:

        try:

            description_service = CanonicalEntityDescriptionService()

            description_updates = description_service.generate_descriptions(canonical_entities_rows, evidence_map)

        except Exception as desc_err:

            logger.warning(f"Failed to refresh canonical descriptions for document {document_id}: {desc_err}")

            description_updates = {}



    name_to_canonical: Dict[str, str] = {}

    for row in canonical_entities_rows:

        cid = row.get("id")

        if not cid:

            continue

        name = (row.get("name") or "").strip().lower()

        if name:

            name_to_canonical.setdefault(name, cid)

            name_to_canonical.setdefault(name.replace(" ", ""), cid)

        aliases = row.get("aliases") or []

        if isinstance(aliases, list):

            for alias in aliases:

                alias_key = (alias or "").strip().lower()

                if alias_key:

                    name_to_canonical.setdefault(alias_key, cid)

                    name_to_canonical.setdefault(alias_key.replace(" ", ""), cid)



    for mention in mentions:

        if mention.canonical_entity_id:

            key = (mention.text or "").strip().lower()

            if key:

                name_to_canonical.setdefault(key, mention.canonical_entity_id)

                name_to_canonical.setdefault(key.replace(" ", ""), mention.canonical_entity_id)



    full_text = "\n\n".join(chunk.chunk_text or "" for chunk in chunks if chunk.chunk_text)



    entity_inputs: List[Dict[str, Any]] = []

    for row in canonical_entities_rows:

        cid = row.get("id")

        if not cid:

            continue

        # Collect mention contexts for this entity (up to 3 examples)
        mention_contexts = []
        for mention in evidence_map.get(cid, [])[:3]:
            if hasattr(mention, 'context') and mention.context:
                mention_contexts.append(mention.context)

        entity_inputs.append({

            "id": cid,

            "name": row.get("name"),

            "type": row.get("type"),

            "count": len(evidence_map.get(cid, [])),

            "aliases": row.get("aliases") or [],

            "metadata": row.get("metadata") or {},

            "contexts": mention_contexts,  # Add mention contexts

        })



    document_metadata_payload: Dict[str, Any] = {

        "id": getattr(document, "id", document_id) if document else document_id,

        "title": getattr(document, "name", None) if document else None,

        "type": None,

        "department": None,

        "security_level": getattr(document, "security_level", None) if document else None,

    }

    meta_source = getattr(document, "metadata", {}) if document else {}

    if isinstance(meta_source, dict):

        document_metadata_payload["type"] = meta_source.get("type") or meta_source.get("document_type")

        document_metadata_payload["department"] = meta_source.get("department")



    supabase.update_job(job_id, {

        "progress": 92,

        "current_step": f"Canonicalized {canonical_count} entities; inferring relationships"

    })



    relationships_raw = []

    # DEBUG: Log condition values for relationship extraction
    logger.warning(f"[REL-CHECK] entity_inputs length: {len(entity_inputs)}, full_text length: {len(full_text)}, full_text.strip() length: {len(full_text.strip())}")
    logger.warning(f"[REL-CHECK] Condition check: entity_inputs={bool(entity_inputs)}, full_text.strip()={bool(full_text.strip())}")

    if entity_inputs and full_text.strip():

        logger.warning(f"[REL-START] Starting relationship extraction with {len(entity_inputs)} entities")

        relationship_extractor = RelationshipExtractor()

        try:

            relationships_raw = asyncio.run(

                relationship_extractor.extract_relationships(

                    text=full_text,

                    entities=entity_inputs,

                    document_metadata=document_metadata_payload,

                )

            )

        except RuntimeError:

            loop = asyncio.new_event_loop()

            try:

                asyncio.set_event_loop(loop)

                relationships_raw = loop.run_until_complete(

                    relationship_extractor.extract_relationships(

                        text=full_text,

                        entities=entity_inputs,

                        document_metadata=document_metadata_payload,

                    )

                )

            finally:

                asyncio.set_event_loop(None)

                loop.close()

        except Exception as rel_err:

            logger.warning(f"[REL-ERROR] Relationship extraction failed for document {document_id}: {rel_err}")

            relationships_raw = []

    else:

        logger.warning(f"[REL-SKIP] Skipping relationship extraction for document {document_id} - entity_inputs: {len(entity_inputs)}, full_text: {len(full_text)}")

    logger.warning(f"[REL-RESULT] Relationship extraction complete: {len(relationships_raw)} raw relationships returned")



    def _resolve_canonical(name: str | None) -> str | None:

        if not name:

            return None

        key = name.strip().lower()

        if not key:

            return None

        if key in name_to_canonical:

            return name_to_canonical[key]

        compact = key.replace(" ", "")

        return name_to_canonical.get(compact)



    canonical_relationships: List[CanonicalRelationship] = []

    if relationships_raw:
        logger.warning(f"[REL-CANON] Processing {len(relationships_raw)} relationships. Available canonical mapping keys: {list(name_to_canonical.keys())[:20]}")
        dedup_map: Dict[tuple[str, str, str], CanonicalRelationship] = {}

        for rel in relationships_raw:

            raw_type_obj = getattr(rel, "relationship_type", None)

            if raw_type_obj and hasattr(raw_type_obj, "value"):

                raw_label = raw_type_obj.value[0] if isinstance(raw_type_obj.value, (tuple, list)) else raw_type_obj.value

            else:

                raw_label = str(raw_type_obj or "")

            canonical_type = canonicalize_relationship_type(raw_label)

            # The relationship extractor returns entity IDs in source_entity/target_entity fields
            # These are already canonical UUIDs, not names
            source_id = getattr(rel, "source_entity", None)
            target_id = getattr(rel, "target_entity", None)

            # Get names from properties for logging (stored during extraction)
            props = getattr(rel, "properties", None)
            additional_props = getattr(props, "additional_properties", {}) if props else {}
            source_name_log = additional_props.get("source_name", source_id)
            target_name_log = additional_props.get("target_name", target_id)

            # Validate IDs exist and are valid UUIDs (skip invalid entries)
            if not source_id or not target_id or source_id == target_id:
                logger.warning(f"[REL-FILTER] Skipping relationship: source_id='{source_id}' (name={source_name_log}), target_id='{target_id}' (name={target_name_log}), type={canonical_type}, reason=missing_or_equal")
                continue

            # Validate UUIDs - skip if either is not a valid UUID format
            try:
                from uuid import UUID
                UUID(str(source_id))
                UUID(str(target_id))
            except (ValueError, AttributeError) as e:
                logger.warning(f"[REL-FILTER] Skipping relationship: source_id='{source_id}' (name={source_name_log}), target_id='{target_id}' (name={target_name_log}), type={canonical_type}, reason=invalid_uuid")
                continue

            properties = getattr(rel, "properties", None)

            confidence = getattr(properties, "confidence", 0.0) if properties else 0.0

            extraction_method = getattr(properties, "extraction_method", None) if properties else None

            source_text = getattr(properties, "source_text", None) if properties else None

            additional_properties = getattr(properties, "additional_properties", None) if properties else None

            extracted_at = None

            if properties and getattr(properties, "extracted_at", None):

                try:

                    extracted_at = properties.extracted_at.isoformat()

                except Exception:

                    extracted_at = None

            metadata = {

                "document_id": document_id,

                "source_name": source_name_log,  # Use name from properties (stored during extraction)

                "source_type": getattr(rel, "source_type", None),

                "target_name": target_name_log,  # Use name from properties (stored during extraction)

                "target_type": getattr(rel, "target_type", None),

                "raw_relationship_type": raw_label,

                "extraction_method": extraction_method,

            }

            if source_text:

                metadata["source_text"] = source_text

            if additional_properties:

                metadata["attributes"] = additional_properties

            if extracted_at:

                metadata["extracted_at"] = extracted_at

            metadata["confidence"] = float(confidence or 0.0)

            key = (source_id, target_id, canonical_type)

            candidate = CanonicalRelationship(

                source_entity_id=source_id,

                target_entity_id=target_id,

                relationship_type=canonical_type,

                confidence_score=float(confidence or 0.0),

                metadata=metadata,

            )

            existing = dedup_map.get(key)

            if not existing or candidate.confidence_score >= existing.confidence_score:

                dedup_map[key] = candidate

        canonical_relationships = list(dedup_map.values())



    relationships_inserted = 0

    if canonical_relationships:

        relationships_inserted = supabase.insert_canonical_relationships(canonical_relationships)

        logger.info(f"Stored {relationships_inserted} canonical relationships for document {document_id}")

    else:

        logger.info(f"No canonical relationships inferred for document {document_id}")



    supabase.update_job(job_id, {

        "progress": 94,

        "current_step": f"Finalising entity metrics ({canonical_count} canonical / {relationships_inserted} relationships)"

    })



    try:

        supabase.refresh_canonical_entity_metrics(

            canonical_id_list,

            description_updates=description_updates,

            existing_metadata=existing_metadata_map,

        )

    except Exception as metrics_err:

        logger.warning(f"Failed to refresh canonical entity metrics for document {document_id}: {metrics_err}")



    chunk_count = embedding_result.get("chunk_count") or len(chunks)

    metadata_updates: Dict[str, Any]

    if document and isinstance(document.metadata, dict):

        metadata_updates = dict(document.metadata)

    else:

        metadata_updates = {}

    entity_metrics = dict(metadata_updates.get("entity_metrics") or {})

    entity_metrics.update({

        "mention_count": mention_count,

        "canonical_count": canonical_count,

        "relationship_count": relationships_inserted,

        "last_extraction_run_id": extraction_run_id,

        "last_extracted_at": datetime.utcnow().isoformat(),

    })

    metadata_updates["entity_metrics"] = entity_metrics



    document_updates = {

        "entity_count": canonical_count,

        "chunk_count": chunk_count,

        "metadata": metadata_updates,

        "processed_at": datetime.utcnow().isoformat(),

    }

    try:

        supabase.update_document(document_id, document_updates)

    except Exception as doc_update_err:

        logger.warning(f"Failed to update document {document_id} with entity metrics: {doc_update_err}")



    supabase.complete_extraction_run(

        extraction_run_id,

        mentions=mention_count,

        canonical=canonical_count,

        relationships=relationships_inserted,

    )



    supabase.update_job(job_id, {

        "progress": 95,

        "current_step": f"Entities ready for review ({canonical_count} canonical / {relationships_inserted} relationships)"

    })



    logger.info(

        "Entity extraction v2 complete for %s: %d mentions -> %d canonical, %d relationships",

        document_id,

        mention_count,

        canonical_count,

        relationships_inserted,

    )



    result = dict(embedding_result)

    result.update({

        "entities_extracted": canonical_count,

        "relationships_extracted": relationships_inserted,

        "mentions_extracted": mention_count,

        "extraction_run_id": extraction_run_id,

    })

    return result





@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="publish_approved_document",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 120},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True
)
def publish_approved_document(self, document_id: str) -> Dict[str, Any]:
    """Publish an approved document to downstream vector and graph stores."""
    logger.info("Publishing approved document %s", document_id)

    supabase = self.supabase
    document = supabase.get_document(document_id)

    if not document:
        logger.error("Document %s not found while attempting to publish", document_id)
        return {"document_id": document_id, "status": "not_found"}

    attempt = (document.publish_attempts or 0) + 1

    def _run_async(coro):
        try:
            return asyncio.run(coro)
        except RuntimeError as runtime_err:
            if "asyncio.run" in str(runtime_err):
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(coro)
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
            raise

    def _ensure_embeddings(chunks: List[Chunk]) -> int:
        missing = [chunk for chunk in chunks if not getattr(chunk, 'embedding_vector', None)]
        if not missing:
            return 0
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not configured; cannot backfill embeddings for %s", document_id)
            return 0
        try:
            embedding_service = EmbeddingService()

            async def _embed_missing() -> List[EmbeddingResult]:
                texts = [chunk.chunk_text for chunk in missing]
                return await embedding_service.embed_batch(texts, batch_size=100)

            results = _run_async(_embed_missing())
            now_iso = datetime.utcnow().isoformat()
            generated = 0
            for chunk, result in zip(missing, results):
                supabase.update_chunk(chunk.id, {
                    "embedding_vector": result.embedding,
                    "embedding_model": result.model,
                    "embedding_dimensions": result.dimensions,
                    "embedded_at": now_iso,
                })
                chunk.embedding_vector = result.embedding
                chunk.embedding_model = result.model
                chunk.embedding_dimensions = result.dimensions
                generated += 1
            return generated
        except Exception as embed_err:
            logger.error("Failed to backfill embeddings for %s: %s", document_id, embed_err)
            raise

    publish_updates = {
        "status": DocumentState.PUBLISHING.value,
        "publish_attempts": attempt,
        "last_publish_error": None,
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        supabase.update_document(document_id, publish_updates)
    except Exception as update_err:
        logger.warning("Failed to set publishing state for %s: %s", document_id, update_err)

    try:
        chunks = supabase.get_document_chunks(document_id)
        embeddings_generated = _ensure_embeddings(chunks) if chunks else 0

        qdrant_points = 0
        if settings.qdrant_url:
            try:
                qdrant_service = QdrantService()
                qdrant_points = _run_async(qdrant_service.store_document_embeddings(document_id, chunks or []))
            except Exception as qdrant_err:
                logger.error("Failed to publish embeddings to Qdrant for %s: %s", document_id, qdrant_err)
                raise
        else:
            logger.info("Qdrant is not configured; skipping vector publish for %s", document_id)

        graph_result: Dict[str, int] = {"entities": 0, "relationships": 0}
        if settings.neo4j_uri:
            try:
                neo4j_service = Neo4jService()
                graph_result = _run_async(neo4j_service.store_document_graph(document_id))
            except Exception as neo4j_err:
                logger.error("Failed to publish graph data to Neo4j for %s: %s", document_id, neo4j_err)
                raise
        else:
            logger.info("Neo4j is not configured; skipping graph publish for %s", document_id)

        metadata = dict(document.metadata or {})
        publishing_info = dict(metadata.get("publishing") or {})
        published_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        publishing_info.update({
            "last_published_at": published_at,
            "qdrant_points": qdrant_points,
            "neo4j_entities": graph_result.get("entities", 0),
            "neo4j_relationships": graph_result.get("relationships", 0),
            "embeddings_generated": embeddings_generated,
        })
        metadata["publishing"] = publishing_info

        supabase.update_document(document_id, {
            "status": DocumentState.PUBLISHED.value,
            "published_at": published_at,
            "metadata": metadata,
            "last_publish_error": None,
            "updated_at": datetime.utcnow().isoformat(),
        })

        logger.info(
            "Published document %s to Qdrant (points=%d) and Neo4j (entities=%d, relationships=%d)",
            document_id,
            qdrant_points,
            graph_result.get("entities", 0),
            graph_result.get("relationships", 0),
        )

        return {
            "document_id": document_id,
            "qdrant_points": qdrant_points,
            "neo4j_entities": graph_result.get("entities", 0),
            "neo4j_relationships": graph_result.get("relationships", 0),
            "embeddings_generated": embeddings_generated,
        }

    except Exception as publish_error:
        logger.error("Publishing pipeline failed for %s: %s", document_id, publish_error)
        supabase.update_document(document_id, {
            "status": DocumentState.PUBLISH_FAILED.value,
            "last_publish_error": str(publish_error),
            "updated_at": datetime.utcnow().isoformat(),
        })
        raise
