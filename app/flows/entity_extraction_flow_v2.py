"""
Entity Extraction Pipeline v2 - CocoIndex Flow (skeleton)

Implements the declarative dataflow pattern:
  documents -> chunk text -> extract mentions -> canonicalize -> validate -> export

Exports:
  - Mentions and canonical entities to Postgres (Supabase)
  - Canonical graph projection events to outbox for Neo4j sync

This file scaffolds the flow; extraction and canonicalization functions can
be incrementally filled without breaking declarative structure.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import List

import cocoindex as cx
from cocoindex import FlowBuilder, DataScope
from cocoindex.functions import ExtractByLlm
from cocoindex.llm import LlmSpec, LlmApiType
from cocoindex.targets import Postgres

from app.models.entity_v2 import EntityMention, CanonicalEntity
from app.config import settings


# LLM instruction for mention extraction with offsets
MENTION_EXTRACTION_INSTRUCTION = (
    "Extract high-quality entity mentions from the provided text. "
    "Return only noun-phrase entities with document-relevant meaning. "
    "Provide character offsets on the normalized input text."
)


@cx.flow_def(name="EntityExtractionV2")
def entity_extraction_flow(flow_builder: FlowBuilder, data_scope: DataScope):
    # Sources are assumed to exist in the encompassing ingestion flow.
    # Here we expect a prepared dataset of chunks with text and ids.
    chunks = data_scope["chunks"]  # Provided by upstream flow

    # Collectors for outputs
    mentions_out = data_scope.add_collector()
    canonical_out = data_scope.add_collector()
    outbox_out = data_scope.add_collector()  # events for graph projection

    with chunks.row() as ch:
        # Step 1: Extract mentions using structured LLM output
        ch["mentions"] = ch["text"].transform(
            ExtractByLlm(
                llm_spec=LlmSpec(
                    api_type=LlmApiType.OpenAI,
                    model="gpt-4o-mini",
                    api_key=settings.openai_api_key,
                ),
                output_type=List[EntityMention],
                instruction=MENTION_EXTRACTION_INSTRUCTION,
            )
        )

        # Step 2: Canonicalize mentions (type-aware normalization)
        @cx.op.function()
        def canonicalize(mentions: List[EntityMention]) -> List[CanonicalEntity]:
            # Simple baseline: unique by normalized (name,type)
            seen: set[tuple[str, str]] = set()
            result: list[CanonicalEntity] = []
            for m in mentions:
                key = (m.text.strip().lower(), m.type)
                if key not in seen:
                    seen.add(key)
                    result.append(CanonicalEntity(name=m.text.strip(), type=m.type))
            return result

        ch["canonical"] = ch["mentions"].transform(canonicalize)

        # Step 3: Emit rows for persistence
        # Mentions
        for m in ch["mentions"]:
            mentions_out.collect(
                document_id=ch["document_id"],
                chunk_id=ch["id"],
                **asdict(m),
            )

        # Canonical entities
        for ce in ch["canonical"]:
            canonical_out.collect(
                **asdict(ce)
            )

    # Exports: write to Postgres (Supabase)
    postgres_conn = {
        "url": settings.database_url or settings.supabase_url,
        "api_key": settings.supabase_key,
    }

    mentions_out.export(
        "entity_mentions",
        Postgres(connection=postgres_conn, table_name="entity_mentions", schema="public"),
    )

    canonical_out.export(
        "canonical_entities",
        Postgres(connection=postgres_conn, table_name="canonical_entities", schema="public"),
    )

