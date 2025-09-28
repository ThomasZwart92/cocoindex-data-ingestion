"""Service to keep canonical entity descriptions up to date."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List

from app.services.llm_service import LLMProvider, LLMService

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_LENGTH = 300
MAX_EVIDENCE_SNIPPETS = 6


def _clean_text(value: str, *, max_length: int | None = None) -> str:
    """Normalize whitespace and optionally trim length."""
    text = " ".join((value or "").split())
    if max_length and len(text) > max_length:
        return text[: max_length - 3].rstrip() + "..."
    return text


def _format_attribute_fragment(attributes: Dict[str, Any]) -> str:
    """Format non-empty attribute dict into a compact string."""
    parts: List[str] = []
    for key, raw_value in attributes.items():
        if raw_value in (None, "", [], {}):
            continue
        value = raw_value
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value if item not in (None, ""))
        elif isinstance(value, dict):
            value = ", ".join(
                f"{inner_key}: {inner_val}"
                for inner_key, inner_val in value.items()
                if inner_val not in (None, "")
            )
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


class CanonicalEntityDescriptionService:
    """Generate concise canonical entity descriptions from mention evidence."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        *,
        max_evidence_snippets: int = MAX_EVIDENCE_SNIPPETS,
        max_concurrency: int = 4,
    ) -> None:
        self.llm_service = llm_service or LLMService()
        self.max_evidence_snippets = max_evidence_snippets
        self.max_concurrency = max(1, max_concurrency)

    def generate_descriptions(
        self,
        canonical_entities: Iterable[Dict[str, Any]],
        evidence_map: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        """Synchronously generate new descriptions for the supplied canonical entities."""
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            coroutine = self._generate_descriptions_async(list(canonical_entities), evidence_map)
            return loop.run_until_complete(coroutine)
        finally:
            try:
                loop.close()
            except Exception:  # pragma: no cover - defensive close
                pass
            if previous_loop is not None:
                asyncio.set_event_loop(previous_loop)
            else:
                asyncio.set_event_loop(None)

    async def _generate_descriptions_async(
        self,
        canonical_entities: List[Dict[str, Any]],
        evidence_map: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: Dict[str, str] = {}

        async def _process(entity: Dict[str, Any]) -> None:
            cid = entity.get("id")
            if not cid:
                return
            evidence = evidence_map.get(cid) or []
            metadata = entity.get("metadata") or {}
            previous_description = metadata.get("description") or ""

            formatted_evidence = self._prepare_evidence(evidence)
            if not formatted_evidence and not previous_description:
                return

            if not formatted_evidence:
                # No new evidence; keep the previous description as-is
                return

            async with semaphore:
                description = await self._call_llm(
                    name=entity.get("name"),
                    type_=entity.get("type"),
                    previous_description=previous_description,
                    evidence=formatted_evidence,
                )

            if description is None:
                return

            if description == previous_description:
                return

            results[cid] = description

        await asyncio.gather(*[_process(entity) for entity in canonical_entities])
        return results

    def _prepare_evidence(self, evidence_items: List[Dict[str, Any]]) -> List[str]:
        formatted: List[str] = []
        seen_snippets: set[str] = set()
        filtered = evidence_items[: self.max_evidence_snippets * 2]
        for item in filtered:
            mention = _clean_text(str(item.get("mention", "")))
            if not mention:
                continue
            context = _clean_text(str(item.get("context", "")), max_length=220)
            summary = _clean_text(str(item.get("summary", "")), max_length=220)
            attributes = item.get("attributes") or {}
            if isinstance(attributes, dict):
                attributes = {
                    key: value
                    for key, value in attributes.items()
                    if key not in {"chunk_index", "chunking_strategy", "chunk_level", "chunk_id"}
                }
            attr_text = _format_attribute_fragment(attributes) if isinstance(attributes, dict) else ""

            parts: List[str] = [f"Mention: {mention}"]
            if summary and summary != context:
                parts.append(f"Summary: {summary}")
            if context:
                parts.append(f"Context: {context}")
            if attr_text:
                parts.append(f"Attributes: {attr_text}")

            snippet = " | ".join(parts)
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            formatted.append(snippet)
            if len(formatted) >= self.max_evidence_snippets:
                break
        return formatted

    async def _call_llm(
        self,
        *,
        name: str | None,
        type_: str | None,
        previous_description: str,
        evidence: List[str],
    ) -> str | None:
        if not evidence:
            return None

        entity_label = _clean_text(f"{name or ''}".strip())
        entity_type = _clean_text(f"{type_ or ''}".strip())

        evidence_block = "\n".join(f"- {snippet}" for snippet in evidence)
        prev_block = previous_description.strip() or "(none)"

        system_prompt = (
            "You maintain factual one-paragraph descriptions of technical entities. "
            "Rely exclusively on the supplied evidence snippets and previously accepted description. "
            "If the evidence does not explicitly support a fact, do not include it. "
            "When nothing concrete is present, respond with 'Insufficient evidence.' instead of guessing. "
            "Keep the description under 280 characters and at most two sentences."
        )
        user_prompt = (
            f"Entity: {entity_label or 'Unknown'}"
            + (f" ({entity_type})" if entity_type else "")
            + "\n\n"
            "Previous description:\n"
            f"{prev_block}\n\n"
            "New evidence:\n"
            f"{evidence_block}\n\n"
            "Write an updated description that only contains facts directly stated in the evidence or previous description. "
            "Quote or closely paraphrase the evidence wording; never add domain knowledge or speculation. "
            "If the evidence lacks concrete facts, reply with 'Insufficient evidence.'" 
            "If no new facts exist beyond the previous description, return the previous description unchanged. "
            "Respond with 2-3 sentences at most, without adding headings, labels, or bullet markers."
        )

        try:
            response = await self.llm_service.call_with_fallback(
                prompt=user_prompt,
                system_prompt=system_prompt,
                primary_provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=250,
            )
        except Exception as exc:  # pragma: no cover - network safeguard
            logger.warning("Failed to update canonical entity description: %s", exc)
            return previous_description or None

        content = (response.content or "").strip()
        if content.startswith("```"):
            # Handle fenced responses that wrap the actual description
            content = content.split("\n", 1)[-1].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

        if not content:
            return previous_description or ""

        normalized = _clean_text(content, max_length=MAX_DESCRIPTION_LENGTH)
        lower_normalized = normalized.lower()
        if "description:" in lower_normalized:
            idx = lower_normalized.find("description:") + len("description:")
            normalized = normalized[idx:].strip()
            lower_normalized = normalized.lower()
        if lower_normalized.startswith("entity:"):
            normalized = normalized.split(":", 1)[-1].strip()
            lower_normalized = normalized.lower()
        if not normalized:
            return previous_description or ""
        lower_normalized = normalized.lower()
        guard_prefixes = (
            "insufficient evidence",
            "no evidence",
            "unknown.",
            "unknown",
            "uncertain",
        )
        if any(lower_normalized.startswith(prefix) for prefix in guard_prefixes):
            return previous_description or ""
        return normalized


__all__ = ["CanonicalEntityDescriptionService"]
