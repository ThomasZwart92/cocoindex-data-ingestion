"""







Entity Extraction Runner v2 (transform flow)















Provides a parameterized CocoIndex transform flow that takes a list of chunk







rows and returns extracted, quality-filtered mentions per chunk. This keeps







extraction declarative while allowing orchestration from Celery tasks.







"""







from __future__ import annotations















from dataclasses import dataclass







from typing import List















import asyncio







import json















try:







    import cocoindex as cx  # type: ignore







    from cocoindex import DataSlice  # type: ignore







except Exception:







    cx = None  # Fallback when cocoindex native engine is unavailable















    class DataSlice:  # type: ignore







        pass















from app.models.entity_v2 import EntityMention







from app.services.llm_service import LLMService, LLMProvider























@dataclass







class ChunkInput:







    id: str







    document_id: str







    text: str























MENTION_INSTRUCTION = (







    "Identify the most important entities in the text. Return a JSON array of objects with keys: "







    "text, type, start_offset, end_offset, confidence. Offsets must be character indexes on the exact input. "







    "The type MUST be one of [PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, COMPONENT, TECHNOLOGY, CHEMICAL, PROCEDURE, SPECIFICATION, SYSTEM, MEASUREMENT, PROBLEM, CONDITION, STATE, TOOL, MATERIAL, CONCEPT, EVENT]. "







    "Do not invent new type names (avoid labels like noun phrase). Choose the closest category and use CONCEPT if unsure. "







    "Skip pronouns, generic filler words (issue, problem, system, update), temporal words (yesterday), and sentence fragments. "







    "Prefer specific multi-word terms and allow well-known acronyms (API, USB, LCD). confidence should be between 0 and 1."







)























ALLOWED_ENTITY_TYPES = {







    'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'PRODUCT', 'COMPONENT', 'TECHNOLOGY',







    'CHEMICAL', 'PROCEDURE', 'SPECIFICATION', 'SYSTEM', 'MEASUREMENT', 'PROBLEM',







    'CONDITION', 'STATE', 'TOOL', 'MATERIAL', 'CONCEPT', 'EVENT'







}















TYPE_SYNONYMS = {







    'NOUN_PHRASE': 'CONCEPT',







    'NOUNPHRASE': 'CONCEPT',







    'PHRASE': 'CONCEPT',







    'ACTION': 'PROCEDURE',







    'PROCESS': 'PROCEDURE',







    'SYMPTOM': 'PROBLEM',







    'ISSUE': 'PROBLEM',







    'ERROR': 'PROBLEM',







    'FAILURE': 'PROBLEM',







    'MALFUNCTION': 'PROBLEM',







    'EQUIPMENT': 'TOOL',







    'SUPPLY': 'MATERIAL',







    'CONSUMABLE': 'MATERIAL',







}















KEYWORD_HINTS = {







    'MATERIAL': ['alcohol', 'solvent', 'lubricant', 'cloth', 'towel', 'microfiber', 'pad', 'wipes', 'wipe', 'lint-free', 'compound', 'chemical', 'cleaner'],







    'TOOL': ['screwdriver', 'brush', 'applicator', 'tool', 'software', 'application'],







    'COMPONENT': ['connector', 'cable', 'display', 'screen', 'module', 'assembly', 'sensor'],







    'PROBLEM': ['corrosion', 'flicker', 'failure', 'issue', 'fault', 'error'],







    'PROCEDURE': ['cleaning', 'wipe', 'install', 'inspection', 'calibration', 'step'],







    'TECHNOLOGY': ['protocol', 'platform', 'system', 'framework'],







    'CHEMICAL': ['acid', 'alcohol', 'solvent', 'adhesive'],







}























DEFAULT_ENTITY_TYPE = 'CONCEPT'























def normalize_entity_type(raw_type: str | None, mention_text: str = '') -> str:







    if raw_type:







        normalized = raw_type.strip().replace('-', '_').replace(' ', '_').upper()







        if normalized in ALLOWED_ENTITY_TYPES:







            return normalized







        if normalized in TYPE_SYNONYMS:







            return TYPE_SYNONYMS[normalized]







        lower = raw_type.strip().lower()







        if lower in TYPE_SYNONYMS:







            return TYPE_SYNONYMS[lower]







    text_lower = (mention_text or '').lower()







    for target, keywords in KEYWORD_HINTS.items():







        for keyword in keywords:







            if keyword in text_lower:







                return target







    return DEFAULT_ENTITY_TYPE























def normalize_confidence(value) -> float:







    try:







        return max(0.0, min(float(value), 1.0))







    except Exception:







        return 0.5























def _filter_quality_mentions(mentions: List[EntityMention]) -> List[EntityMention]:







    from app.utils.entity_quality import EntityQualityValidator







    filtered: List[EntityMention] = []







    for m in mentions:







        if not m or not m.text:







            continue







        t = m.text.strip()







        if not t:







            continue







        if len(t) < 3 and t.upper() != t:







            continue







        if len(t.split()) == 1 and t.lower() in EntityQualityValidator.GENERIC_STOPWORDS:







            continue







        m.type = normalize_entity_type(m.type, t)
        # Extra bias: treat cloth/towel/pad/wipe terms as MATERIAL unless strong tool counter-signal
        _lx = (t or '').lower()
        if (
            m.type != 'MATERIAL'
            and any(w in _lx for w in ['cloth','towel','paper towel','shop towel','microfiber','micro-fiber','micro fibre','wipe','wipes','wiping','pad','pads','lint-free','lint free'])
            and not any(w in _lx for w in ['applicator tool','applicator-tip','applicator tip','specialized tool','equipment'])
        ):
            m.type = 'MATERIAL'







        is_valid, _ = EntityQualityValidator.is_valid_entity(t, m.type)







        if not is_valid:







            continue







        # Align with doc-wide validator thresholds (>=0.25)







        if m.confidence is not None and m.confidence < 0.25:







            continue







        filtered.append(m)







    return filtered































_DASH_TRANSLATION = str.maketrans({

    "\u2010": "-",

    "\u2011": "-",

    "\u2012": "-",

    "\u2013": "-",

    "\u2014": "-",

    "\u2212": "-",

    "\u00A0": " ",

})





def _normalize_for_alignment(value: str) -> str:

    if not value:

        return ''

    return value.translate(_DASH_TRANSLATION)





def _extract_mentions_llm(text: str) -> List[EntityMention]:







    """Use LLMService with OpenAI GPT-5 (reasoning_effort=minimal) and fallback to Gemini 2.5 Pro."""







    import logging







    logger = logging.getLogger(__name__)







    logger.info(f"V2 LLM extraction called with text length: {len(text)}")







    







    service = LLMService()







    system = (







        "You extract entity mentions as strict JSON. Only return an array of objects with keys: "







        "text, type, start_offset, end_offset, confidence."







    )















    async def _call():







        return await service.call_with_fallback(







            prompt=f"{MENTION_INSTRUCTION}\n\nTEXT:\n{text}",







            primary_provider=LLMProvider.OPENAI,







            system_prompt=system,







            model="gpt-5",  # Use standard gpt-5 model







            temperature=0.0,







            max_completion_tokens=4000,







            reasoning_effort="minimal",







            timeout=120  # Explicit timeout to match ThreadPoolExecutor timeout







        )















    # Run async call in a separate thread with its own loop to avoid nested loop issues







    from concurrent.futures import ThreadPoolExecutor







    def _runner():







        return asyncio.run(_call())







    try:







        with ThreadPoolExecutor(max_workers=1) as ex:







            resp = ex.submit(_runner).result(timeout=120)







    except Exception as e:







        logger.error(f"LLM mention extraction failed: {e}")







        return []















    content = resp.content.strip()







    if content.startswith("```json"):







        content = content[7:].strip()







        if content.endswith("```"):







            content = content[:-3].strip()







    elif content.startswith("```") and content.endswith("```"):







        content = content[3:-3].strip()















    data = []







    try:







        data = json.loads(content)







        if not isinstance(data, list):







            data = []







    except Exception:







        data = []















    logger.warning(f"V2 LLM response parsed, got {len(data)} items")







    







    result: List[EntityMention] = []







    for item in data:







        try:







            text_value = str(item.get("text", "")).strip()







            type_value = normalize_entity_type(item.get("type"), text_value)







            confidence_value = normalize_confidence(item.get("confidence", 0.5))







            result.append(







                EntityMention(







                    text=text_value,







                    type=type_value,







                    start_offset=int(item.get("start_offset", 0)),







                    end_offset=int(item.get("end_offset", 0)),







                    confidence=confidence_value,







                )







            )







        except Exception:







            continue







    return result























def run_extract_mentions(chunks: list[ChunkInput]) -> list[list[EntityMention]]:







    """Extract and filter entity mentions from chunks."""







    import logging







    logger = logging.getLogger(__name__)







    logger.info(f"=== V2 EXTRACTION CALLED with {len(chunks)} chunks ===")







    







    result = []







    for chunk in chunks:







        # Extract mentions from each chunk







        mentions_raw = _extract_mentions_llm(chunk.text)







        # Apply quality filtering







        mentions_filtered = _filter_quality_mentions(mentions_raw)







        # Validate and adjust offsets when needed







        text = chunk.text or ""







        adjusted: List[EntityMention] = []







        normalized_text = _normalize_for_alignment(text)







        normalized_text_lower = normalized_text.lower()







        for m in mentions_filtered:







            try:







                valid_range = (







                    isinstance(m.start_offset, int)







                    and isinstance(m.end_offset, int)







                    and 0 <= m.start_offset < m.end_offset <= len(text)







                )







                matches_text = False







                mention_text = m.text or ''







                sanitized_mention = _normalize_for_alignment(mention_text)







                if valid_range:







                    segment = text[m.start_offset:m.end_offset]







                    if segment == mention_text:







                        matches_text = True







                    else:







                        matches_text = (







                            _normalize_for_alignment(segment) == sanitized_mention







                        )







                if not matches_text:







                    if not sanitized_mention:







                        continue







                    pos = normalized_text.find(sanitized_mention)







                    if pos == -1:







                        pos = normalized_text_lower.find(sanitized_mention.lower())







                    if pos != -1:







                        m.start_offset = pos







                        m.end_offset = pos + len(mention_text)







                        adjusted.append(m)







                        continue







                    continue







                adjusted.append(m)







            except Exception:







                continue







        mentions_filtered = adjusted







        logger.info(f"V2: Chunk {chunk.id}: {len(mentions_raw)} raw -> {len(mentions_filtered)} filtered")







        # Add chunk_id to each mention







        for mention in mentions_filtered:







            mention.chunk_id = chunk.id







            mention.document_id = chunk.document_id







        result.append(mentions_filtered)







    return result







