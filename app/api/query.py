"""
Query API Endpoints
Provides a streaming answer endpoint that performs hybrid search with optional
metadata filters (department, security tier) and generates an answer using a
selected LLM model, streaming text tokens to the client.
"""
from typing import Optional, Dict, Any, AsyncGenerator, List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import logging

from app.services.search_service import SearchService
from app.services.llm_service import LLMService, LLMProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])

search_service = SearchService()
llm_service = LLMService()


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    model: str = Field("gpt-4o-mini", description="Model id (e.g. gpt-4o, gpt-4o-mini, gpt-5, gemini-2.5-pro)")
    # Multi-select filters
    departments: Optional[List[str]] = Field(None, description="Departments to include (match metadata.department)")
    security_tiers: Optional[List[str]] = Field(None, description="Security tiers to include (metadata.security_level)")
    # Back-compat single-select (deprecated)
    department: Optional[str] = Field(None, description="Deprecated: single department")
    security_tier: Optional[str] = Field(None, description="Deprecated: single security tier")
    top_k: int = Field(6, ge=1, le=20, description="Number of context items")


def _filters_from_request(req: QueryRequest) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    # Qdrant payload uses nested metadata; allow dot-notation
    departments = req.departments or ([] if req.department is None else [req.department])
    tiers = req.security_tiers or ([] if req.security_tier is None else [req.security_tier])
    if departments:
        filters["metadata.department"] = departments
    if tiers:
        filters["metadata.security_level"] = tiers
    return filters


def _provider_from_model(model: str) -> LLMProvider:
    m = (model or "").lower()
    if m.startswith("gpt"):
        return LLMProvider.OPENAI
    if m.startswith("gemini"):
        return LLMProvider.GEMINI
    # default
    return LLMProvider.OPENAI


def _build_prompt(user_query: str, contexts: list[dict]) -> str:
    parts = [
        "You are a helpful assistant answering using the provided context.",
        "If the answer is not in the context, say you don't have enough information.",
        "Cite the document ids inline like [doc:ID] where relevant.",
        "\nContext:"
    ]
    for i, c in enumerate(contexts, 1):
        cid = c.get("metadata", {}).get("document_id") or c.get("metadata", {}).get("documentId") or c.get("metadata", {}).get("document_id")
        snippet = c.get("content") or c.get("metadata", {}).get("chunk_text") or ""
        parts.append(f"[{i}] (doc:{cid}) {snippet}")
    parts.append("\nUser question:")
    parts.append(user_query)
    parts.append("\nAnswer:")
    return "\n".join(parts)


async def _stream_openai_answer(prompt: str, model: str) -> AsyncGenerator[str, None]:
    """Stream tokens from OpenAI; fallback to full completion if streaming unsupported."""
    try:
        client = llm_service.openai_async
        if not client:
            raise ValueError("OpenAI client not configured")

        messages = [{"role": "user", "content": prompt}]
        # For GPT-5 we must use special params; streaming for GPT-5 via Chat Completions may be limited.
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            except Exception:
                # ignore non-delta events
                continue
        return
    except Exception as e:
        logger.warning(f"OpenAI streaming failed, falling back to non-streaming: {e}")
        # Fallback: call once and chunk the content
        resp = await llm_service.call_llm(prompt=prompt, provider=LLMProvider.OPENAI, model=model, temperature=0.2, max_tokens=1000)
        text = resp.content or ""
        for i in range(0, len(text), 200):
            yield text[i:i+200]
            await asyncio.sleep(0)  # allow flush


async def _stream_gemini_answer(prompt: str, model: str) -> AsyncGenerator[str, None]:
    # Gemini SDK in this repo uses sync generate_content; stream not wired. Use chunked fallback.
    resp = await llm_service.call_llm(prompt=prompt, provider=LLMProvider.GEMINI, model=model, temperature=0.2, max_tokens=1500)
    text = resp.content or ""
    for i in range(0, len(text), 200):
        yield text[i:i+200]
        await asyncio.sleep(0)


@router.post("/stream")
async def query_stream(request: Request):
    """
    Stream an answer for the user's query using selected model and metadata filters.
    """
    try:
        body = await request.json()
        req = QueryRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    filters = _filters_from_request(req)

    # Fetch context via hybrid search (vector + bm25), applying filters in vector part
    try:
        # Use vector search with filters for precision; fall back to hybrid without filters
        vector_results, _ = await search_service.vector_search(
            query=req.query,
            collection="document_chunks",
            limit=req.top_k,
            score_threshold=0.0,
            filters=filters or None,
        )
    except Exception as e:
        logger.warning(f"Vector search with filters failed: {e}, falling back to hybrid")
        vector_results = []

    contexts = []
    for r in vector_results:
        contexts.append({
            "id": r.id,
            "score": r.score,
            "content": r.content,
            "metadata": r.metadata,
        })

    prompt = _build_prompt(req.query, contexts)
    provider = _provider_from_model(req.model)

    async def streamer() -> AsyncGenerator[bytes, None]:
        try:
            if provider == LLMProvider.OPENAI:
                async for token in _stream_openai_answer(prompt, req.model):
                    yield token.encode("utf-8")
            else:
                async for token in _stream_gemini_answer(prompt, req.model):
                    yield token.encode("utf-8")
        except Exception as e:
            err = f"\n[error] {str(e)}"
            yield err.encode("utf-8")

    return StreamingResponse(streamer(), media_type="text/plain")
