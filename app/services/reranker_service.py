"""
Reranker Service
Provides an abstraction for reranking search candidates.
Uses Cohere if available (and configured), else falls back to lexical overlap.
"""
from __future__ import annotations

from typing import List
import os
import logging

from app.services.search_service import SearchResult

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        self.use_cohere = False
        self.client = None
        self.model = os.getenv('COHERE_RERANK_MODEL', 'rerank-english-v3.0')
        api_key = os.getenv('COHERE_API_KEY')
        if api_key:
            try:
                import cohere  # type: ignore
                self.client = cohere.Client(api_key)
                self.use_cohere = True
            except Exception as e:
                logger.warning(f"Cohere not available for reranking: {e}")
                self.use_cohere = False

    async def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 20) -> List[SearchResult]:
        # Prefer Cohere if configured
        if self.use_cohere and self.client:
            try:
                docs = [c.content or '' for c in candidates]
                resp = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                    top_n=min(top_k, len(docs))
                )
                # Cohere returns items with index and relevance_score
                reranked: List[SearchResult] = []
                for item in resp.results:
                    idx = getattr(item, 'index', None)
                    if idx is None or idx < 0 or idx >= len(candidates):
                        continue
                    c = candidates[idx]
                    c.score = float(getattr(item, 'relevance_score', 0.0))
                    reranked.append(c)
                return reranked
            except Exception as e:
                logger.warning(f"Cohere rerank failed, falling back: {e}")

        # Fallback: simple keyword overlap rerank
        q_words = set((query or '').lower().split())
        for c in candidates:
            text = (c.content or '').lower()
            overlap = sum(1 for w in q_words if w in text)
            c.score = c.score * (1 + 0.1 * overlap)
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
