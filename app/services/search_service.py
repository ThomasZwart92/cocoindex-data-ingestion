"""
Unified Search Service
Combines vector search (Qdrant) and graph search (Neo4j) with <200ms latency
"""
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from app.services.qdrant_service import QdrantService
from app.services.supabase_service import SupabaseService
from app.services.neo4j_service import Neo4jService
from app.services.embedding_service import EmbeddingService
from app.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result"""
    id: str
    score: float
    source: str  # 'vector' or 'graph'
    title: str
    content: str
    metadata: Dict[str, Any]
    relationships: Optional[List[Dict]] = None
    chunk_location: Optional[str] = None


class SearchService:
    """Unified search across vector and graph databases"""
    
    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        neo4j_service: Optional[Neo4jService] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.qdrant = qdrant_service or QdrantService()
        self.neo4j = neo4j_service or Neo4jService()
        self.embedder = embedding_service or EmbeddingService()
        self.supabase = SupabaseService()
        
        # Performance tracking
        self.search_metrics = {
            "vector_searches": [],
            "graph_searches": [],
            "hybrid_searches": []
        }
    
    async def vector_search(
        self,
        query: str,
        collection: str = "document_chunks",
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict] = None
    ) -> Tuple[List[SearchResult], float]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query text
            collection: Qdrant collection name
            limit: Maximum results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            (results, latency_ms)
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            embed_res = await self.embedder.embed_text(query)
            query_embedding = embed_res.embedding
            
            # Search in Qdrant
            results = await self.qdrant.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filters
            )
            
            # Convert to SearchResult
            search_results = []
            for result in results:
                payload = result.payload or {}
                search_result = SearchResult(
                    id=result.id,
                    score=result.score,
                    source="vector",
                    title=f"Document {payload.get('document_id', '')}",
                    content=payload.get("chunk_text", "")[:500],
                    metadata=payload,
                    chunk_location=payload.get("location")
                )
                search_results.append(search_result)
            
            latency_ms = (time.time() - start_time) * 1000
            self.search_metrics["vector_searches"].append(latency_ms)
            
            logger.info(f"Vector search completed in {latency_ms:.2f}ms, found {len(search_results)} results")
            return search_results, latency_ms
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return [], latency_ms
    
    async def graph_search(
        self,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 10,
        depth: int = 2
    ) -> Tuple[List[SearchResult], float]:
        """
        Perform graph-based search
        
        Args:
            entity_name: Entity name pattern
            entity_type: Entity type filter
            relationship_type: Relationship type filter
            limit: Maximum results
            depth: Graph traversal depth
            
        Returns:
            (results, latency_ms)
        """
        start_time = time.time()
        
        try:
            # Build Cypher query based on parameters
            query = self._build_graph_query(
                entity_name, entity_type, relationship_type, limit, depth
            )
            
            async with self.neo4j.driver.session() as session:
                result = await session.run(query)
                records = await result.data()
            
            # Convert to SearchResult
            search_results = []
            for record in records:
                # Extract relevant information
                entity = record.get("entity", {})
                relationships = record.get("relationships", [])
                
                search_result = SearchResult(
                    id=entity.get("id", ""),
                    score=1.0,  # Graph search doesn't have similarity scores
                    source="graph",
                    title=entity.get("name", "Unknown"),
                    content=f"Entity Type: {entity.get('type', 'Unknown')}",
                    metadata=entity,
                    relationships=relationships
                )
                search_results.append(search_result)
            
            latency_ms = (time.time() - start_time) * 1000
            self.search_metrics["graph_searches"].append(latency_ms)
            
            logger.info(f"Graph search completed in {latency_ms:.2f}ms, found {len(search_results)} results")
            return search_results, latency_ms
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return [], latency_ms
    
    async def hybrid_search(
        self,
        query: str,
        use_vector: bool = True,
        use_graph: bool = True,
        use_bm25: bool = True,
        vector_weight: float = 0.7,
        limit: int = 10,
        rerank: bool = True
    ) -> Tuple[List[SearchResult], float]:
        """
        Perform hybrid search combining vector and graph results
        
        Args:
            query: Search query
            use_vector: Include vector search
            use_graph: Include graph search
            vector_weight: Weight for vector scores (0-1)
            limit: Maximum results
            
        Returns:
            (results, latency_ms)
        """
        start_time = time.time()
        all_results = []
        
        if use_vector:
            try:
                # Single embed, query three collections
                embed_res = await self.embedder.embed_text(query)
                qv = embed_res.embedding
                # chunks
                chunks = await self.qdrant.search(query_vector=qv, collection_name="document_chunks", limit=limit)
                # tables
                tables = await self.qdrant.search(query_vector=qv, collection_name="document_tables", limit=limit)
                # images (text vector)
                images = await self.qdrant.search(query_vector=qv, collection_name="document_images", limit=limit, vector_name="text")

                def map_results(items, label):
                    out = []
                    for it in items:
                        payload = it.payload or {}
                        content = payload.get("chunk_text") or ""
                        out.append(SearchResult(
                            id=it.id,
                            score=it.score,
                            source=label,
                            title=f"Document {payload.get('document_id', '')}",
                            content=(content or "")[:500],
                            metadata=payload,
                            chunk_location=None
                        ))
                    return out

                all_results.extend(map_results(chunks, "vector_chunk"))
                all_results.extend(map_results(tables, "vector_table"))
                all_results.extend(map_results(images, "vector_image"))
            except Exception as e:
                logger.error(f"Vector multi-collection search failed: {e}")

        bm25_results: List[SearchResult] = []
        if use_bm25:
            try:
                bm25_results, _ = await self.bm25_search(query, limit=limit * 2)
                all_results.extend(bm25_results)
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")

        if use_graph:
            try:
                # Simple keyword-based entity seed
                entities = [word for word in query.split() if word[:1].isupper()]
                entity_name = entities[0] if entities else None
                graph_results, _ = await self.graph_search(entity_name=entity_name, limit=limit)
                all_results.extend(graph_results)
            except Exception as e:
                logger.error(f"Graph search failed: {e}")
        
        # Merge and rank results using reciprocal rank fusion for vector + bm25, then blend graph
        merged_results = self._fuse_results(all_results, primary_sources={"vector_chunk", "vector_table", "vector_image", "vector"},
                                            bm25_sources={"bm25"}, limit=limit * 2)
        
        # Optional reranking
        if rerank and merged_results:
            try:
                from app.services.reranker_service import RerankerService
                rr = RerankerService()
                merged_results = await rr.rerank(query, merged_results, top_k=limit)
            except Exception as e:
                logger.warning(f"Rerank failed or unavailable: {e}")
                merged_results = merged_results[:limit]
        else:
            merged_results = merged_results[:limit]
        
        latency_ms = (time.time() - start_time) * 1000
        self.search_metrics["hybrid_searches"].append(latency_ms)
        
        logger.info(f"Hybrid search completed in {latency_ms:.2f}ms, found {len(merged_results)} results")
        return merged_results, latency_ms

    def _tokenize(self, text: str) -> List[str]:
        import re
        stop = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','as','is','was','are','were'}
        tokens = re.findall(r"\b\w+\b", (text or '').lower())
        return [t for t in tokens if t not in stop and len(t) > 2]

    async def bm25_search(self, query: str, limit: int = 10) -> Tuple[List[SearchResult], float]:
        """Lexical BM25-style search using Supabase (Postgres).
        Approximates BM25 by fetching candidates via ILIKE and scoring in Python.
        """
        start = time.time()
        tokens = self._tokenize(query)
        if not tokens:
            return [], 0.0

        # Fetch candidates per token (cap per-token to control latency)
        per_token_limit = max(20, limit)
        candidates: Dict[str, Dict[str, Any]] = {}
        df: Dict[str, int] = {t: 0 for t in tokens}

        for t in tokens:
            try:
                # Query chunks with contextualized_text ILIKE token
                res = (
                    self.supabase.client
                    .table('chunks')
                    .select('id,document_id,chunk_text,metadata,contextualized_text,bm25_tokens')
                    .ilike('contextualized_text', f'%{t}%')
                    .limit(per_token_limit)
                    .execute()
                )
                rows = res.data or []
                df[t] = len(rows)
                for r in rows:
                    cid = r['id']
                    if cid not in candidates:
                        candidates[cid] = r
            except Exception as e:
                logger.warning(f"BM25 candidate fetch failed for token '{t}': {e}")

        N = max(1, len(candidates))
        avg_len = 100.0  # fallback if tokens missing; not critical for ranking
        lengths: Dict[str, int] = {}
        for cid, r in candidates.items():
            toks = r.get('bm25_tokens') or self._tokenize(r.get('contextualized_text') or r.get('chunk_text') or '')
            lengths[cid] = len(toks)
        if lengths:
            avg_len = sum(lengths.values()) / len(lengths)

        # BM25 scoring params
        k1 = 1.2
        b = 0.75

        results: List[SearchResult] = []
        for cid, r in candidates.items():
            toks = r.get('bm25_tokens') or self._tokenize(r.get('contextualized_text') or r.get('chunk_text') or '')
            # term frequency per token
            score = 0.0
            for t in tokens:
                tf = toks.count(t)
                if tf == 0:
                    continue
                # doc frequency df[t] over candidate set; smooth
                dft = max(1, df.get(t, 1))
                idf = np.log((N - dft + 0.5) / (dft + 0.5) + 1)
                dl = max(1, lengths.get(cid, len(toks)))
                denom = tf + k1 * (1 - b + b * dl / avg_len)
                score += idf * (tf * (k1 + 1)) / denom
            if score > 0:
                payload = {
                    'document_id': r.get('document_id'),
                    **(r.get('metadata') or {})
                }
                content = (r.get('contextualized_text') or r.get('chunk_text') or '')
                results.append(SearchResult(
                    id=str(cid),
                    score=float(score),
                    source='bm25',
                    title=f"Document {r.get('document_id','')}",
                    content=content[:500],
                    metadata=payload,
                    chunk_location=None
                ))

        # Sort by BM25 score
        results.sort(key=lambda x: x.score, reverse=True)
        latency = (time.time() - start) * 1000
        return results[:limit], latency

    def _fuse_results(self, results: List[SearchResult], primary_sources: set, bm25_sources: set, limit: int) -> List[SearchResult]:
        """Reciprocal Rank Fusion (RRF) between vector and BM25, then blend graph.
        - Compute ranks per source group; score = sum(1/(k + rank)) with k=60.
        - Graph results are appended with modest boost if they share IDs with others.
        """
        k = 60
        # Split by source
        vec = [r for r in results if r.source in primary_sources]
        bm = [r for r in results if r.source in bm25_sources]
        gr = [r for r in results if r.source == 'graph']

        def rank_map(items: List[SearchResult]) -> Dict[str, int]:
            items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
            return {it.id: idx + 1 for idx, it in enumerate(items_sorted)}

        vr = rank_map(vec)
        br = rank_map(bm)

        # Collect all IDs from vec and bm25
        ids = set(vr.keys()) | set(br.keys())
        fused: Dict[str, SearchResult] = {}

        # Base payload choose highest-score instance
        best_by_id: Dict[str, SearchResult] = {}
        for item in vec + bm:
            if item.id not in best_by_id or item.score > best_by_id[item.id].score:
                best_by_id[item.id] = item

        for cid in ids:
            rr_score = 0.0
            if cid in vr:
                rr_score += 1.0 / (k + vr[cid])
            if cid in br:
                rr_score += 1.0 / (k + br[cid])
            base = best_by_id[cid]
            fused[cid] = SearchResult(
                id=cid,
                score=rr_score,
                source='hybrid',
                title=base.title,
                content=base.content,
                metadata=base.metadata,
                relationships=None,
                chunk_location=base.chunk_location,
            )

        # Optionally attach graph insights by boosting if an ID is present in graph (by document id mapping if available)
        # For simplicity, append graph items that are not already present with small base score
        for g in gr:
            if g.id not in fused:
                fused[g.id] = g

        ranked = sorted(fused.values(), key=lambda x: x.score, reverse=True)
        return ranked[:limit]
    
    async def semantic_search(
        self,
        query: str,
        context: Optional[str] = None,
        limit: int = 10,
        rerank: bool = True
    ) -> Tuple[List[SearchResult], float]:
        """
        Advanced semantic search with context and reranking
        
        Args:
            query: Search query
            context: Additional context for the search
            limit: Maximum results
            rerank: Whether to rerank results
            
        Returns:
            (results, latency_ms)
        """
        start_time = time.time()
        
        # Enhance query with context
        enhanced_query = query
        if context:
            enhanced_query = f"{context} {query}"
        
        # Perform initial search
        results, _ = await self.vector_search(enhanced_query, limit=limit * 2)
        
        # Rerank if requested
        if rerank and results:
            results = await self._rerank_results(query, results, limit)
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Semantic search completed in {latency_ms:.2f}ms")
        return results[:limit], latency_ms
    
    def _build_graph_query(
        self,
        entity_name: Optional[str],
        entity_type: Optional[str],
        relationship_type: Optional[str],
        limit: int,
        depth: int
    ) -> str:
        """Build Cypher query for graph search"""
        
        where_clauses = []
        if entity_name:
            where_clauses.append(f"e.name CONTAINS '{entity_name}'")
        if entity_type:
            where_clauses.append(f"e.type = '{entity_type}'")
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        rel_pattern = f"-[r:{relationship_type}]-" if relationship_type else "-[r]-"
        
        query = f"""
        MATCH (e:Entity) {where_clause}
        OPTIONAL MATCH path = (e){rel_pattern}(connected)
        WITH e, collect({{
            type: type(r),
            target: connected.name,
            properties: properties(r)
        }}) as relationships
        RETURN {{
            id: e.id,
            name: e.name,
            type: e.type,
            properties: properties(e)
        }} as entity, relationships
        LIMIT {limit}
        """
        
        return query
    
    def _merge_and_rank(
        self,
        results: List[SearchResult],
        vector_weight: float = 0.7
    ) -> List[SearchResult]:
        """Merge and rank results from different sources"""
        
        # Group by ID to handle duplicates
        merged = {}
        
        for result in results:
            result_id = result.id
            
            if result_id not in merged:
                merged[result_id] = result
            else:
                # Combine scores based on source
                existing = merged[result_id]
                
                if result.source == "vector" and existing.source == "graph":
                    # Weighted combination
                    existing.score = (result.score * vector_weight + 
                                    existing.score * (1 - vector_weight))
                elif result.source == "graph" and existing.source == "vector":
                    existing.score = (existing.score * vector_weight + 
                                    result.score * (1 - vector_weight))
                else:
                    # Same source, take higher score
                    existing.score = max(existing.score, result.score)
                
                # Merge relationships if available
                if result.relationships:
                    if existing.relationships:
                        existing.relationships.extend(result.relationships)
                    else:
                        existing.relationships = result.relationships
        
        # Sort by score
        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        
        return ranked
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        limit: int
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder or other method"""
        
        # Simple reranking based on keyword overlap
        # In production, use a cross-encoder model
        
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            overlap = len(query_words & content_words)
            
            # Boost score based on overlap
            result.score *= (1 + overlap * 0.1)
        
        # Re-sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        
        def calculate_stats(metrics: List[float]) -> Dict:
            if not metrics:
                return {"count": 0, "avg_ms": 0, "p95_ms": 0, "p99_ms": 0}
            
            metrics_sorted = sorted(metrics)
            return {
                "count": len(metrics),
                "avg_ms": np.mean(metrics),
                "min_ms": metrics_sorted[0],
                "max_ms": metrics_sorted[-1],
                "p50_ms": np.percentile(metrics_sorted, 50),
                "p95_ms": np.percentile(metrics_sorted, 95),
                "p99_ms": np.percentile(metrics_sorted, 99)
            }
        
        return {
            "vector": calculate_stats(self.search_metrics["vector_searches"]),
            "graph": calculate_stats(self.search_metrics["graph_searches"]),
            "hybrid": calculate_stats(self.search_metrics["hybrid_searches"]),
            "target_latency_ms": 200,
            "meeting_target": all(
                calculate_stats(metrics)["p95_ms"] < 200
                for metrics in self.search_metrics.values()
                if metrics
            )
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of search services"""
        
        health = {
            "qdrant": False,
            "neo4j": False,
            "overall": False
        }
        
        # Check Qdrant
        try:
            collections = await self.qdrant.get_collections()
            health["qdrant"] = True
        except:
            pass
        
        # Check Neo4j
        health["neo4j"] = await self.neo4j.health_check()
        
        # Overall health
        health["overall"] = health["qdrant"] and health["neo4j"]
        
        return health


# Test function
async def test_search_performance():
    """Test search performance to ensure <200ms latency"""
    print("\n" + "="*60)
    print("SEARCH PERFORMANCE TEST")
    print("="*60)
    
    search = SearchService()
    
    # Test queries
    queries = [
        "Model X500 water dispenser",
        "error code E501",
        "temperature sensor control board",
        "premium market enterprise customers",
        "firmware updates engineering team"
    ]
    
    print("\n1. Vector Search Performance:")
    for query in queries:
        results, latency = await search.vector_search(query, limit=5)
        status = "✓" if latency < 200 else "✗"
        print(f"  {status} Query: '{query[:30]}...' - {latency:.2f}ms ({len(results)} results)")
    
    print("\n2. Graph Search Performance:")
    entities = ["Model X500", "Engineering", "E501", "Premium Line"]
    for entity in entities:
        results, latency = await search.graph_search(entity_name=entity, limit=5)
        status = "✓" if latency < 200 else "✗"
        print(f"  {status} Entity: '{entity}' - {latency:.2f}ms ({len(results)} results)")
    
    print("\n3. Hybrid Search Performance:")
    for query in queries[:3]:
        results, latency = await search.hybrid_search(query, limit=5)
        status = "✓" if latency < 200 else "✗"
        print(f"  {status} Query: '{query[:30]}...' - {latency:.2f}ms ({len(results)} results)")
    
    # Get statistics
    stats = await search.get_performance_stats()
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for search_type, metrics in stats.items():
        if search_type in ["vector", "graph", "hybrid"] and metrics["count"] > 0:
            print(f"\n{search_type.upper()} Search:")
            print(f"  Total searches: {metrics['count']}")
            print(f"  Average latency: {metrics['avg_ms']:.2f}ms")
            print(f"  P95 latency: {metrics['p95_ms']:.2f}ms")
            print(f"  P99 latency: {metrics['p99_ms']:.2f}ms")
    
    print(f"\n✅ Meeting <200ms target: {stats['meeting_target']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_search_performance())
