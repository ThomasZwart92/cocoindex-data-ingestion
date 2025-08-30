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
        
        # Performance tracking
        self.search_metrics = {
            "vector_searches": [],
            "graph_searches": [],
            "hybrid_searches": []
        }
    
    async def vector_search(
        self,
        query: str,
        collection: str = "documents",
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
            query_embedding = await self.embedder.embed_text(query)
            
            # Search in Qdrant
            results = await self.qdrant.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to SearchResult
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.id,
                    score=result.score,
                    source="vector",
                    title=result.payload.get("title", "Untitled"),
                    content=result.payload.get("text", "")[:500],
                    metadata=result.payload,
                    chunk_location=result.payload.get("location")
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
        vector_weight: float = 0.7,
        limit: int = 10
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
        
        tasks = []
        if use_vector:
            tasks.append(self.vector_search(query, limit=limit))
        
        if use_graph:
            # Extract entities from query for graph search
            # Simple approach: look for capitalized words
            entities = [word for word in query.split() if word[0].isupper()]
            entity_name = entities[0] if entities else None
            tasks.append(self.graph_search(entity_name=entity_name, limit=limit))
        
        # Run searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
                continue
            
            search_results, _ = result
            all_results.extend(search_results)
        
        # Merge and rank results
        merged_results = self._merge_and_rank(
            all_results, 
            vector_weight=vector_weight
        )[:limit]
        
        latency_ms = (time.time() - start_time) * 1000
        self.search_metrics["hybrid_searches"].append(latency_ms)
        
        logger.info(f"Hybrid search completed in {latency_ms:.2f}ms, found {len(merged_results)} results")
        return merged_results, latency_ms
    
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