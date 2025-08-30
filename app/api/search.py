"""
Search API Endpoints
FastAPI routes for vector, graph, and hybrid search
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
import time

from app.services.search_service import SearchService, SearchResult
from app.services.auth_service import get_current_user
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

# Initialize search service
search_service = SearchService()


class VectorSearchRequest(BaseModel):
    """Vector search request model"""
    query: str = Field(..., description="Search query text")
    collection: str = Field(default="documents", description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    score_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class GraphSearchRequest(BaseModel):
    """Graph search request model"""
    entity_name: Optional[str] = Field(default=None, description="Entity name pattern")
    entity_type: Optional[str] = Field(default=None, description="Entity type filter")
    relationship_type: Optional[str] = Field(default=None, description="Relationship type filter")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    depth: int = Field(default=2, ge=1, le=5, description="Graph traversal depth")


class HybridSearchRequest(BaseModel):
    """Hybrid search request model"""
    query: str = Field(..., description="Search query text")
    use_vector: bool = Field(default=True, description="Include vector search")
    use_graph: bool = Field(default=True, description="Include graph search")
    vector_weight: float = Field(default=0.7, ge=0, le=1, description="Weight for vector scores")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[Dict[str, Any]]
    count: int
    latency_ms: float
    search_type: str


@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    # current_user: Dict = Depends(get_current_user)  # Uncomment when auth is ready
):
    """
    Perform vector similarity search on document chunks
    
    Returns results ranked by semantic similarity to the query.
    Target latency: <200ms for 95% of requests.
    """
    try:
        start_time = time.time()
        
        results, latency = await search_service.vector_search(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filters=request.filters
        )
        
        # Convert results to dict format
        result_dicts = []
        for result in results:
            result_dict = {
                "id": result.id,
                "score": result.score,
                "title": result.title,
                "content": result.content,
                "metadata": result.metadata,
                "chunk_location": result.chunk_location
            }
            result_dicts.append(result_dict)
        
        total_latency = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=result_dicts,
            count=len(result_dicts),
            latency_ms=total_latency,
            search_type="vector"
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph", response_model=SearchResponse)
async def graph_search(
    request: GraphSearchRequest,
    # current_user: Dict = Depends(get_current_user)
):
    """
    Perform graph-based search on entities and relationships
    
    Searches the knowledge graph for entities and their relationships.
    Returns connected entities up to the specified depth.
    """
    try:
        start_time = time.time()
        
        results, latency = await search_service.graph_search(
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            relationship_type=request.relationship_type,
            limit=request.limit,
            depth=request.depth
        )
        
        # Convert results to dict format
        result_dicts = []
        for result in results:
            result_dict = {
                "id": result.id,
                "score": result.score,
                "title": result.title,
                "content": result.content,
                "metadata": result.metadata,
                "relationships": result.relationships
            }
            result_dicts.append(result_dict)
        
        total_latency = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=result_dicts,
            count=len(result_dicts),
            latency_ms=total_latency,
            search_type="graph"
        )
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    # current_user: Dict = Depends(get_current_user)
):
    """
    Perform hybrid search combining vector and graph results
    
    Runs both vector and graph searches in parallel and merges results.
    Provides the best of both semantic and structural search.
    """
    try:
        start_time = time.time()
        
        results, latency = await search_service.hybrid_search(
            query=request.query,
            use_vector=request.use_vector,
            use_graph=request.use_graph,
            vector_weight=request.vector_weight,
            limit=request.limit
        )
        
        # Convert results to dict format
        result_dicts = []
        for result in results:
            result_dict = {
                "id": result.id,
                "score": result.score,
                "source": result.source,
                "title": result.title,
                "content": result.content,
                "metadata": result.metadata,
                "relationships": result.relationships,
                "chunk_location": result.chunk_location
            }
            result_dicts.append(result_dict)
        
        total_latency = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=result_dicts,
            count=len(result_dicts),
            latency_ms=total_latency,
            search_type="hybrid"
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick")
async def quick_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Quick search endpoint for simple queries
    
    Performs a hybrid search with default settings.
    Optimized for speed and ease of use.
    """
    try:
        results, latency = await search_service.hybrid_search(
            query=q,
            limit=limit
        )
        
        # Simplified response format
        simple_results = []
        for result in results:
            simple_results.append({
                "title": result.title,
                "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "score": result.score,
                "type": result.metadata.get("type", "document")
            })
        
        return {
            "query": q,
            "results": simple_results,
            "count": len(simple_results),
            "latency_ms": latency
        }
        
    except Exception as e:
        logger.error(f"Quick search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggest")
async def search_suggestions(
    q: str = Query(..., min_length=2, description="Partial query for suggestions"),
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Get search suggestions based on partial query
    
    Returns entity names and common queries that match the input.
    Used for autocomplete functionality.
    """
    try:
        # Search for matching entities
        entities = await search_service.neo4j.search_entities(
            name_pattern=q,
            limit=limit
        )
        
        suggestions = []
        for entity in entities:
            suggestions.append({
                "text": entity["name"],
                "type": entity["type"],
                "category": "entity"
            })
        
        # Add common query patterns (in production, track actual queries)
        if "error" in q.lower():
            suggestions.append({
                "text": f"{q} troubleshooting",
                "type": "query",
                "category": "suggested"
            })
        
        return {
            "query": q,
            "suggestions": suggestions[:limit]
        }
        
    except Exception as e:
        logger.error(f"Search suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def search_statistics():
    """
    Get search performance statistics
    
    Returns metrics about search latency and performance.
    Useful for monitoring if we're meeting the <200ms target.
    """
    try:
        stats = await search_service.get_performance_stats()
        health = await search_service.health_check()
        
        return {
            "performance": stats,
            "health": health,
            "status": "healthy" if health["overall"] else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Failed to get search statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def search_health():
    """
    Check health of search services
    
    Verifies connectivity to Qdrant and Neo4j.
    """
    try:
        health = await search_service.health_check()
        
        if not health["overall"]:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "services": health
                }
            )
        
        return {
            "status": "healthy",
            "services": health
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "message": str(e)}
        )