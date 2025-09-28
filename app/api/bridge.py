"""
API Bridge Pattern Implementation
Fetches data from optimized stores (Qdrant, Neo4j) for UI consumption
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Import our service clients
from app.services.qdrant_service import QdrantService
from app.services.neo4j_service import Neo4jService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["bridge"])

# Initialize services
qdrant_service = QdrantService()
neo4j_service = Neo4jService()
supabase_service = SupabaseService()

@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str, limit: int = Query(100, ge=1, le=1000)) -> List[Dict[str, Any]]:
    """
    Fetch chunks from Qdrant for a specific document.
    
    Returns chunks with their text content but excludes embeddings for performance.
    """
    try:
        logger.info(f"Fetching chunks for document {document_id} from Qdrant")
        
        # Search Qdrant for chunks belonging to this document
        # Using scroll instead of search since we want all chunks, not similarity
        chunks = []
        
        # Try to fetch chunks using document_id filter
        try:
            # For Qdrant, we need to use the scroll API to get all points with a filter
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Scroll through all matching chunks
            offset = None
            while True:
                response = qdrant_service.client.scroll(
                    collection_name="document_chunks",
                    scroll_filter=filter_condition,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't include embeddings in response
                )
                
                if not response or not response[0]:
                    break
                
                points, next_offset = response
                
                for point in points:
                    text_content = point.payload.get("text", "")
                    chunk_data = {
                        "id": str(point.id),
                        "chunk_number": point.payload.get("chunk_number", 0),
                        "chunk_text": text_content,  # Changed from "text" to "chunk_text"
                        "chunk_size": len(text_content),  # Added chunk_size field
                        "metadata": point.payload.get("metadata", {}),
                        "document_id": document_id,
                        "start_position": point.payload.get("start_position", 0),
                        "end_position": point.payload.get("end_position", 0)
                    }
                    chunks.append(chunk_data)
                
                if next_offset is None:
                    break
                offset = next_offset
                
        except Exception as e:
            logger.warning(f"Qdrant scroll failed, trying search: {e}")
            
            # Fallback: Try search without filter (less efficient)
            search_results = qdrant_service.search_similar(
                collection_name="document_chunks",
                query_text="",  # Empty query to get all
                limit=limit
            )
            
            # Filter results to only include chunks from this document
            chunks = []
            for result in search_results:
                if result.payload.get("document_id") == document_id:
                    text_content = result.payload.get("text", "")
                    chunks.append({
                        "id": str(result.id),
                        "chunk_number": result.payload.get("chunk_number", 0),
                        "chunk_text": text_content,  # Changed from "text" to "chunk_text"
                        "chunk_size": len(text_content),  # Added chunk_size field
                        "metadata": result.payload.get("metadata", {}),
                        "document_id": document_id,
                        "start_position": result.payload.get("start_position", 0),
                        "end_position": result.payload.get("end_position", 0)
                    })
        
        # Sort chunks by chunk_number for proper ordering
        chunks.sort(key=lambda x: x.get("chunk_number", 0))
        
        logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error fetching chunks from Qdrant: {e}")
        # Return empty list instead of error to allow graceful degradation
        return []

# Changed route to avoid conflict with entities.py handler
@router.get("/documents/{document_id}/entities-neo4j")
async def get_document_entities_neo4j(document_id: str) -> List[Dict[str, Any]]:
    """
    Fetch entities from Neo4j for a specific document.
    
    Returns entities with their relationships to the document.
    """
    try:
        logger.info(f"Fetching entities for document {document_id} from Neo4j")
        
        # Query Neo4j for entities related to this document
        query = """
        MATCH (d:Document {id: $document_id})-[r:HAS_ENTITY|MENTIONS|REFERENCES]->(e:Entity)
        RETURN e.id as id, 
               e.name as name, 
               e.type as type,
               e.confidence as confidence,
               e.metadata as metadata,
               type(r) as relationship_type,
               r.confidence as relationship_confidence
        ORDER BY e.type, e.name
        """
        
        try:
            result = neo4j_service.execute_query(query, {"document_id": document_id})
            
            entities = []
            for record in result:
                entity_data = {
                    "id": record.get("id", ""),
                    "entity_name": record.get("name", ""),  # Changed from "name" to "entity_name"
                    "entity_type": record.get("type", "Unknown"),  # Changed from "type" to "entity_type"
                    "document_id": document_id,  # Added document_id
                    "confidence": record.get("confidence", 0.0),
                    "metadata": record.get("metadata", {}),
                    "relationship_type": record.get("relationship_type", "MENTIONS"),
                    "relationship_confidence": record.get("relationship_confidence", 0.0)
                }
                entities.append(entity_data)
            
            logger.info(f"Retrieved {len(entities)} entities for document {document_id}")
            return entities
            
        except Exception as e:
            logger.warning(f"Neo4j query failed, trying alternative approach: {e}")
            
            # Alternative: Try simpler query
            simple_query = """
            MATCH (d:Document {id: $document_id})
            MATCH (d)-[]->(e:Entity)
            RETURN DISTINCT e
            """
            
            result = neo4j_service.execute_query(simple_query, {"document_id": document_id})
            
            entities = []
            for record in result:
                if 'e' in record:
                    entity_node = record['e']
                    entity_data = {
                        "id": entity_node.get("id", ""),
                        "entity_name": entity_node.get("name", ""),  # Changed from "name" to "entity_name"
                        "entity_type": entity_node.get("type", "Unknown"),  # Changed from "type" to "entity_type"
                        "document_id": document_id,  # Added document_id
                        "confidence": entity_node.get("confidence", 0.0),
                        "metadata": entity_node.get("metadata", {})
                    }
                    entities.append(entity_data)
            
            return entities
        
    except Exception as e:
        logger.error(f"Error fetching entities from Neo4j: {e}")
        # Return empty list instead of error to allow graceful degradation
        return []

@router.get("/documents/{document_id}/relationships")
async def get_document_relationships(document_id: str) -> List[Dict[str, Any]]:
    """
    Fetch the knowledge graph relationships for a document.
    
    Returns relationships array for the document.
    """
    try:
        logger.info(f"Fetching relationships for document {document_id} from Supabase")
        
        # Find canonical ids present in this document via mentions
        client = supabase_service.client
        mentions = client.table("entity_mentions").select("canonical_entity_id").eq("document_id", document_id).not_.is_("canonical_entity_id", "null").execute()
        ids = [m.get("canonical_entity_id") for m in (mentions.data or []) if m.get("canonical_entity_id")]
        if not ids:
            logger.info(f"No canonical entities found for document {document_id}")
            return []
        ids = list(set(ids))

        # Fetch canonical relationships where either end is present in this doc
        src_rels = client.table("canonical_relationships").select("*").in_("source_entity_id", ids).execute()
        tgt_rels = client.table("canonical_relationships").select("*").in_("target_entity_id", ids).execute()

        # Combine, deduplicate, map confidence field
        all_relationships: list[Dict[str, Any]] = []
        seen_ids = set()
        for rel in (src_rels.data or []) + (tgt_rels.data or []):
            rid = rel.get('id')
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rel_copy = rel.copy()
            rel_copy['confidence'] = float(rel_copy.get('confidence_score') or 0.0)
            all_relationships.append(rel_copy)

        if all_relationships:
            all_relationships.sort(key=lambda x: x.get('created_at', ''))
            logger.info(f"Retrieved {len(all_relationships)} canonical relationships for document {document_id}")
            return all_relationships
        else:
            logger.info(f"No canonical relationships found for document {document_id}")
            return []
        
    except Exception as e:
        logger.error(f"Error fetching relationships: {e}")
        # Return empty array instead of error to prevent frontend crash
        return []

@router.post("/search/vector")
async def search_vector(
    query: str,
    limit: int = Query(10, ge=1, le=100),
    threshold: float = Query(0.7, ge=0.0, le=1.0)
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search in Qdrant.
    
    Must return results in <200ms to meet performance requirements.
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Performing vector search for query: '{query[:50]}...'")
        
        # Search Qdrant for similar chunks
        search_results = qdrant_service.search_similar(
            collection_name="document_chunks",
            query_text=query,
            limit=limit
        )
        
        # Format results for UI
        results = []
        for result in search_results:
            if result.score >= threshold:
                results.append({
                    "id": str(result.id),
                    "document_id": result.payload.get("document_id", ""),
                    "text": result.payload.get("text", ""),
                    "score": float(result.score),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_number": result.payload.get("chunk_number", 0)
                })
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Vector search completed in {elapsed_ms:.0f}ms, found {len(results)} results")
        
        # Log warning if latency requirement not met
        if elapsed_ms > 200:
            logger.warning(f"Vector search latency {elapsed_ms:.0f}ms exceeds 200ms target")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/search/graph")
async def search_graph(
    entity_name: str,
    depth: int = Query(2, ge=1, le=5),
    limit: int = Query(50, ge=1, le=200)
) -> Dict[str, Any]:
    """
    Perform graph traversal search in Neo4j.
    
    Finds entities and their relationships up to specified depth.
    """
    try:
        logger.info(f"Performing graph search for entity: '{entity_name}' with depth {depth}")
        
        # Query Neo4j for entity and its neighborhood
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($entity_name)
        WITH e LIMIT 1
        MATCH path = (e)-[*0..""" + str(depth) + """]->(related:Entity)
        WITH e, related, path
        LIMIT $limit
        RETURN DISTINCT
            e.id as root_id,
            e.name as root_name,
            e.type as root_type,
            related.id as related_id,
            related.name as related_name,
            related.type as related_type,
            length(path) as distance
        ORDER BY distance
        """
        
        result = neo4j_service.execute_query(
            query, 
            {"entity_name": entity_name, "limit": limit}
        )
        
        # Build graph structure
        nodes = {}
        edges = []
        
        for record in result:
            # Add root node
            root_id = record.get("root_id")
            if root_id and root_id not in nodes:
                nodes[root_id] = {
                    "id": root_id,
                    "name": record.get("root_name", ""),
                    "type": record.get("root_type", "Unknown"),
                    "distance": 0
                }
            
            # Add related node
            related_id = record.get("related_id")
            if related_id and related_id not in nodes:
                nodes[related_id] = {
                    "id": related_id,
                    "name": record.get("related_name", ""),
                    "type": record.get("related_type", "Unknown"),
                    "distance": record.get("distance", 0)
                }
        
        graph_data = {
            "query": entity_name,
            "nodes": list(nodes.values()),
            "edges": edges,
            "depth": depth
        }
        
        logger.info(f"Graph search found {len(nodes)} entities")
        return graph_data
        
    except Exception as e:
        logger.error(f"Error during graph search: {e}")
        return {"query": entity_name, "nodes": [], "edges": [], "depth": depth}

@router.post("/search/hybrid")
async def search_hybrid(
    query: str,
    vector_weight: float = Query(0.7, ge=0.0, le=1.0),
    graph_weight: float = Query(0.3, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=100)
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector similarity and graph relationships.
    
    Combines results from both Qdrant and Neo4j with weighted scoring.
    """
    try:
        logger.info(f"Performing hybrid search for query: '{query[:50]}...'")
        
        # Perform vector search
        vector_results = await search_vector(query, limit=limit * 2)
        
        # Extract entities from query for graph search
        # For now, use the full query as entity search
        graph_results = await search_graph(query, depth=2, limit=limit * 2)
        
        # Combine and weight results
        combined_scores = {}
        
        # Add vector search results
        for result in vector_results:
            doc_id = result.get("document_id", "")
            if doc_id:
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {
                        "document_id": doc_id,
                        "vector_score": 0,
                        "graph_score": 0,
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {})
                    }
                combined_scores[doc_id]["vector_score"] = max(
                    combined_scores[doc_id]["vector_score"],
                    result.get("score", 0)
                )
        
        # Add graph search results (simplified - would need document mapping)
        for node in graph_results.get("nodes", []):
            # This is simplified - in reality, we'd map entities back to documents
            distance = node.get("distance", 0)
            graph_score = 1.0 / (1 + distance)  # Inverse distance scoring
            
            # For demo, create synthetic document ID from entity
            doc_id = f"doc_{node.get('id', '')}"
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "document_id": doc_id,
                    "vector_score": 0,
                    "graph_score": 0,
                    "text": f"Entity: {node.get('name', '')}",
                    "metadata": {"entity_type": node.get("type", "")}
                }
            combined_scores[doc_id]["graph_score"] = max(
                combined_scores[doc_id]["graph_score"],
                graph_score
            )
        
        # Calculate final weighted scores
        results = []
        for doc_id, scores in combined_scores.items():
            final_score = (
                scores["vector_score"] * vector_weight +
                scores["graph_score"] * graph_weight
            )
            results.append({
                "document_id": doc_id,
                "score": final_score,
                "vector_score": scores["vector_score"],
                "graph_score": scores["graph_score"],
                "text": scores["text"],
                "metadata": scores["metadata"]
            })
        
        # Sort by final score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]
        
        logger.info(f"Hybrid search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

