"""Qdrant Vector Database Service for storing and searching embeddings"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4
import json

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    UpdateStatus,
    CollectionStatus, OptimizersConfigDiff
)
from qdrant_client.http import models

from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from vector search"""
    id: str
    score: float
    payload: Dict[str, Any]
    document_id: Optional[str] = None
    chunk_number: Optional[int] = None
    text: Optional[str] = None

class QdrantService:
    """Service for managing vector storage in Qdrant"""
    
    # Collection configurations
    COLLECTIONS = {
        "document_chunks": {
            "vector_size": 1536,  # text-embedding-3-small
            "distance": Distance.COSINE,
            "description": "Document chunk embeddings for semantic search"
        },
        "document_images": {
            "vectors": {
                "visual": {"size": 1024, "distance": Distance.COSINE},  # ColPali
                "text": {"size": 1536, "distance": Distance.COSINE}     # text-embedding-3-small
            },
            "description": "Image embeddings with visual and text vectors"
        }
    }
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Qdrant service
        
        Args:
            url: Qdrant server URL (defaults to settings)
            api_key: API key for Qdrant Cloud (optional)
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        
        # Initialize clients
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            self.async_client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
            self.async_client = AsyncQdrantClient(url=self.url)
            
        logger.info(f"Initialized QdrantService with URL: {self.url}")
    
    async def ensure_collection(
        self, 
        collection_name: str = "document_chunks",
        recreate: bool = False
    ) -> bool:
        """
        Ensure collection exists with proper configuration
        
        Args:
            collection_name: Name of the collection
            recreate: Whether to recreate if exists
            
        Returns:
            True if collection is ready
        """
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            exists = any(col.name == collection_name for col in collections.collections)
            
            if exists and not recreate:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            if exists and recreate:
                logger.info(f"Recreating collection '{collection_name}'")
                await self.async_client.delete_collection(collection_name)
            
            # Get collection config
            if collection_name not in self.COLLECTIONS:
                raise ValueError(f"Unknown collection: {collection_name}")
            
            config = self.COLLECTIONS[collection_name]
            
            # Create collection based on type
            if collection_name == "document_chunks":
                # Simple single-vector collection
                await self.async_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=config["vector_size"],
                        distance=config["distance"]
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=10000,
                        memmap_threshold=50000
                    )
                )
            elif collection_name == "document_images":
                # Multi-vector collection
                vectors_config = {}
                for name, vec_config in config["vectors"].items():
                    vectors_config[name] = VectorParams(
                        size=vec_config["size"],
                        distance=vec_config["distance"]
                    )
                
                await self.async_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=10000,
                        memmap_threshold=50000
                    )
                )
            
            logger.info(f"Created collection '{collection_name}'")
            
            # Wait for collection to be ready
            for _ in range(10):
                info = await self.async_client.get_collection(collection_name)
                if info.status == CollectionStatus.GREEN:
                    logger.info(f"Collection '{collection_name}' is ready")
                    return True
                await asyncio.sleep(0.5)
            
            logger.warning(f"Collection '{collection_name}' not ready after waiting")
            return False
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    async def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        collection_name: str = "document_chunks"
    ) -> int:
        """
        Upsert document chunks with embeddings
        
        Args:
            chunks: List of chunk data dictionaries
            embeddings: List of embedding vectors
            collection_name: Target collection
            
        Returns:
            Number of points upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
        
        # Ensure collection exists
        await self.ensure_collection(collection_name)
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Generate or use existing ID
            point_id = chunk.get("id") or str(uuid4())
            
            # Prepare payload
            payload = {
                "document_id": chunk.get("document_id"),
                "chunk_number": chunk.get("chunk_number"),
                "chunk_text": chunk.get("chunk_text", ""),
                "chunk_size": chunk.get("chunk_size"),
                "start_position": chunk.get("start_position"),
                "end_position": chunk.get("end_position"),
                "metadata": chunk.get("metadata", {}),
                "source_type": chunk.get("source_type"),
                "filename": chunk.get("filename"),
                "created_at": chunk.get("created_at")
            }
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Upsert in batches
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            result = await self.async_client.upsert(
                collection_name=collection_name,
                points=batch
            )
            
            if result.status == UpdateStatus.COMPLETED:
                total_upserted += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} points)")
            else:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}")
        
        logger.info(f"Successfully upserted {total_upserted} chunks to '{collection_name}'")
        return total_upserted
    
    async def search(
        self,
        query_vector: List[float],
        collection_name: str = "document_chunks",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            collection_name: Collection to search
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filter_conditions: Additional filters (e.g., {"document_id": "xyz"})
            
        Returns:
            List of search results
        """
        # Build filter if conditions provided
        search_filter = None
        if filter_conditions:
            must_conditions = []
            for field, value in filter_conditions.items():
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            search_filter = Filter(must=must_conditions)
        
        # Perform search
        results = await self.async_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(SearchResult(
                id=str(result.id),
                score=result.score,
                payload=payload,
                document_id=payload.get("document_id"),
                chunk_number=payload.get("chunk_number"),
                text=payload.get("chunk_text")
            ))
        
        logger.info(f"Found {len(search_results)} results in '{collection_name}'")
        return search_results
    
    async def hybrid_search(
        self,
        text_vector: Optional[List[float]] = None,
        visual_vector: Optional[List[float]] = None,
        collection_name: str = "document_images",
        limit: int = 10,
        fusion_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        Perform hybrid search on multi-vector collection
        
        Args:
            text_vector: Text embedding vector
            visual_vector: Visual embedding vector
            collection_name: Collection to search
            limit: Maximum results
            fusion_weight: Weight for combining scores (0=visual only, 1=text only)
            
        Returns:
            Fused search results
        """
        results_map = {}
        
        # Search with text vector
        if text_vector:
            text_results = await self.async_client.search(
                collection_name=collection_name,
                query_vector=("text", text_vector),
                limit=limit * 2,  # Get more for fusion
                with_payload=True
            )
            
            for result in text_results:
                results_map[result.id] = {
                    "text_score": result.score,
                    "visual_score": 0,
                    "payload": result.payload
                }
        
        # Search with visual vector
        if visual_vector:
            visual_results = await self.async_client.search(
                collection_name=collection_name,
                query_vector=("visual", visual_vector),
                limit=limit * 2,
                with_payload=True
            )
            
            for result in visual_results:
                if result.id in results_map:
                    results_map[result.id]["visual_score"] = result.score
                else:
                    results_map[result.id] = {
                        "text_score": 0,
                        "visual_score": result.score,
                        "payload": result.payload
                    }
        
        # Compute fused scores
        search_results = []
        for point_id, data in results_map.items():
            # Linear fusion
            fused_score = (
                fusion_weight * data["text_score"] + 
                (1 - fusion_weight) * data["visual_score"]
            )
            
            payload = data["payload"] or {}
            search_results.append(SearchResult(
                id=str(point_id),
                score=fused_score,
                payload=payload,
                document_id=payload.get("document_id"),
                chunk_number=payload.get("page_number"),
                text=payload.get("caption")
            ))
        
        # Sort by fused score and limit
        search_results.sort(key=lambda x: x.score, reverse=True)
        search_results = search_results[:limit]
        
        logger.info(f"Hybrid search returned {len(search_results)} results")
        return search_results
    
    async def delete_by_document(
        self,
        document_id: str,
        collection_name: str = "document_chunks"
    ) -> int:
        """
        Delete all points for a document
        
        Args:
            document_id: Document ID
            collection_name: Collection name
            
        Returns:
            Number of points deleted
        """
        # First, count points to delete
        count_result = await self.async_client.count(
            collection_name=collection_name,
            count_filter=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )
        )
        
        # Delete points
        result = await self.async_client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )
        )
        
        if result.status == UpdateStatus.COMPLETED:
            logger.info(f"Deleted {count_result.count} points for document {document_id}")
            return count_result.count
        else:
            logger.error(f"Failed to delete points for document {document_id}")
            return 0
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            info = await self.async_client.get_collection(collection_name)
            return {
                "name": collection_name,
                "status": str(info.status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "vectors_config": info.config.params.vectors,
                "disk_size_bytes": getattr(info, "disk_data_size", 0),
                "memory_size_bytes": getattr(info, "ram_data_size", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible"""
        try:
            # Try to get collections as health check
            collections = await self.async_client.get_collections()
            logger.info(f"Qdrant is healthy. Found {len(collections.collections)} collections")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def close(self):
        """Close client connections"""
        await self.async_client.close()
        logger.info("Closed Qdrant connections")