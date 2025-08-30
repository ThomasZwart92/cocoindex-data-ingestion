"""Embedding Service for generating text embeddings using OpenAI"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from openai import OpenAI, AsyncOpenAI
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: List[float]
    model: str
    dimensions: int
    latency_ms: int
    cost_estimate: float = 0.0
    
class EmbeddingService:
    """Service for generating text embeddings"""
    
    # Supported embedding models with dimensions and pricing
    MODELS = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "price_per_1k_tokens": 0.00002  # $0.020 per 1M tokens
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "price_per_1k_tokens": 0.00013  # $0.130 per 1M tokens
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "price_per_1k_tokens": 0.00010  # $0.100 per 1M tokens
        }
    }
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding service
        
        Args:
            model: OpenAI embedding model to use
        """
        if model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}. Choose from {list(self.MODELS.keys())}")
            
        self.model = model
        self.model_info = self.MODELS[model]
        
        # Initialize OpenAI client
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
            
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        logger.info(f"Initialized EmbeddingService with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def embed_text(
        self,
        text: str,
        timeout: int = 30
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            timeout: Request timeout in seconds
            
        Returns:
            EmbeddingResult with embedding vector
        """
        start_time = time.time()
        
        try:
            # Use asyncio timeout
            async with asyncio.timeout(timeout):
                response = await self.async_client.embeddings.create(
                    input=text,
                    model=self.model
                )
            
            embedding = response.data[0].embedding
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Estimate cost (approximate token count)
            token_count = len(text.split()) * 1.3  # Rough estimate
            cost = (token_count / 1000) * self.model_info["price_per_1k_tokens"]
            
            return EmbeddingResult(
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms,
                cost_estimate=round(cost, 8)
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Embedding generation timed out after {timeout} seconds")
            raise TimeoutError(f"Embedding generation timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 20,
        timeout_per_batch: int = 60
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batches with retry logic
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (max 2048 for OpenAI)
            timeout_per_batch: Timeout for each batch request
            
        Returns:
            List of EmbeddingResults
        """
        if batch_size > 2048:
            logger.warning(f"Batch size {batch_size} exceeds OpenAI limit, setting to 2048")
            batch_size = 2048
            
        results = []
        total_cost = 0.0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            start_time = time.time()
            
            try:
                async with asyncio.timeout(timeout_per_batch):
                    response = await self.async_client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Calculate cost for batch
                token_count = sum(len(text.split()) * 1.3 for text in batch)
                batch_cost = (token_count / 1000) * self.model_info["price_per_1k_tokens"]
                total_cost += batch_cost
                
                # Create results for each embedding
                for j, data in enumerate(response.data):
                    results.append(EmbeddingResult(
                        embedding=data.embedding,
                        model=self.model,
                        dimensions=len(data.embedding),
                        latency_ms=latency_ms // len(batch),  # Distribute latency
                        cost_estimate=round(batch_cost / len(batch), 8)
                    ))
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except asyncio.TimeoutError:
                logger.error(f"Batch embedding timed out for items {i} to {i+len(batch)}")
                # Add empty results for failed batch
                for _ in batch:
                    results.append(EmbeddingResult(
                        embedding=[0.0] * self.model_info["dimensions"],
                        model=self.model,
                        dimensions=self.model_info["dimensions"],
                        latency_ms=0,
                        cost_estimate=0.0
                    ))
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise
        
        logger.info(f"Generated {len(results)} embeddings. Total cost: ${total_cost:.6f}")
        return results
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric (cosine, euclidean, dot)
            
        Returns:
            Similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
            
        elif metric == "euclidean":
            # Euclidean distance (inverted to similarity)
            distance = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + distance))
            
        elif metric == "dot":
            # Dot product
            return float(np.dot(vec1, vec2))
            
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    async def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "chunk_text",
        batch_size: int = 20
    ) -> List[Tuple[Dict[str, Any], EmbeddingResult]]:
        """
        Embed a list of chunks
        
        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text to embed
            batch_size: Batch size for embedding
            
        Returns:
            List of (chunk, embedding_result) tuples
        """
        # Extract texts from chunks
        texts = [chunk.get(text_field, "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.embed_batch(texts, batch_size)
        
        # Combine chunks with embeddings
        results = list(zip(chunks, embeddings))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            "model": self.model,
            "dimensions": self.model_info["dimensions"],
            "price_per_1k_tokens": self.model_info["price_per_1k_tokens"],
            "price_per_1m_tokens": self.model_info["price_per_1k_tokens"] * 1000
        }