"""
FastAPI Main Application
Entry point for the data ingestion portal API
"""
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager

from app.api import documents, documents_review, processing, chunks, entities, relationships, bridge, sse, search, query
# Temporarily disabled due to import issues: search, ingestion
from app.config import settings
from app.services.neo4j_service import Neo4jService
from app.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up application...")
    
    # Initialize database connections
    try:
        neo4j = Neo4jService()
        await neo4j.connect()
        await neo4j.ensure_constraints()
    except Exception as e:
        logger.warning(f"Neo4j connection failed (continuing without it): {e}")
    
    qdrant = QdrantService()
    # Ensure collections exist
    try:
        await qdrant.create_collection("documents")
    except:
        pass  # Collection might already exist
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    try:
        await neo4j.close()
    except:
        pass  # Neo4j might not have been initialized
    logger.info("Application shutdown complete")


# Create FastAPI app
tags_metadata = [
    {"name": "documents", "description": "Manage documents: list, fetch, update, process, and progress."},
    {"name": "Document Review", "description": "Approve or reject processed documents; check review status."},
    {"name": "processing", "description": "Processing jobs and streaming status updates."},
    {"name": "chunks", "description": "Chunk management: view and edit chunks for documents."},
    {"name": "entities", "description": "Entity management: create, update, delete, and quality review."},
    {"name": "relationships", "description": "Create and manage relationships between entities."},
    {"name": "bridge", "description": "Bridge endpoints for graph/search integrations (Supabase/Qdrant/Neo4j)."},
    {"name": "sse", "description": "Server-Sent Events streams for realtime updates."},
    {"name": "search", "description": "Search endpoints (hybrid vector + BM25)."},
]

app = FastAPI(
    title="Data Ingestion Portal API",
    description="High-quality document ingestion with human-in-the-loop review",
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    # Log slow requests
    if process_time > 200:
        logger.warning(f"Slow request: {request.method} {request.url.path} took {process_time:.2f}ms")
    
    return response


# Include routers
# Enable search API
app.include_router(search.router)
# Temporarily disabled: app.include_router(ingestion.router)
app.include_router(documents.router)
app.include_router(documents_review.router, prefix="/api/documents", tags=["Document Review"])
app.include_router(processing.router)
app.include_router(chunks.router)
app.include_router(entities.router)
app.include_router(relationships.router)
app.include_router(bridge.router)  # API Bridge for Qdrant/Neo4j data
app.include_router(sse.router)  # Server-Sent Events for real-time updates
app.include_router(query.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Data Ingestion Portal API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "documents": "/api/documents",
            "processing": "/api/process",
            "chunks": "/api/chunks",
            "entities": "/api/entities",
            "search": "/api/search",
            "ingestion": "/api/ingestion"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "services": {}
    }
    
    # Check Neo4j
    try:
        neo4j = Neo4jService()
        neo4j_healthy = await neo4j.health_check()
        health_status["services"]["neo4j"] = "healthy" if neo4j_healthy else "unhealthy"
    except Exception as e:
        health_status["services"]["neo4j"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Qdrant
    try:
        qdrant = QdrantService()
        collections = await qdrant.async_client.get_collections()
        health_status["services"]["qdrant"] = "healthy"
    except Exception as e:
        health_status["services"]["qdrant"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis (for Celery)
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.redis_url)
        await r.ping()
        health_status["services"]["redis"] = "healthy"
        await r.close()
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Celery Worker (simplified check)
    try:
        from app.celery_app import celery_app
        import asyncio
        
        # Quick check with timeout to avoid blocking
        loop = asyncio.get_event_loop()
        
        def check_workers():
            try:
                inspect = celery_app.control.inspect(timeout=0.5)
                stats = inspect.stats()
                if stats:
                    return f"healthy ({len(stats)} workers)"
                else:
                    return "no workers running"
            except:
                return "no workers running"
        
        # Run with timeout
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, check_workers),
                timeout=1.0
            )
            health_status["services"]["celery"] = result
            if result == "no workers running":
                health_status["status"] = "degraded"
        except asyncio.TimeoutError:
            health_status["services"]["celery"] = "no workers running"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["celery"] = "no workers running"
        health_status["status"] = "degraded"
    
    return health_status


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    logger.warning(f"404 handler triggered for: {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
