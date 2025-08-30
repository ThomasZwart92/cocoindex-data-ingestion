"""
Fixed FastAPI Main Application
Resolves startup issues with graceful service handling
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
import uuid
from typing import Optional

# Import API routers
from app.api import documents, processing, chunks, entities
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with graceful service handling"""
    # Startup
    logger.info("Starting up application...")
    
    # Try to initialize services but don't fail if they're not available
    services_status = {}
    
    # Neo4j - optional
    try:
        from app.services.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()
        await neo4j.ensure_constraints()
        services_status['neo4j'] = 'connected'
        logger.info("Neo4j connected successfully")
    except Exception as e:
        services_status['neo4j'] = f'unavailable: {str(e)}'
        logger.warning(f"Neo4j connection failed (non-critical): {e}")
    
    # Qdrant - optional
    try:
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        # Ensure the main collection exists
        await qdrant.ensure_collection("document_chunks")
        services_status['qdrant'] = 'connected'
        logger.info("Qdrant connected successfully")
    except Exception as e:
        services_status['qdrant'] = f'unavailable: {str(e)}'
        logger.warning(f"Qdrant connection failed (non-critical): {e}")
    
    # Store service status in app state
    app.state.services_status = services_status
    logger.info(f"Application startup complete. Services: {services_status}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Close connections if they exist
    try:
        if 'neo4j' in locals():
            await neo4j.close()
    except:
        pass
    
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Data Ingestion Portal API",
    description="High-quality document ingestion with human-in-the-loop review",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
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
app.include_router(documents.router)
app.include_router(processing.router)
app.include_router(chunks.router)
app.include_router(entities.router)


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
            "entities": "/api/entities"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint with service status"""
    services_status = getattr(request.app.state, 'services_status', {})
    
    # Determine overall health
    overall_status = "healthy"
    if any('unavailable' in str(status) for status in services_status.values()):
        overall_status = "degraded"
    
    health_status = {
        "status": overall_status,
        "api": "operational",
        "services": services_status
    }
    
    # Add Redis check
    try:
        import redis
        r = redis.from_url(settings.redis_url or "redis://localhost:6379")
        r.ping()
        health_status["services"]["redis"] = "connected"
    except Exception as e:
        health_status["services"]["redis"] = f"unavailable: {str(e)}"
    
    # Add PostgreSQL check
    try:
        from app.services.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["services"]["postgresql"] = "connected"
    except Exception as e:
        health_status["services"]["postgresql"] = f"unavailable: {str(e)}"
    
    return health_status


# Processing endpoints
@app.post("/api/process/notion")
async def trigger_notion_scan():
    """Trigger Notion source scanning."""
    try:
        # In a real implementation, this would trigger an async Celery task
        # For now, we'll return a mock response
        return {
            "status": "success",
            "message": "Notion scan triggered",
            "job_id": str(uuid.uuid4()),
            "count": 0,
            "new": 0,
            "updated": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/gdrive")
async def trigger_gdrive_scan():
    """Trigger Google Drive source scanning."""
    try:
        # In a real implementation, this would trigger an async Celery task
        # For now, we'll return a mock response
        return {
            "status": "success",
            "message": "Google Drive scan triggered",
            "job_id": str(uuid.uuid4()),
            "count": 0,
            "new": 0,
            "updated": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sources/scan")
async def trigger_source_scan(source: str = None):
    """Trigger scanning for a specific source."""
    if source == "notion":
        return await trigger_notion_scan()
    elif source == "gdrive":
        return await trigger_gdrive_scan()
    else:
        # Default behavior when no source specified
        notion_result = await trigger_notion_scan()
        gdrive_result = await trigger_gdrive_scan()
        return {
            "status": "success",
            "message": "All sources scanned",
            "results": {
                "notion": notion_result,
                "gdrive": gdrive_result
            }
        }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
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
        "app.main_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )