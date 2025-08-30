"""
Minimal FastAPI Main Application (without database startup)
For testing API structure without external dependencies
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.api import documents, processing, chunks, entities
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app without lifespan dependency
app = FastAPI(
    title="Data Ingestion Portal API",
    description="High-quality document ingestion with human-in-the-loop review",
    version="1.0.0"
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
        "mode": "minimal",
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
async def health_check():
    """Health check endpoint (minimal version)"""
    return {
        "status": "healthy",
        "mode": "minimal",
        "services": {
            "api": "healthy",
            "database": "skipped",
            "neo4j": "skipped",
            "qdrant": "skipped",
            "redis": "skipped"
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
        "app.main_minimal:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )