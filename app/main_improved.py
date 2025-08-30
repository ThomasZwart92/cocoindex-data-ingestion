"""
Improved FastAPI Main Application
- Centralized dependency management
- Graceful service degradation
- Better error handling and logging
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager

from app.api import documents, processing, chunks, entities
from app.config import settings
from app.services.dependency_manager import dependency_manager

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
    logger.info("🚀 Starting Data Ingestion Portal API...")
    
    # Initialize all services
    health_status = await dependency_manager.initialize_all()
    
    # Log startup summary
    healthy_services = [name for name, health in health_status.items() 
                       if health.status.value == "healthy"]
    logger.info(f"✅ Startup complete - {len(healthy_services)} services healthy: {healthy_services}")
    
    if not healthy_services:
        logger.warning("⚠️ No services are healthy - API will have limited functionality")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")
    await dependency_manager.shutdown_all()
    logger.info("👋 Shutdown complete")

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
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing and logging middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request processing time and logging"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    # Log response
    status_emoji = "✅" if response.status_code < 400 else "❌"
    logger.info(f"📤 {status_emoji} {response.status_code} {request.method} {request.url.path} ({process_time:.2f}ms)")
    
    # Log slow requests
    if process_time > 200:
        logger.warning(f"🐌 Slow request: {request.method} {request.url.path} took {process_time:.2f}ms")
    
    return response

# Include routers
app.include_router(documents.router)
app.include_router(processing.router)
app.include_router(chunks.router)
app.include_router(entities.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service status"""
    health_status = await dependency_manager.health_check()
    
    return {
        "name": "Data Ingestion Portal API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "services": {
            name: health.status.value 
            for name, health in health_status.items()
        },
        "endpoints": {
            "documents": "/api/documents",
            "processing": "/api/process", 
            "chunks": "/api/chunks",
            "entities": "/api/entities"
        }
    }

# Comprehensive health check endpoint
@app.get("/health")
async def health_check():
    """Detailed health check with service status"""
    health_status = await dependency_manager.health_check()
    
    # Determine overall status
    healthy_count = sum(1 for h in health_status.values() if h.status.value == "healthy")
    total_count = len(health_status)
    
    if healthy_count == total_count:
        overall_status = "healthy"
    elif healthy_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "services": {
            name: {
                "status": health.status.value,
                "message": health.message
            }
            for name, health in health_status.items()
        },
        "summary": {
            "healthy": healthy_count,
            "total": total_count,
            "percentage": round((healthy_count / total_count) * 100, 1) if total_count > 0 else 0
        }
    }

# Service-specific health endpoints
@app.get("/health/{service_name}")
async def service_health(service_name: str):
    """Get health status for specific service"""
    health_status = await dependency_manager.health_check()
    
    if service_name not in health_status:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    health = health_status[service_name]
    return {
        "service": service_name,
        "status": health.status.value,
        "message": health.message,
        "last_check": health.last_check
    }

# Error handlers with better logging
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    logger.warning(f"📍 404 Not Found: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"💥 500 Internal Error: {request.method} {request.url.path} - {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": str(request.url.path),
            "method": request.method
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    """Handle validation errors"""
    logger.warning(f"📝 422 Validation Error: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_improved:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )