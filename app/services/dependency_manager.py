"""
Dependency Manager for Service Initialization
Centralized management of external service connections
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service connection status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy" 
    UNAVAILABLE = "unavailable"
    ERROR = "error"

@dataclass
class ServiceHealth:
    """Service health information"""
    name: str
    status: ServiceStatus
    message: Optional[str] = None
    last_check: Optional[str] = None
    
class DependencyManager:
    """Manages all external service dependencies"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.health_status: Dict[str, ServiceHealth] = {}
    
    async def initialize_all(self) -> Dict[str, ServiceHealth]:
        """Initialize all services and return health status"""
        logger.info("Initializing all services...")
        
        await self._init_database()
        await self._init_neo4j()
        await self._init_qdrant()
        await self._init_redis()
        
        # Log summary
        healthy = sum(1 for h in self.health_status.values() if h.status == ServiceStatus.HEALTHY)
        total = len(self.health_status)
        logger.info(f"Service initialization complete: {healthy}/{total} services healthy")
        
        return self.health_status
    
    async def _init_database(self):
        """Initialize PostgreSQL/Supabase connection"""
        try:
            from app.services.database import engine, init_database
            # Test connection
            async with engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            # Initialize tables
            init_database()
            
            self.services["database"] = engine
            self.health_status["database"] = ServiceHealth(
                name="database",
                status=ServiceStatus.HEALTHY,
                message="PostgreSQL connected"
            )
            logger.info("✓ Database connected")
            
        except Exception as e:
            self.health_status["database"] = ServiceHealth(
                name="database",
                status=ServiceStatus.ERROR,
                message=str(e)
            )
            logger.warning(f"⚠️ Database unavailable: {e}")
    
    async def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            from app.services.neo4j_service import Neo4jService
            neo4j = Neo4jService()
            await neo4j.connect()
            await neo4j.ensure_constraints()
            
            self.services["neo4j"] = neo4j
            self.health_status["neo4j"] = ServiceHealth(
                name="neo4j",
                status=ServiceStatus.HEALTHY,
                message="Neo4j connected"
            )
            logger.info("✓ Neo4j connected")
            
        except Exception as e:
            self.health_status["neo4j"] = ServiceHealth(
                name="neo4j",
                status=ServiceStatus.UNAVAILABLE,
                message=str(e)
            )
            logger.warning(f"⚠️ Neo4j unavailable: {e}")
    
    async def _init_qdrant(self):
        """Initialize Qdrant connection"""
        try:
            from app.services.qdrant_service import QdrantService
            qdrant = QdrantService()
            # Test connection by listing collections
            await qdrant.get_collections()
            
            # Ensure default collection exists
            try:
                await qdrant.create_collection("documents")
            except:
                pass  # Collection might already exist
            
            self.services["qdrant"] = qdrant
            self.health_status["qdrant"] = ServiceHealth(
                name="qdrant",
                status=ServiceStatus.HEALTHY,
                message="Qdrant connected"
            )
            logger.info("✓ Qdrant connected")
            
        except Exception as e:
            self.health_status["qdrant"] = ServiceHealth(
                name="qdrant",
                status=ServiceStatus.UNAVAILABLE,
                message=str(e)
            )
            logger.warning(f"⚠️ Qdrant unavailable: {e}")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            from app.config import settings
            
            r = redis.from_url(settings.redis_url)
            await r.ping()
            await r.close()
            
            self.health_status["redis"] = ServiceHealth(
                name="redis",
                status=ServiceStatus.HEALTHY,
                message="Redis connected"
            )
            logger.info("✓ Redis connected")
            
        except Exception as e:
            self.health_status["redis"] = ServiceHealth(
                name="redis",
                status=ServiceStatus.UNAVAILABLE,
                message=str(e)
            )
            logger.warning(f"⚠️ Redis unavailable: {e}")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get service instance by name"""
        return self.services.get(name)
    
    def is_service_healthy(self, name: str) -> bool:
        """Check if service is healthy"""
        health = self.health_status.get(name)
        return health and health.status == ServiceStatus.HEALTHY
    
    async def health_check(self) -> Dict[str, ServiceHealth]:
        """Perform health check on all services"""
        # Could implement periodic health checks here
        return self.health_status
    
    async def shutdown_all(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down all services...")
        
        # Close Neo4j
        neo4j = self.services.get("neo4j")
        if neo4j:
            try:
                await neo4j.close()
                logger.info("✓ Neo4j closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j: {e}")
        
        # Close database connections
        database = self.services.get("database")
        if database:
            try:
                await database.dispose()
                logger.info("✓ Database closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        
        logger.info("All services shut down")

# Global dependency manager instance
dependency_manager = DependencyManager()