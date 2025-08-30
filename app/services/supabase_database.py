"""
Supabase database service for managing database connections
This replaces the local PostgreSQL with Supabase's PostgreSQL
"""
from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
import logging
import os

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Build Supabase PostgreSQL URL
# Format: postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres
def get_supabase_database_url() -> str:
    """Build Supabase database URL from environment variables"""
    # Use environment variable or fallback to our config
    # Try the correct format for Supabase
    
    # Option 1: Try using DATABASE_URL from env if it's updated
    database_url = os.getenv("DATABASE_URL")
    if database_url and "supabase" in database_url:
        logger.info(f"Using DATABASE_URL from environment")
        return database_url
    
    # Option 2: Build from components
    password = os.getenv("SUPABASE_DB_PASSWORD", "vq32NaCZ9Au1NZKu")
    
    # For Supabase, the format varies:
    # Direct: postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
    # Pooler: postgresql://postgres.[project-ref]:[password]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
    
    # Let's use the pooler with correct format
    project_ref = "ycddtddyffnqqehmjsid"
    
    # Try pooler with correct user format
    host = "aws-0-us-east-1.pooler.supabase.com"
    port = 5432
    database = "postgres"
    user = f"postgres.{project_ref}"
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{database}?pgbouncer=true"
    logger.info(f"Using Supabase pooler at {host}:{port}")
    return url

# Create engine
database_url = get_supabase_database_url()
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    echo=False,
    pool_size=5,
    max_overflow=10
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize database tables in Supabase"""
    try:
        # Import all models to ensure they're registered
        from app.services.database import (
            DocumentTable, 
            DocumentChunkTable, 
            EntityTable,
            RelationshipTable,
            ProcessingJobTable,
            DocumentStateTable
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Supabase database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create Supabase database tables: {e}")
        raise


def get_db_session() -> Generator[Session, None, None]:
    """Get database session for Supabase"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test the Supabase database connection"""
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT version()")
            version = result.scalar()
            logger.info(f"Connected to Supabase PostgreSQL: {version}")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False