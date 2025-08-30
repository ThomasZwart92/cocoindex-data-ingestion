"""
Database service for managing database connections
"""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Create engine - use PostgreSQL on port 5432 if not specified
database_url = settings.database_url or "postgresql://postgres:postgres@localhost:5432/cocoindex"
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def get_db_session() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# For SQLAlchemy models, we need actual table definitions
# These would normally be in separate files but adding them here for simplicity
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime


class DocumentTable(Base):
    """Document database table"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)  # Changed from title to name
    source_type = Column(String, nullable=False)
    source_id = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    file_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    status = Column(String, default="discovered")
    processing_error = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    parse_tier = Column(String, nullable=True)
    parse_confidence = Column(Float, nullable=True)
    doc_metadata = Column("metadata", JSON, default={})  # Map to metadata column in DB
    tags = Column(JSON, default=[])  # Using JSON instead of ARRAY for compatibility
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(UUID(as_uuid=True), nullable=True)
    version = Column(Integer, default=1)
    parent_document_id = Column(UUID(as_uuid=True), nullable=True)


class DocumentChunkTable(Base):
    """Document chunk database table"""
    __tablename__ = "chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    chunk_index = Column(Integer, nullable=False)  # Changed from chunk_number
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer)
    start_position = Column(Integer, default=0)
    end_position = Column(Integer, default=0)
    chunk_metadata = Column("metadata", JSON, default={})  # Map to metadata column in DB
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


class EntityTable(Base):
    """Entity database table"""
    __tablename__ = "entities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    confidence = Column(Float, default=1.0)
    entity_metadata = Column("metadata", JSON, default={})  # Map to metadata column in DB
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


# Map Pydantic models to SQLAlchemy models for compatibility
Document = DocumentTable
DocumentChunk = DocumentChunkTable
Entity = EntityTable