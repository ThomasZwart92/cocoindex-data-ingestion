"""Configuration management for the application"""
import os
import json
from typing import Optional, Dict, List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = environment == "development"
    
    # Database URLs
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = "document_embeddings"
    
    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    
    # LlamaParse
    llamaparse_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY", "")
    llamaparse_base_url: str = os.getenv("LLAMA_PARSE_BASE_URL", "https://api.cloud.llamaindex.ai/api/v1")
    
    # LLM APIs
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_ai_api_key: str = os.getenv("GOOGLE_AI_API_KEY", "")
    
    # Processing settings
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Notion settings - Multiple security levels
    # Security levels: public, client, partner, employee, management
    notion_tokens: Dict[str, str] = {
        "public": os.getenv("NOTION_API_KEY_PUBLIC_ACCESS", ""),
        "client": os.getenv("NOTION_API_KEY_CLIENT_ACCESS", ""),
        "partner": os.getenv("NOTION_API_KEY_PARTNER_ACCESS", ""),
        "employee": os.getenv("NOTION_API_KEY_EMPLOYEE_ACCESS", ""),
        "management": os.getenv("NOTION_API_KEY_MANAGEMENT_ACCESS", "")
    }
    
    # Legacy single token support (defaults to employee level)
    notion_api_key: str = os.getenv("NOTION_API_KEY", "") or os.getenv("NOTION_API_KEY_EMPLOYEE_ACCESS", "")
    notion_database_ids: List[str] = json.loads(os.getenv("NOTION_DATABASE_IDS", "[]"))
    
    # Security level hierarchy (higher number = more access)
    security_levels: Dict[str, int] = {
        "public": 1,      # Public website, marketing materials
        "client": 2,      # Client-facing documentation, FAQs
        "partner": 3,     # Partner integrations, API docs
        "employee": 4,    # Internal docs, support tickets
        "management": 5   # Strategic plans, sensitive data
    }
    
    # Google Drive settings
    google_drive_credentials_path: str = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "")
    google_drive_folder_ids: List[str] = json.loads(os.getenv("GOOGLE_DRIVE_FOLDER_IDS", "[]"))
    
    # CORS origins
    cors_origins: List[str] = json.loads(os.getenv(
        "CORS_ORIGINS",
        '["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"]'
    ))
    
    # Celery settings
    celery_broker_url: str = redis_url
    celery_result_backend: str = redis_url
    celery_task_time_limit: int = 3600  # 1 hour
    celery_task_soft_time_limit: int = 3300  # 55 minutes

    # Entity pipeline controls
    entity_pipeline_version: str = os.getenv("ENTITY_PIPELINE_VERSION", "v2")
    legacy_entity_extractor_enabled: bool = os.getenv("COCOINDEX_LEGACY_ENTITY_EXTRACTOR", "0") == "1"
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env

# Create global settings instance
settings = Settings()
