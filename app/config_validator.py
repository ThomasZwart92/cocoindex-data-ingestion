"""Configuration validation to ensure required settings are provided"""
import sys
from typing import List
from app.config import settings

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass

def validate_config() -> None:
    """
    Validate that all required configuration values are present.
    Raises ConfigurationError if any required values are missing.
    """
    errors: List[str] = []
    
    # Required for basic operation
    required_configs = {
        "database_url": "DATABASE_URL",
        "redis_url": "REDIS_URL",
        "supabase_url": "SUPABASE_URL",
        "supabase_key": "SUPABASE_KEY",
        "qdrant_url": "QDRANT_URL",
        "neo4j_uri": "NEO4J_URI",
        "neo4j_user": "NEO4J_USER",
        "neo4j_password": "NEO4J_PASSWORD",
    }
    
    # Check required configs
    for attr, env_var in required_configs.items():
        value = getattr(settings, attr, None)
        if not value:
            errors.append(f"Missing required environment variable: {env_var}")
    
    # Required for document processing
    if not settings.llamaparse_api_key:
        errors.append("Missing LLAMA_CLOUD_API_KEY for document parsing")
    
    # At least one LLM API key required
    if not settings.openai_api_key and not settings.google_ai_api_key:
        errors.append("At least one LLM API key required: OPENAI_API_KEY or GOOGLE_AI_API_KEY")
    
    # Raise error with all missing configs
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ConfigurationError(error_message)

def get_config_status() -> dict:
    """Get status of all configuration values for debugging"""
    return {
        "environment": settings.environment,
        "database_configured": bool(settings.database_url),
        "redis_configured": bool(settings.redis_url),
        "supabase_configured": bool(settings.supabase_url and settings.supabase_key),
        "qdrant_configured": bool(settings.qdrant_url),
        "neo4j_configured": bool(settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password),
        "llamaparse_configured": bool(settings.llamaparse_api_key),
        "openai_configured": bool(settings.openai_api_key),
        "google_ai_configured": bool(settings.google_ai_api_key),
    }

# Validate on import in production
if settings.environment == "production":
    try:
        validate_config()
    except ConfigurationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)