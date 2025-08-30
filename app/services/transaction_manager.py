"""
Transaction management utilities for database operations
"""
from contextlib import contextmanager
from typing import Generator
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


@contextmanager
def database_transaction(db: Session) -> Generator[Session, None, None]:
    """
    Context manager for database transactions with automatic commit/rollback
    
    Usage:
        with database_transaction(db) as session:
            # Perform database operations
            session.add(document)
            session.add(chunk)
            # Automatic commit on success, rollback on exception
    """
    try:
        yield db
        db.commit()
        logger.debug("Transaction committed successfully")
    except Exception as e:
        logger.error(f"Transaction failed, rolling back: {e}")
        db.rollback()
        raise
    finally:
        # Don't close the session here - let FastAPI dependency handle it
        pass


def transactional(func):
    """
    Decorator for transactional operations
    Requires 'db' parameter in the function signature
    """
    def wrapper(*args, **kwargs):
        db = kwargs.get('db')
        if not db:
            raise ValueError("Transaction decorator requires 'db' parameter")
        
        with database_transaction(db):
            return func(*args, **kwargs)
    
    return wrapper