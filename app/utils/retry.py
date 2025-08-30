"""
Retry utilities for external API calls
Extracted from LLM service for reuse across all services
"""
import asyncio
import time
import logging
from typing import TypeVar, Callable, Any
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Standard retry decorator for external API calls
def retry_on_failure(
    max_attempts: int = 3,
    min_wait: int = 4,
    max_wait: int = 60,
    exceptions: tuple = (TimeoutError, ConnectionError, Exception)
):
    """
    Decorator for retrying failed operations with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        exceptions: Tuple of exceptions to retry on
    
    Usage:
        @retry_on_failure()
        async def fetch_data():
            # API call that might fail
            pass
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before=before_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


# Async retry wrapper for non-decorated functions
async def retry_async(
    func: Callable,
    *args,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff
    
    Args:
        func: Async function to retry
        *args: Positional arguments for the function
        max_attempts: Maximum number of attempts
        backoff_factor: Multiplier for wait time
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result from the function
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    wait_time = initial_wait
    
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_attempts} for {func.__name__}")
            
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"Retry successful for {func.__name__}")
            
            return result
            
        except Exception as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
            )
            
            if attempt < max_attempts - 1:
                sleep_time = min(wait_time, max_wait)
                logger.info(f"Waiting {sleep_time:.1f}s before retry...")
                await asyncio.sleep(sleep_time)
                wait_time *= backoff_factor
    
    logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
    raise last_exception


# Synchronous retry wrapper
def retry_sync(
    func: Callable,
    *args,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    **kwargs
) -> Any:
    """
    Retry a synchronous function with exponential backoff
    
    Args:
        func: Function to retry
        *args: Positional arguments for the function
        max_attempts: Maximum number of attempts
        backoff_factor: Multiplier for wait time
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result from the function
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    wait_time = initial_wait
    
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_attempts} for {func.__name__}")
            
            result = func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"Retry successful for {func.__name__}")
            
            return result
            
        except Exception as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
            )
            
            if attempt < max_attempts - 1:
                sleep_time = min(wait_time, max_wait)
                logger.info(f"Waiting {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)
                wait_time *= backoff_factor
    
    logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
    raise last_exception