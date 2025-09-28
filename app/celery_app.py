"""Celery application configuration"""
from celery import Celery
import os
from app.config import settings

# Create Celery app
celery_app = Celery(
    "ingestion_portal",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Prefer thread/solo pool on Windows to avoid prefork issues
if os.name == "nt":
    # Use threads by default on Windows; can be overridden via CELERY_CONCURRENCY
    celery_app.conf.worker_pool = "threads"
    try:
        celery_app.conf.worker_concurrency = int(os.getenv("CELERY_CONCURRENCY", "4"))
    except Exception:
        celery_app.conf.worker_concurrency = 4
