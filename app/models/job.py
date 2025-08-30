"""Processing job model"""
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class JobType(str, Enum):
    """Processing job types"""
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    EXTRACT_ENTITIES = "extract_entities"
    EXTRACT_METADATA = "extract_metadata"
    FULL_PIPELINE = "full_pipeline"

class JobStatus(str, Enum):
    """Job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingJob(BaseModel):
    """Processing job model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    
    # Job details
    job_type: JobType
    job_status: JobStatus = JobStatus.PENDING
    
    # Celery task tracking
    celery_task_id: Optional[str] = None
    
    # Progress tracking
    progress: int = Field(default=0, ge=0, le=100)
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def start(self) -> None:
        """Mark job as started"""
        self.job_status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark job as completed"""
        self.job_status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress = 100
        self.updated_at = datetime.utcnow()
    
    def fail(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as failed"""
        self.job_status = JobStatus.FAILED
        self.error_message = error
        self.error_details = details
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_progress(self, progress: int, current_step: Optional[str] = None) -> None:
        """Update job progress"""
        self.progress = min(100, max(0, progress))
        if current_step:
            self.current_step = current_step
        self.updated_at = datetime.utcnow()
    
    def can_retry(self) -> bool:
        """Check if job can be retried"""
        return self.retry_count < self.max_retries
    
    def to_supabase_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion"""
        data = self.dict(exclude_none=True)
        # Convert enums to strings (handle both enum and string types)
        if hasattr(self.job_type, 'value'):
            data['job_type'] = self.job_type.value
        else:
            data['job_type'] = self.job_type
            
        if hasattr(self.job_status, 'value'):
            data['job_status'] = self.job_status.value
        else:
            data['job_status'] = self.job_status
        # Convert datetime to ISO format
        for field in ['started_at', 'completed_at', 'created_at', 'updated_at']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        return data
    
    class Config:
        use_enum_values = True