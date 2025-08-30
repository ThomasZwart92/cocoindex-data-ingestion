"""
Job Tracker Service for Async Task Management
"""
from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobTracker:
    """
    Simple in-memory job tracker
    In production, this should use Redis or a database
    """
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, job_id: str, job_type: str, metadata: Optional[dict] = None) -> dict:
        """Create a new job"""
        job = {
            "id": job_id,
            "type": job_type,
            "status": JobStatus.QUEUED,
            "progress": 0,
            "message": "Job queued",
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error": None,
            "result": {}
        }
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} of type {job_type}")
        return job
    
    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[dict] = None
    ) -> Optional[dict]:
        """Update job status and details"""
        if job_id not in self.jobs:
            logger.warning(f"Job {job_id} not found for update")
            return None
        
        job = self.jobs[job_id]
        job["status"] = status
        job["updated_at"] = datetime.utcnow().isoformat()
        
        if message is not None:
            job["message"] = message
        if progress is not None:
            job["progress"] = min(100, max(0, progress))
        if error is not None:
            job["error"] = error
        if result is not None:
            job["result"] = result
        
        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job["completed_at"] = datetime.utcnow().isoformat()
            if status == JobStatus.COMPLETED:
                job["progress"] = 100
        
        logger.info(f"Updated job {job_id}: status={status}, progress={job.get('progress')}%")
        return job
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details"""
        return self.jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[dict]:
        """List jobs with optional filtering"""
        jobs = list(self.jobs.values())
        
        # Filter by status
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        
        # Filter by type
        if job_type:
            jobs = [j for j in jobs if j["type"] == job_type]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        return jobs[offset:offset + limit]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job["status"] = JobStatus.CANCELLED
        job["completed_at"] = datetime.utcnow().isoformat()
        job["message"] = "Job cancelled by user"
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def cleanup_old_jobs(self, hours: int = 24):
        """Remove completed jobs older than specified hours"""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        to_remove = []
        
        for job_id, job in self.jobs.items():
            if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.get("completed_at"):
                    completed_time = datetime.fromisoformat(job["completed_at"]).timestamp()
                    if completed_time < cutoff:
                        to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")
        
        return len(to_remove)