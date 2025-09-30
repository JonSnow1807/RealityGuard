"""
Job management service for async video processing
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


class JobManager:
    """Manages async processing jobs."""

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize job manager."""
        if not hasattr(self, "initialized"):
            self.jobs: Dict[str, Dict] = {}
            self.streams: Dict[str, Dict] = {}
            self.stream_frames: Dict[str, np.ndarray] = {}
            self.initialized = True

    async def create_job(self, job_id: str, metadata: Dict) -> Dict:
        """Create a new job."""
        self.jobs[job_id] = {
            "id": job_id,
            "status": "created",
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "metadata": metadata
        }
        logger.info(f"Created job: {job_id}")
        return self.jobs[job_id]

    async def update_job(
        self,
        job_id: str,
        status: str,
        progress: float,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Update job status."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        self.jobs[job_id]["status"] = status
        self.jobs[job_id]["progress"] = progress
        self.jobs[job_id]["updated_at"] = time.time()

        if metadata:
            self.jobs[job_id]["metadata"].update(metadata)

        logger.debug(f"Updated job {job_id}: status={status}, progress={progress:.1%}")
        return self.jobs[job_id]

    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    async def list_jobs(self, status: Optional[str] = None) -> List[Dict]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j["status"] == status]

        return sorted(jobs, key=lambda x: x["created_at"], reverse=True)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info(f"Deleted job: {job_id}")
            return True
        return False

    async def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Clean up old completed/failed jobs."""
        current_time = time.time()
        to_delete = []

        for job_id, job in self.jobs.items():
            if job["status"] in ["completed", "failed"]:
                age = current_time - job["updated_at"]
                if age > max_age_seconds:
                    to_delete.append(job_id)

        for job_id in to_delete:
            await self.delete_job(job_id)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old jobs")

    # Stream management
    async def create_stream(self, stream_id: str, metadata: Dict) -> Dict:
        """Create a new stream processing job."""
        self.streams[stream_id] = {
            "id": stream_id,
            "status": "active",
            "created_at": time.time(),
            "updated_at": time.time(),
            "metadata": metadata,
            "frames_processed": 0
        }
        logger.info(f"Created stream: {stream_id}")
        return self.streams[stream_id]

    async def update_stream(self, stream_id: str, status: str, metadata: Optional[Dict] = None):
        """Update stream status."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        self.streams[stream_id]["status"] = status
        self.streams[stream_id]["updated_at"] = time.time()

        if metadata:
            self.streams[stream_id]["metadata"].update(metadata)

    async def update_stream_frame(self, stream_id: str, frame: np.ndarray):
        """Update the latest frame for a stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        self.stream_frames[stream_id] = frame
        self.streams[stream_id]["frames_processed"] += 1
        self.streams[stream_id]["updated_at"] = time.time()

    async def get_stream_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a stream."""
        return self.stream_frames.get(stream_id)

    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream."""
        if stream_id in self.streams:
            self.streams[stream_id]["status"] = "stopped"
            self.streams[stream_id]["updated_at"] = time.time()

            # Clean up frame buffer
            if stream_id in self.stream_frames:
                del self.stream_frames[stream_id]

            logger.info(f"Stopped stream: {stream_id}")
            return True
        return False

    async def get_stats(self) -> Dict:
        """Get job manager statistics."""
        job_stats = {}
        for status in ["created", "processing", "completed", "failed"]:
            job_stats[status] = sum(1 for j in self.jobs.values() if j["status"] == status)

        stream_stats = {}
        for status in ["active", "stopped", "failed"]:
            stream_stats[status] = sum(1 for s in self.streams.values() if s["status"] == status)

        return {
            "jobs": {
                "total": len(self.jobs),
                "by_status": job_stats
            },
            "streams": {
                "total": len(self.streams),
                "by_status": stream_stats,
                "active_frames": len(self.stream_frames)
            }
        }