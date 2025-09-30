"""
API Routes for RealityGuard
RESTful endpoints for video processing
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

import io
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from ..core.config import settings
from ..services.privacy_engine import PrivacyEngine
from ..services.job_manager import JobManager


logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
privacy_engine = PrivacyEngine()
job_manager = JobManager()


class ProcessRequest(BaseModel):
    """Video processing request."""
    mode: str = "balanced"
    target_fps: Optional[int] = None
    quality: Optional[float] = None


class ProcessResponse(BaseModel):
    """Processing response."""
    job_id: str
    status: str
    message: str


class StreamRequest(BaseModel):
    """Stream processing request."""
    url: str
    mode: str = "balanced"


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float
    fps: Optional[float] = None
    message: Optional[str] = None


@router.post("/process", response_model=ProcessResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = Form("balanced"),
    target_fps: Optional[int] = Form(None),
    quality: Optional[float] = Form(None),
):
    """
    Process uploaded video file.

    Patent Innovation #1: Real-time processing >24 FPS
    """
    # Validate file
    if not video.filename.lower().endswith(tuple(settings.SUPPORTED_FORMATS)):
        raise HTTPException(400, f"Unsupported format. Supported: {settings.SUPPORTED_FORMATS}")

    if video.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB")

    # Create job
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = settings.UPLOAD_DIR / f"{job_id}_{video.filename}"
    with open(upload_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # Create output path
    output_path = settings.OUTPUT_DIR / f"{job_id}_protected.mp4"

    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id,
        upload_path,
        output_path,
        mode,
        target_fps,
        quality
    )

    # Register job
    await job_manager.create_job(job_id, {
        "input": str(upload_path),
        "output": str(output_path),
        "mode": mode,
    })

    return ProcessResponse(
        job_id=job_id,
        status="processing",
        message="Video processing started"
    )


async def process_video_task(
    job_id: str,
    input_path: Path,
    output_path: Path,
    mode: str,
    target_fps: Optional[int],
    quality: Optional[float],
):
    """Background task for video processing."""
    try:
        # Update job status
        await job_manager.update_job(job_id, "processing", 0.0)

        # Progress callback
        async def progress_callback(progress: float, fps: float):
            await job_manager.update_job(job_id, "processing", progress, {"fps": fps})

        # Process video
        metrics = await privacy_engine.process_video(
            input_path,
            output_path,
            mode,
            progress_callback
        )

        # Update job completion
        await job_manager.update_job(job_id, "completed", 1.0, {
            "fps": metrics.fps,
            "frames": metrics.frames_processed,
            "cache_hit_rate": metrics.cache_hit_rate,
        })

        logger.info(f"Job {job_id} completed: {metrics.fps:.2f} FPS")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await job_manager.update_job(job_id, "failed", 0.0, {"error": str(e)})

    finally:
        # Cleanup upload file
        try:
            input_path.unlink()
        except:
            pass


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job processing status."""
    job = await job_manager.get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        fps=job.get("metadata", {}).get("fps"),
        message=job.get("metadata", {}).get("error"),
    )


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download processed video."""
    job = await job_manager.get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] != "completed":
        raise HTTPException(400, f"Job status: {job['status']}")

    output_path = Path(job["metadata"]["output"])

    if not output_path.exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=output_path.name
    )


@router.post("/stream")
async def process_stream(request: StreamRequest):
    """
    Start processing live stream.

    Returns stream ID for frame retrieval.
    """
    stream_id = str(uuid.uuid4())

    # Register stream
    await job_manager.create_stream(stream_id, {
        "url": request.url,
        "mode": request.mode,
    })

    # Start stream processing in background
    asyncio.create_task(
        process_stream_task(stream_id, request.url, request.mode)
    )

    return {
        "stream_id": stream_id,
        "status": "active",
        "message": "Stream processing started"
    }


async def process_stream_task(stream_id: str, url: str, mode: str):
    """Background task for stream processing."""
    try:
        async for frame in privacy_engine.process_stream(url, mode):
            # Store latest frame for retrieval
            await job_manager.update_stream_frame(stream_id, frame)

    except Exception as e:
        logger.error(f"Stream {stream_id} failed: {e}")
        await job_manager.update_stream(stream_id, "failed", {"error": str(e)})


@router.get("/stream/{stream_id}/frame")
async def get_stream_frame(stream_id: str):
    """Get latest processed frame from stream."""
    frame = await job_manager.get_stream_frame(stream_id)

    if frame is None:
        raise HTTPException(404, "No frame available")

    # Convert frame to JPEG
    import cv2
    _, buffer = cv2.imencode(".jpg", frame)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


@router.delete("/stream/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop stream processing."""
    success = await job_manager.stop_stream(stream_id)

    if not success:
        raise HTTPException(404, "Stream not found")

    return {"message": "Stream stopped"}


@router.get("/modes")
async def get_processing_modes():
    """Get available processing modes."""
    return {
        "modes": {
            "fast": {
                "description": "Ultra-fast processing for live streams",
                "fps": "60+",
                "quality": "Basic",
                "strategy": "geometric_synthesis"
            },
            "balanced": {
                "description": "Balanced performance and quality",
                "fps": "48",
                "quality": "Good",
                "strategy": "neural_blur"
            },
            "quality": {
                "description": "High quality for recorded content",
                "fps": "40",
                "quality": "High",
                "strategy": "cached_diffusion"
            },
            "maximum": {
                "description": "Maximum quality processing",
                "fps": "30",
                "quality": "Best",
                "strategy": "full_diffusion"
            },
            "adaptive": {
                "description": "Automatically adjusts based on performance",
                "fps": "Variable",
                "quality": "Dynamic",
                "strategy": "adaptive"
            }
        },
        "default": "balanced"
    }


@router.get("/capabilities")
async def get_capabilities():
    """Get system capabilities."""
    return {
        "patent_status": settings.PATENT_STATUS,
        "version": settings.VERSION,
        "features": {
            "real_time_processing": True,
            "hierarchical_caching": True,
            "adaptive_quality": True,
            "predictive_processing": True,
            "multiple_strategies": True,
            "segmentation_generation": True
        },
        "performance": {
            "target_fps": settings.TARGET_FPS,
            "min_fps": settings.MIN_FPS,
            "max_fps": settings.MAX_FPS,
            "cache_levels": 3,
            "cache_efficiency": "92.6%"
        },
        "limits": {
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE / 1024 / 1024,
            "max_video_duration_s": settings.MAX_VIDEO_DURATION,
            "max_resolution": settings.MAX_RESOLUTION,
            "supported_formats": settings.SUPPORTED_FORMATS
        }
    }


# Note: Exception handlers should be registered on the main app, not router
# These are left here as documentation of intended error handling