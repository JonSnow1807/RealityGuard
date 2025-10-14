#!/usr/bin/env python3
"""
RealityGuard Production System
Patent-Protected AI Privacy Protection System
Author: Chinmay Shrivastava
Patent: Pending (Filed Sept 27, 2025)
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.routes import router
from src.core.config import settings
from src.core.logging_config import setup_logging
from src.core.metrics import metrics_manager
from src.services.health_checker import HealthChecker

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RealityGuard API",
    description="AI-Powered Privacy Protection System",
    version="1.0.0",
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Health checker
health_checker = HealthChecker()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RealityGuard Production System")
    logger.info(f"Patent Status: {settings.PATENT_STATUS}")

    # Initialize metrics
    await metrics_manager.initialize()

    # Start health monitoring
    asyncio.create_task(health_checker.start_monitoring())

    # Warm up ML models
    from src.services.privacy_engine import PrivacyEngine
    engine = PrivacyEngine()
    await engine.warmup()

    logger.info("System initialization complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RealityGuard")

    # Stop health monitoring
    await health_checker.stop_monitoring()

    # Flush metrics
    await metrics_manager.flush()

    # Cleanup resources
    from src.services.privacy_engine import PrivacyEngine
    engine = PrivacyEngine()
    await engine.cleanup()

    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RealityGuard",
        "status": "operational",
        "patent": settings.PATENT_STATUS,
        "version": "1.0.0",
        "fps_target": settings.TARGET_FPS,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return await health_checker.get_status()


@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    return await metrics_manager.get_metrics()


if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ACCESS_LOG,
        use_colors=False,
        server_header=False,  # Security: Don't expose server info
        date_header=False,    # Security: Don't expose date
    )