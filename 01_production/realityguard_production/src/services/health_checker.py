"""
Health checking service for RealityGuard
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import aiohttp


logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status of a component."""
    healthy: bool
    message: str
    latency_ms: Optional[float] = None
    last_check: Optional[float] = None


class HealthChecker:
    """Monitors system health."""

    def __init__(self):
        """Initialize health checker."""
        self.components: Dict[str, HealthStatus] = {}
        self.monitoring = False
        self.check_interval = 30  # seconds

    async def start_monitoring(self):
        """Start health monitoring."""
        self.monitoring = True
        logger.info("Health monitoring started")

        while self.monitoring:
            await self._check_all_components()
            await asyncio.sleep(self.check_interval)

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        logger.info("Health monitoring stopped")

    async def _check_all_components(self):
        """Check health of all components."""
        # Check API health
        self.components["api"] = await self._check_api()

        # Check GPU availability
        self.components["gpu"] = await self._check_gpu()

        # Check model loading
        self.components["models"] = await self._check_models()

        # Check cache
        self.components["cache"] = await self._check_cache()

        # Check disk space
        self.components["disk"] = await self._check_disk_space()

    async def _check_api(self) -> HealthStatus:
        """Check API health."""
        try:
            start = time.time()
            # Check if API is responsive (internal check)
            latency = (time.time() - start) * 1000

            return HealthStatus(
                healthy=True,
                message="API is responsive",
                latency_ms=latency,
                last_check=time.time()
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"API check failed: {e}",
                last_check=time.time()
            )

    async def _check_gpu(self) -> HealthStatus:
        """Check GPU availability."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_free = torch.cuda.mem_get_info()[0] / 1024**3  # GB

                return HealthStatus(
                    healthy=True,
                    message=f"GPU available: {gpu_name} ({memory_free:.1f}GB free)",
                    last_check=time.time()
                )
            else:
                return HealthStatus(
                    healthy=True,  # Not critical if no GPU
                    message="No GPU available, using CPU",
                    last_check=time.time()
                )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"GPU check failed: {e}",
                last_check=time.time()
            )

    async def _check_models(self) -> HealthStatus:
        """Check if models are loaded."""
        try:
            from ..services.privacy_engine import PrivacyEngine

            engine = PrivacyEngine()
            if hasattr(engine, "segmentation_model"):
                return HealthStatus(
                    healthy=True,
                    message="Models loaded successfully",
                    last_check=time.time()
                )
            else:
                return HealthStatus(
                    healthy=False,
                    message="Models not loaded",
                    last_check=time.time()
                )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Model check failed: {e}",
                last_check=time.time()
            )

    async def _check_cache(self) -> HealthStatus:
        """Check cache health."""
        try:
            from ..services.privacy_engine import PrivacyEngine

            engine = PrivacyEngine()
            if hasattr(engine, "cache"):
                hit_rate = engine.cache.get_hit_rate()
                return HealthStatus(
                    healthy=True,
                    message=f"Cache operational (hit rate: {hit_rate:.1%})",
                    last_check=time.time()
                )
            else:
                return HealthStatus(
                    healthy=False,
                    message="Cache not initialized",
                    last_check=time.time()
                )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Cache check failed: {e}",
                last_check=time.time()
            )

    async def _check_disk_space(self) -> HealthStatus:
        """Check available disk space."""
        try:
            import shutil
            from ..core.config import settings

            # Check upload directory space
            stat = shutil.disk_usage(settings.UPLOAD_DIR)
            free_gb = stat.free / (1024**3)
            used_percent = (stat.used / stat.total) * 100

            if free_gb < 1:  # Less than 1GB free
                return HealthStatus(
                    healthy=False,
                    message=f"Low disk space: {free_gb:.1f}GB free",
                    last_check=time.time()
                )
            else:
                return HealthStatus(
                    healthy=True,
                    message=f"Disk space OK: {free_gb:.1f}GB free ({used_percent:.1f}% used)",
                    last_check=time.time()
                )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Disk check failed: {e}",
                last_check=time.time()
            )

    async def get_status(self) -> Dict:
        """Get current health status."""
        overall_healthy = all(
            status.healthy for status in self.components.values()
            if status.healthy is not None
        )

        return {
            "healthy": overall_healthy,
            "timestamp": time.time(),
            "components": {
                name: {
                    "healthy": status.healthy,
                    "message": status.message,
                    "latency_ms": status.latency_ms,
                    "last_check": status.last_check
                }
                for name, status in self.components.items()
            }
        }