"""
Metrics collection and management for RealityGuard
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest


@dataclass
class MetricEntry:
    """Single metric entry."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsManager:
    """Manages application metrics."""

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize metrics manager."""
        if not hasattr(self, "initialized"):
            # Prometheus metrics
            self.fps_gauge = Gauge("realityguard_fps", "Current FPS")
            self.frame_time_histogram = Histogram(
                "realityguard_frame_time_ms",
                "Frame processing time in milliseconds",
                buckets=(5, 10, 25, 50, 75, 100, 250, 500, 750, 1000)
            )
            self.cache_hit_counter = Counter(
                "realityguard_cache_hits",
                "Cache hit count",
                ["level"]
            )
            self.frames_processed_counter = Counter(
                "realityguard_frames_processed",
                "Total frames processed"
            )
            self.quality_gauge = Gauge(
                "realityguard_quality_level",
                "Current quality level"
            )
            self.adaptations_counter = Counter(
                "realityguard_quality_adaptations",
                "Quality adaptations count"
            )

            # Internal metrics storage
            self.metrics: Dict[str, List[MetricEntry]] = defaultdict(list)
            self.max_entries = 1000
            self.initialized = True

    async def initialize(self):
        """Initialize metrics collection."""
        # Start background metrics collection
        asyncio.create_task(self._collect_system_metrics())

    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                # Collect GPU metrics if available
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.record("gpu.memory_used", gpu.memoryUsed)
                        self.record("gpu.memory_free", gpu.memoryFree)
                        self.record("gpu.utilization", gpu.load * 100)
                        self.record("gpu.temperature", gpu.temperature)
                except ImportError:
                    pass

                # Collect CPU metrics
                try:
                    import psutil
                    self.record("cpu.percent", psutil.cpu_percent())
                    self.record("memory.percent", psutil.virtual_memory().percent)
                except ImportError:
                    pass

            except Exception as e:
                print(f"Error collecting system metrics: {e}")

            await asyncio.sleep(10)  # Collect every 10 seconds

    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        entry = MetricEntry(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )

        self.metrics[metric_name].append(entry)

        # Limit stored entries
        if len(self.metrics[metric_name]) > self.max_entries:
            self.metrics[metric_name] = self.metrics[metric_name][-self.max_entries:]

        # Update Prometheus metrics
        self._update_prometheus_metrics(metric_name, value, labels)

    def _update_prometheus_metrics(self, metric_name: str, value: float, labels: Optional[Dict[str, str]]):
        """Update Prometheus metrics based on metric name."""
        if metric_name == "fps.current":
            self.fps_gauge.set(value)
        elif metric_name == "frame.time":
            self.frame_time_histogram.observe(value)
        elif metric_name.startswith("cache.") and metric_name.endswith(".hit"):
            level = metric_name.split(".")[1]
            self.cache_hit_counter.labels(level=level).inc()
        elif metric_name == "frames.processed":
            self.frames_processed_counter.inc()
        elif metric_name == "quality.level":
            self.quality_gauge.set(value)
        elif metric_name == "quality.adaptations":
            self.adaptations_counter.inc()

    async def get_metrics(self) -> Dict:
        """Get current metrics summary."""
        summary = {}

        for metric_name, entries in self.metrics.items():
            if entries:
                recent_entries = entries[-100:]  # Last 100 entries
                values = [e.value for e in recent_entries]

                summary[metric_name] = {
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(entries)
                }

        # Add Prometheus metrics export
        summary["prometheus"] = generate_latest().decode("utf-8")

        return summary

    async def flush(self):
        """Flush metrics to persistent storage if configured."""
        # Could implement writing to database or file
        pass

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


# Global metrics manager instance
metrics_manager = MetricsManager()