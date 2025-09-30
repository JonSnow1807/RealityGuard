"""
Configuration Management for RealityGuard
Production-ready settings with environment variable support
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # Application Info
    APP_NAME: str = "RealityGuard"
    PATENT_STATUS: str = "Patent Pending - Application #[PENDING]"
    VERSION: str = "1.0.0"

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    RELOAD: bool = Field(default=False, env="RELOAD")

    # Security
    SECRET_KEY: str = Field(default="", env="SECRET_KEY")
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "https://realityguard.ai"],
        env="ALLOWED_ORIGINS"
    )
    MAX_UPLOAD_SIZE: int = Field(default=500 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 500MB

    # Performance Settings
    TARGET_FPS: int = Field(default=30, env="TARGET_FPS")
    MIN_FPS: int = Field(default=24, env="MIN_FPS")
    MAX_FPS: int = Field(default=60, env="MAX_FPS")

    # Cache Configuration
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")
    L1_CACHE_SIZE: int = Field(default=50, env="L1_CACHE_SIZE")
    L2_CACHE_SIZE: int = Field(default=100, env="L2_CACHE_SIZE")
    L3_CACHE_SIZE: int = Field(default=200, env="L3_CACHE_SIZE")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # seconds

    # Quality Settings
    ENABLE_ADAPTIVE_QUALITY: bool = Field(default=True, env="ENABLE_ADAPTIVE_QUALITY")
    MIN_QUALITY: float = Field(default=0.3, env="MIN_QUALITY")
    MAX_QUALITY: float = Field(default=1.0, env="MAX_QUALITY")
    DEFAULT_QUALITY: float = Field(default=0.7, env="DEFAULT_QUALITY")

    # ML Model Settings
    MODEL_PATH: Path = Field(default=Path("models"), env="MODEL_PATH")
    YOLO_MODEL: str = Field(default="yolov8n-seg.pt", env="YOLO_MODEL")
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    GPU_DEVICE: int = Field(default=0, env="GPU_DEVICE")

    # Processing Modes
    DEFAULT_MODE: str = Field(default="balanced", env="DEFAULT_MODE")
    ENABLE_PREDICTIVE: bool = Field(default=True, env="ENABLE_PREDICTIVE")
    PREDICTION_WINDOW: int = Field(default=5, env="PREDICTION_WINDOW")

    # Storage
    UPLOAD_DIR: Path = Field(default=Path("uploads"), env="UPLOAD_DIR")
    OUTPUT_DIR: Path = Field(default=Path("outputs"), env="OUTPUT_DIR")
    TEMP_DIR: Path = Field(default=Path("temp"), env="TEMP_DIR")

    # Database (for future use)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    ENABLE_TRACING: bool = Field(default=False, env="ENABLE_TRACING")
    JAEGER_HOST: Optional[str] = Field(default=None, env="JAEGER_HOST")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    ACCESS_LOG: bool = Field(default=True, env="ACCESS_LOG")
    LOG_FILE: Optional[Path] = Field(default=None, env="LOG_FILE")

    # API Settings
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX")
    ENABLE_DOCS: bool = Field(default=True, env="ENABLE_DOCS")
    RATE_LIMIT: int = Field(default=100, env="RATE_LIMIT")  # requests per minute

    # Video Processing Limits
    MAX_VIDEO_DURATION: int = Field(default=300, env="MAX_VIDEO_DURATION")  # seconds
    MAX_RESOLUTION: tuple = Field(default=(3840, 2160), env="MAX_RESOLUTION")  # 4K
    SUPPORTED_FORMATS: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv", "webm"],
        env="SUPPORTED_FORMATS"
    )

    @field_validator("SECRET_KEY", mode="before")
    def validate_secret_key(cls, v):
        """Ensure secret key is set in production."""
        if not v and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("SECRET_KEY must be set in production")
        return v or "dev-secret-key-change-in-production"

    @field_validator("MODEL_PATH", "UPLOAD_DIR", "OUTPUT_DIR", "TEMP_DIR", mode="before")
    def ensure_paths_exist(cls, v):
        """Create directories if they don't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Export configuration groups for easy access
CACHE_CONFIG = {
    "enabled": settings.ENABLE_CACHE,
    "l1_size": settings.L1_CACHE_SIZE,
    "l2_size": settings.L2_CACHE_SIZE,
    "l3_size": settings.L3_CACHE_SIZE,
    "ttl": settings.CACHE_TTL,
}

QUALITY_CONFIG = {
    "adaptive": settings.ENABLE_ADAPTIVE_QUALITY,
    "min": settings.MIN_QUALITY,
    "max": settings.MAX_QUALITY,
    "default": settings.DEFAULT_QUALITY,
}

PERFORMANCE_CONFIG = {
    "target_fps": settings.TARGET_FPS,
    "min_fps": settings.MIN_FPS,
    "max_fps": settings.MAX_FPS,
    "predictive": settings.ENABLE_PREDICTIVE,
    "prediction_window": settings.PREDICTION_WINDOW,
}