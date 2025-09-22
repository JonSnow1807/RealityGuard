"""Configuration system for RealityGuard privacy protection system."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


class PrivacyMode(Enum):
    """Privacy protection levels."""
    OFF = "off"
    SMART = "smart"
    SOCIAL = "social"
    WORKSPACE = "workspace"
    MAXIMUM = "maximum"


class SafetyMode(Enum):
    """Content safety filtering levels."""
    FAMILY = "family"
    MODERATE = "moderate"
    WORKPLACE = "workplace"
    UNRESTRICTED = "unrestricted"


@dataclass
class DetectionConfig:
    """Configuration for detection systems."""
    # Face detection
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: Tuple[int, int] = (30, 30)
    face_detection_confidence: float = 0.5

    # Screen detection
    screen_brightness_threshold: int = 200
    screen_min_area: int = 5000
    screen_aspect_ratio_range: Tuple[float, float] = (1.0, 2.5)

    # Content detection
    text_confidence_threshold: float = 0.7
    inappropriate_content_threshold: float = 0.8
    skin_exposure_threshold: float = 0.15

    # Performance settings
    frame_skip_interval: int = 2
    detection_cache_duration: int = 30  # frames
    downscale_factor: float = 0.3

    # Blur settings
    blur_kernel_size: Tuple[int, int] = (21, 21)
    pixelation_size: int = 20


@dataclass
class PerformanceConfig:
    """Performance optimization settings."""
    target_fps: int = 120
    enable_gpu: bool = True
    max_memory_mb: int = 512
    thread_count: int = 4
    enable_caching: bool = True
    adaptive_quality: bool = True


@dataclass
class SecurityConfig:
    """Security and privacy settings."""
    log_detections: bool = False
    encrypt_logs: bool = True
    store_faces: bool = False
    auto_delete_cache: bool = True
    cache_ttl_seconds: int = 300


class Config:
    """Main configuration manager for RealityGuard."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_file: Optional path to JSON configuration file
        """
        self.detection = DetectionConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()

        # Default privacy settings per mode
        self.privacy_presets: Dict[PrivacyMode, dict] = {
            PrivacyMode.OFF: {
                "blur_faces": False,
                "blur_screens": False,
                "filter_content": False
            },
            PrivacyMode.SMART: {
                "blur_faces": True,
                "blur_screens": False,
                "filter_content": False,
                "known_faces_exempt": True
            },
            PrivacyMode.SOCIAL: {
                "blur_faces": True,
                "blur_screens": True,
                "filter_content": True,
                "known_faces_exempt": True,
                "content_level": SafetyMode.MODERATE
            },
            PrivacyMode.WORKSPACE: {
                "blur_faces": True,
                "blur_screens": True,
                "filter_content": True,
                "known_faces_exempt": False,
                "content_level": SafetyMode.WORKPLACE
            },
            PrivacyMode.MAXIMUM: {
                "blur_faces": True,
                "blur_screens": True,
                "filter_content": True,
                "known_faces_exempt": False,
                "content_level": SafetyMode.FAMILY,
                "aggressive_filtering": True
            }
        }

        # Safety presets
        self.safety_presets: Dict[SafetyMode, dict] = {
            SafetyMode.UNRESTRICTED: {
                "filter_violence": False,
                "filter_nudity": False,
                "filter_medical": False,
                "filter_pii": False
            },
            SafetyMode.WORKPLACE: {
                "filter_violence": True,
                "filter_nudity": True,
                "filter_medical": False,
                "filter_pii": True
            },
            SafetyMode.MODERATE: {
                "filter_violence": True,
                "filter_nudity": True,
                "filter_medical": True,
                "filter_pii": True
            },
            SafetyMode.FAMILY: {
                "filter_violence": True,
                "filter_nudity": True,
                "filter_medical": True,
                "filter_pii": True,
                "strict_mode": True
            }
        }

        if config_file and config_file.exists():
            self.load_from_file(config_file)

    def load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Update detection config
            if 'detection' in data:
                for key, value in data['detection'].items():
                    if hasattr(self.detection, key):
                        setattr(self.detection, key, value)

            # Update performance config
            if 'performance' in data:
                for key, value in data['performance'].items():
                    if hasattr(self.performance, key):
                        setattr(self.performance, key, value)

            # Update security config
            if 'security' in data:
                for key, value in data['security'].items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)

        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    def save_to_file(self, config_file: Path) -> None:
        """Save current configuration to JSON file.

        Args:
            config_file: Path to save configuration
        """
        data = {
            'detection': {
                'face_scale_factor': self.detection.face_scale_factor,
                'face_min_neighbors': self.detection.face_min_neighbors,
                'face_min_size': list(self.detection.face_min_size),
                'face_detection_confidence': self.detection.face_detection_confidence,
                'screen_brightness_threshold': self.detection.screen_brightness_threshold,
                'screen_min_area': self.detection.screen_min_area,
                'screen_aspect_ratio_range': list(self.detection.screen_aspect_ratio_range),
                'text_confidence_threshold': self.detection.text_confidence_threshold,
                'inappropriate_content_threshold': self.detection.inappropriate_content_threshold,
                'skin_exposure_threshold': self.detection.skin_exposure_threshold,
                'frame_skip_interval': self.detection.frame_skip_interval,
                'detection_cache_duration': self.detection.detection_cache_duration,
                'downscale_factor': self.detection.downscale_factor,
                'blur_kernel_size': list(self.detection.blur_kernel_size),
                'pixelation_size': self.detection.pixelation_size
            },
            'performance': {
                'target_fps': self.performance.target_fps,
                'enable_gpu': self.performance.enable_gpu,
                'max_memory_mb': self.performance.max_memory_mb,
                'thread_count': self.performance.thread_count,
                'enable_caching': self.performance.enable_caching,
                'adaptive_quality': self.performance.adaptive_quality
            },
            'security': {
                'log_detections': self.security.log_detections,
                'encrypt_logs': self.security.encrypt_logs,
                'store_faces': self.security.store_faces,
                'auto_delete_cache': self.security.auto_delete_cache,
                'cache_ttl_seconds': self.security.cache_ttl_seconds
            }
        }

        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_privacy_settings(self, mode: PrivacyMode) -> dict:
        """Get privacy settings for a specific mode.

        Args:
            mode: Privacy mode to get settings for

        Returns:
            Dictionary of privacy settings
        """
        return self.privacy_presets.get(mode, self.privacy_presets[PrivacyMode.SMART])

    def get_safety_settings(self, mode: SafetyMode) -> dict:
        """Get safety settings for a specific mode.

        Args:
            mode: Safety mode to get settings for

        Returns:
            Dictionary of safety settings
        """
        return self.safety_presets.get(mode, self.safety_presets[SafetyMode.MODERATE])


# Global configuration instance
global_config = Config()


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        Global Config instance
    """
    return global_config


def load_config(config_file: Path) -> Config:
    """Load configuration from file and update global config.

    Args:
        config_file: Path to configuration file

    Returns:
        Updated global Config instance
    """
    global global_config
    global_config.load_from_file(config_file)
    return global_config