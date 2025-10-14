"""Unit tests for configuration system."""
import pytest
import json
import tempfile
from pathlib import Path
import sys
sys.path.append('../src')

from src.config import Config, PrivacyMode, SafetyMode, get_config


class TestConfig:
    """Test configuration management."""

    def test_default_config_initialization(self):
        """Test default configuration values."""
        config = Config()

        # Test detection defaults
        assert config.detection.face_scale_factor == 1.1
        assert config.detection.face_min_neighbors == 5
        assert config.detection.face_min_size == (30, 30)
        assert config.detection.face_detection_confidence == 0.5

        # Test performance defaults
        assert config.performance.target_fps == 120
        assert config.performance.enable_gpu == True
        assert config.performance.enable_caching == True

        # Test security defaults
        assert config.security.log_detections == False
        assert config.security.auto_delete_cache == True

    def test_load_from_file(self):
        """Test loading configuration from JSON file."""
        # Create temporary config file
        config_data = {
            "detection": {
                "face_scale_factor": 1.2,
                "face_min_neighbors": 3,
                "face_detection_confidence": 0.7
            },
            "performance": {
                "target_fps": 60,
                "enable_gpu": False
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            # Load config from file
            config = Config(temp_path)

            # Verify loaded values
            assert config.detection.face_scale_factor == 1.2
            assert config.detection.face_min_neighbors == 3
            assert config.detection.face_detection_confidence == 0.7
            assert config.performance.target_fps == 60
            assert config.performance.enable_gpu == False

            # Verify unchanged defaults
            assert config.detection.face_min_size == (30, 30)
            assert config.security.log_detections == False

        finally:
            temp_path.unlink()

    def test_save_to_file(self):
        """Test saving configuration to JSON file."""
        config = Config()

        # Modify some values
        config.detection.face_scale_factor = 1.5
        config.performance.target_fps = 90

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save config
            config.save_to_file(temp_path)

            # Load and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert data['detection']['face_scale_factor'] == 1.5
            assert data['performance']['target_fps'] == 90

        finally:
            temp_path.unlink()

    def test_privacy_mode_presets(self):
        """Test privacy mode preset settings."""
        config = Config()

        # Test OFF mode
        off_settings = config.get_privacy_settings(PrivacyMode.OFF)
        assert off_settings['blur_faces'] == False
        assert off_settings['blur_screens'] == False
        assert off_settings['filter_content'] == False

        # Test SMART mode
        smart_settings = config.get_privacy_settings(PrivacyMode.SMART)
        assert smart_settings['blur_faces'] == True
        assert smart_settings['blur_screens'] == False
        assert smart_settings['known_faces_exempt'] == True

        # Test MAXIMUM mode
        max_settings = config.get_privacy_settings(PrivacyMode.MAXIMUM)
        assert max_settings['blur_faces'] == True
        assert max_settings['blur_screens'] == True
        assert max_settings['filter_content'] == True
        assert max_settings['known_faces_exempt'] == False
        assert max_settings['aggressive_filtering'] == True

    def test_safety_mode_presets(self):
        """Test safety mode preset settings."""
        config = Config()

        # Test UNRESTRICTED mode
        unrestricted = config.get_safety_settings(SafetyMode.UNRESTRICTED)
        assert unrestricted['filter_violence'] == False
        assert unrestricted['filter_nudity'] == False
        assert unrestricted['filter_pii'] == False

        # Test FAMILY mode
        family = config.get_safety_settings(SafetyMode.FAMILY)
        assert family['filter_violence'] == True
        assert family['filter_nudity'] == True
        assert family['filter_medical'] == True
        assert family['filter_pii'] == True
        assert family['strict_mode'] == True

    def test_global_config_instance(self):
        """Test global configuration instance."""
        config1 = get_config()
        config2 = get_config()

        # Should be the same instance
        assert config1 is config2

        # Modifications should persist
        config1.detection.face_scale_factor = 2.0
        assert config2.detection.face_scale_factor == 2.0

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {")
            temp_path = Path(f.name)

        try:
            # Should handle error gracefully
            config = Config(temp_path)

            # Should use defaults
            assert config.detection.face_scale_factor == 1.1

        finally:
            temp_path.unlink()

    def test_partial_config_file(self):
        """Test loading partial configuration."""
        config_data = {
            "detection": {
                "face_scale_factor": 1.3
            }
            # Missing performance and security sections
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)

            # Specified value should be loaded
            assert config.detection.face_scale_factor == 1.3

            # Missing values should use defaults
            assert config.performance.target_fps == 120
            assert config.security.log_detections == False

        finally:
            temp_path.unlink()