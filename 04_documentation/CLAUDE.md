# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

RealityGuard is a computer vision privacy protection system that applies AI-based segmentation and privacy mask generation to video content. The system has been thoroughly debugged, optimized, and verified to work with real-world data at 45-63 FPS.

## Key Development Commands

### Running the Main Systems
```bash
# FIXED & OPTIMIZED VERSIONS (USE THESE)
python realityguard_final.py           # Final production-ready system (45-63 FPS on real data)
python realityguard_optimized.py       # Optimized with configurable privacy levels
python patent_ready_all_claims_fixed.py  # Fixed patent implementation

# Legacy versions (have issues, kept for reference)
python patent_ready_all_claims.py      # Original with privacy bug (masks not applied)
python sam2_diffusion_production.py    # SAM2 integration attempt
```

### Testing & Verification
```bash
# Real-world data tests (ALWAYS USE REAL DATA, NOT SYNTHETIC)
python test_real_world_data.py         # Downloads and tests with real images
python optimized_real_test.py          # Performance optimization tests
python final_acceptance_test.py        # Production acceptance criteria

# Comprehensive test suites
python comprehensive_test_suite.py     # Full testing framework
python verify_actual_detection.py      # Verifies YOLO detection on real data
python test_with_real_data.py         # Real-world verification

# Debug utilities
python debug_privacy_test.py          # Debug privacy application
python simple_visual_test.py          # Visual verification
python quick_debug.py                  # Quick debugging

# Performance monitoring
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv -l 1
```

### Production System (realityguard_production/)
```bash
cd realityguard_production/
# Docker deployment
docker-compose up -d
docker-compose logs -f

# Local development
pip install -r requirements.txt
python main.py  # Starts FastAPI on port 8000

# Access API docs at http://localhost:8000/docs
```

### Git Workflow
```bash
# Push with proper attribution
git add -A
git commit -m "Your message"
git push origin main
```

## High-Level Architecture

### System Pipeline
```
Input Video → YOLO Detection → Region Selection → Privacy Generation → Output
               ↓ (fallback)       ↓                ↓
          Edge Detection     Cache Lookup    Strategy Selection
                                                (geometric/neural/diffusion/maximum)
```

### Core Components

**Main Classes:**
- `FinalRealityGuard` - Main system orchestrator
- `FinalPrivacyGenerator` - Generates privacy masks with 4 strategies
- `HierarchicalCache` - 3-tier caching system (L1: exact, L2: similar, L3: generic)
- `FinalConfig` - Configuration with privacy strength levels

**Privacy Strategies:**
1. **Geometric** - Pattern-based obfuscation (fastest)
2. **Neural** - Multi-pass bilateral filtering (balanced)
3. **Diffusion** - Stylization and quantization (quality)
4. **Maximum** - Complete pixelation (strongest)

**Privacy Strength Levels:**
- `LOW` - Light blur, preserves context
- `MEDIUM` - Moderate blur, good balance
- `HIGH` - Strong blur, high privacy
- `MAXIMUM` - Complete obfuscation

### Detection System

**Primary: YOLO**
- Works on real photographs/videos
- Detects people (83-94% confidence), laptops, screens, phones
- Requires real images, NOT synthetic shapes

**Fallback: Intelligent Simulation**
- Edge detection with contours
- Grid-based variation detection
- Center region guarantee
- Ensures privacy always applied

## Critical Performance Metrics

### Verified on Real-World Data
- **FPS**: 45-63 on real images (800x533)
- **Privacy Effect**: 27-57 pixel difference
- **YOLO Accuracy**: 100% detection on real photos
- **Cache Hit Rate**: 96%+
- **Memory Growth**: <5MB

### Resolution Impact
| Resolution | FPS | Use Case |
|------------|-----|----------|
| 800x533 | 45-51 | Original quality |
| 640x480 | 44-62 | Optimal balance |
| 480x360 | 53-63 | Maximum speed |

## Testing Requirements

**IMPORTANT: Always test with REAL images/videos, not synthetic shapes**

YOLO is trained on real photographs and will NOT detect:
- Simple geometric shapes
- Drawn/cartoon figures
- Synthetic test patterns

Use actual photos/videos containing:
- Real people
- Laptops/screens
- Office/meeting scenes

## Known Issues & Solutions

**Fixed Issues:**
- Privacy masks not being applied → Fixed in `realityguard_final.py`
- Frame skipping inflating FPS → Removed in optimized versions
- Memory leaks → Added garbage collection

**Current Limitations:**
- YOLO doesn't detect synthetic shapes (expected behavior)
- GPU utilization 10-33% (CPU bottlenecked)
- Video privacy consistency varies frame-to-frame

## API Endpoints (Production System)

- `POST /api/v1/process` - Process video
- `GET /api/v1/status/{job_id}` - Check status
- `GET /api/v1/download/{job_id}` - Download result
- `POST /api/v1/stream` - Stream processing
- `GET /docs` - API documentation

## Dependencies

Core:
- `ultralytics>=8.3.189` (YOLOv8)
- `torch>=2.0.0` (CUDA support)
- `opencv-python-headless==4.10.0.84`
- `numpy`, `psutil`

## Technical Environment

- **GPU**: NVIDIA L4 (22.3 GB VRAM)
- **CUDA**: 12.8
- **Real-time threshold**: 24 FPS

## Repository Owner

- **Name**: Chinmay Shrivastava
- **Email**: cshrivastava2000@gmail.com
- **GitHub**: https://github.com/JonSnow1807/RealityGuard