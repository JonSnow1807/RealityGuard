# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

RealityGuard is a patented computer vision privacy protection system that combines segmentation AI with generative AI to CREATE privacy-safe content instead of destroying it. The system has evolved from initial research to production-ready implementation with all 6 patent claims validated at 48.7 FPS.

## Key Development Commands

### Running the Patent-Ready Systems
```bash
# All 6 patent claims validated (46.9 FPS)
python patent_ready_all_claims.py

# Production SAM2 + Diffusion system (36.9-77 FPS verified)
python sam2_diffusion_production.py
python advanced_sam2_diffusion.py  # Fast/Balanced/Quality modes

# Patent-ready optimized version (45.9 FPS)
python patent_ready_optimized.py

# Baseline blur system (60-295 FPS)
python optimized_realtime_blur.py
```

### Performance Testing & Verification
```bash
# Comprehensive test suite (all systems)
python comprehensive_test.py

# Patent validation (checks all 6 claims)
python patent_ready_all_claims.py  # Validates hierarchical cache, adaptive quality, etc.

# GPU utilization analysis
python verify_gpu_utilization.py

# Cross-validation tests
python final_cross_validation.py
python thorough_verification_test.py

# Real-time monitoring
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv -l 1
```

### Production System (realityguard_production/)
```bash
# Local development with Docker
cd realityguard_production/
docker-compose up -d  # Start all services (app, redis, postgres, monitoring)

# Without Docker
pip install -r requirements.txt
python main.py  # Runs FastAPI server on port 8000

# API Documentation
# Access at http://localhost:8000/docs after starting server

# Run production tests
cd realityguard_production/
pytest tests/  # Run all tests
pytest tests/test_privacy_engine.py  # Run specific test
pytest --cov=src tests/  # With coverage
```

### Creating Demos and Benchmarks
```bash
# Generate investor demo package
python investor_demo.py  # Creates video, charts, pitch deck

# Test with generated videos
python advanced_sam2_diffusion.py  # Benchmarks all modes

# Production-ready tests
python production_ready_system.py
```

### Git Workflow
```bash
# Quick push with auto-attribution
./auto_push.sh "Patent: All 6 claims validated at 47 FPS"

# Manual commit with attribution
git add -A
git commit -m "Your message

Co-Authored-By: Chinmay Shrivastava <cshrivastava2000@gmail.com>"
git push origin main
```

## High-Level Architecture

### Production System Structure
The repository contains two main implementations:

1. **Research/Testing Scripts** (root directory)
   - Individual Python files for testing and validation
   - Each file is standalone and demonstrates specific features
   - Used for benchmarking and patent validation

2. **Production System** (`realityguard_production/`)
   - Full FastAPI application with REST API
   - Docker containerized with all dependencies
   - Includes monitoring, caching, and database integration
   - Ready for cloud deployment

### The Patent-Ready Pipeline
```
Video → Segmentation → Hierarchical Cache → Adaptive Quality → Generation → Output
         (YOLO/SAM2)     (L1/L2/L3)         (Dynamic FPS)     (4 strategies)
```

### Core Patent Innovations (All 6 Validated)

1. **Real-time Processing** (46.9 FPS average)
   - Exceeds 24 FPS cinema standard
   - Stable performance across scenarios

2. **Hierarchical Caching System** (`HierarchicalCache` class)
   - L1: Exact match cache (fastest)
   - L2: Similar region cache (55.6% hit rate)
   - L3: Generic pattern cache (37.0% hit rate)
   - Overall 92.6% cache efficiency

3. **Adaptive Quality Control** (`AdaptiveQualityController` class)
   - Dynamic quality: 0.3 to 1.0
   - Strategy switching: geometric → neural → cached → diffusion
   - 96 adaptations per 150 frames typical

4. **Predictive Processing** (`PredictiveProcessor` class)
   - Motion vector calculation
   - IoU-based tracking
   - Pre-generation for predicted regions

5. **Multiple Privacy Strategies** (`OptimizedPrivacyGenerator` class)
   - Geometric synthesis (ultra-fast)
   - Neural blur (balanced)
   - Cached diffusion (efficient)
   - Full diffusion (quality)

6. **Segmentation + Generation**
   - YOLOv8n-seg for detection
   - Privacy mask generation (not just blur)

### System Components

**Research & Validation Files (root):**
- `patent_ready_all_claims.py` - All 6 patent claims validated (latest)
- `sam2_diffusion_production.py` - Production system prototype
- `advanced_sam2_diffusion.py` - Multi-mode system
- `optimized_realtime_blur.py` - Baseline comparison

**Production Implementation (`realityguard_production/`):**
- `main.py` - FastAPI application entry point
- `src/core/privacy_engine.py` - Core patent implementations
- `src/services/privacy_engine.py` - Service orchestration
- `src/api/routes.py` - REST API endpoints
- `src/core/config.py` - Configuration management

**Performance Profiles:**
```
Patent-Ready All Claims: 46.9 FPS (all innovations)
Production SAM2:         36.9 FPS (stable)
Advanced Fast Mode:      57.2 FPS (speed priority)
Advanced Balanced:       77.0 FPS (optimal)
Advanced Quality:        41.8 FPS (quality priority)
Blur Baseline:          294.9 FPS (simple blur only)
```

## Critical Technical Facts

- **GPU**: NVIDIA L4 (22.3 GB VRAM)
- **CUDA**: Version 12.8
- **PyTorch**: 2.7.1+cu128
- **GPU Utilization**: 10-33% (CPU bottlenecked)
- **Memory Usage**: < 1.3 GB GPU memory
- **Real-time threshold**: 24 FPS (cinema), 30 FPS (broadcast)

## Patent & Business Context

- **Patent Status**: Provisional patent filed September 27, 2025
- **Patent Documents**: See `PROVISIONAL_PATENT_APPLICATION.md`, `USPTO_*.txt` files
- **Performance**: 48.7 FPS validated (exceeds 24 FPS requirement by 2x)
- **Innovation**: World's first seg+gen privacy system
- **Valuation**: $10-50M with patent protection
- **Market**: $15B video privacy market

## Testing Philosophy

Always conduct rigorous verification:
1. Test multiple times for consistency
2. Use realistic video (1280x720 minimum)
3. Document actual vs theoretical performance
4. Test with static, moving, and multiple objects
5. Validate all patent claims explicitly

## API Endpoints (Production System)

When running the production system, these endpoints are available:

- `POST /api/v1/process` - Upload and process video
- `GET /api/v1/status/{job_id}` - Check processing status
- `GET /api/v1/download/{job_id}` - Download processed video
- `POST /api/v1/stream` - Start stream processing
- `GET /api/v1/stream/{stream_id}/frame` - Get latest frame
- `GET /api/v1/capabilities` - System capabilities
- `GET /docs` - Interactive API documentation

## Dependencies

Core requirements:
- `ultralytics>=8.3.189` - YOLOv8 models
- `torch>=2.0.0` - With CUDA support
- `opencv-python-headless==4.10.0.84`
- `numpy`, `scipy`, `Pillow`

Production system (`realityguard_production/`):
- `fastapi`, `uvicorn` - API framework
- `redis` - Caching layer
- `prometheus-client` - Metrics
- See `realityguard_production/requirements.txt` for full list

## Performance Bottlenecks & Solutions

**Current Bottleneck**: CPU-bound at 10-33% GPU utilization
- System still achieves target FPS despite bottleneck
- Future optimization: Multi-threading, batch processing

**Proven Optimizations**:
- Hierarchical caching: 92.6% hit rate
- Adaptive quality: Maintains target FPS
- Frame skipping: Process every N frames
- Resolution scaling: 0.3-1.0x adaptive

## Repository Owner

- **Name**: Chinmay Shrivastava
- **Email**: cshrivastava2000@gmail.com
- **GitHub**: https://github.com/JonSnow1807/RealityGuard

Always attribute commits to the owner when pushing to GitHub.