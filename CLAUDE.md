# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

RealityGuard is a computer vision project focused on real-time video privacy protection. The repository contains extensive testing of various approaches, with the SAM2 + Diffusion Hybrid emerging as the only genuinely viable groundbreaking solution after rigorous testing.

## Key Development Commands

### Running Core Systems
```bash
# Run the optimized real-time blur system (verified 58-255 FPS)
python optimized_realtime_blur.py

# Run comprehensive tests (includes all performance metrics)
python comprehensive_test.py

# Test current performance with realistic workloads
python test_current_performance.py

# Verify GPU utilization
python verify_gpu_utilization.py
```

### Testing Groundbreaking Approaches
```bash
# Test SAM2 + Diffusion Hybrid (the winning approach)
python test_groundbreaking_approaches.py

# Run thorough verification tests
python thorough_verification_test.py

# Cross-validate all results
python final_cross_validation.py
```

### Git Workflow
```bash
# Auto-push discoveries immediately
./auto_push.sh "Description of discovery"

# Manual commit with Co-Author attribution
git add -A
git commit -m "Your message

Co-Authored-By: Chinmay Shrivastava <cshrivastava2000@gmail.com>"
git push origin main
```

## Architecture & Key Components

### Verified Working Systems
1. **optimized_realtime_blur.py** - Production-ready system achieving 58-255 FPS
   - Uses YOLOv8n for detection
   - Implements frame skipping and batch processing
   - Three modes: High Quality (58 FPS), Optimized (178 FPS), Fast (255 FPS)

2. **SAM2 + Diffusion Hybrid** (Groundbreaking approach)
   - Combines SAM2 segmentation with diffusion model inpainting
   - Generates privacy-safe replacements instead of blur
   - Verified 25-97 FPS performance
   - Patent-worthy innovation

### Critical Performance Facts
- **GPU**: NVIDIA L4 with CUDA 12.8 required
- **Baseline YOLOv8n**: 95-103 FPS
- **GPU Utilization**: ~46% (CPU bottlenecked)
- **Memory Usage**: < 1.3 GB GPU memory

### Debunked Approaches (Do NOT pursue)
- **Gaussian Splatting**: Only 0.53 FPS mobile (claimed 100+)
- **NeRF**: Requires 15s training, can't handle video
- **Federated Vision**: Decreases accuracy by 48%

## Testing Philosophy

This codebase emphasizes **rigorous verification** over inflated claims. Always:
1. Test each approach at least 3 times
2. Use realistic workloads, not simplified simulations
3. Cross-validate results statistically
4. Document actual performance, not theoretical maximums

## Project Context

### Historical Journey
- Started with inflated claims of 244-326 FPS worth $75M
- Reality: 85 FPS on GPU, worth $1-3M
- Lesson: Most CV performance claims are inflated 3-10x

### Current Status
- **Winner**: SAM2 + Diffusion Hybrid approach
- **Performance**: Verified 25-97 FPS
- **Innovation**: First to combine these technologies for privacy
- **Implementation**: Ready for production in 4-6 weeks

## Important Files

### Documentation
- **FINAL_VALIDATED_TRUTH.md** - Absolute truth after 3 rounds of testing
- **GROUNDBREAKING_PROPOSAL.md** - Five innovative approaches analyzed
- **GROUNDBREAKING_TEST_RESULTS.md** - Comprehensive test results

### Test Results (JSON)
- **thorough_verification_results.json** - Most rigorous test results
- **groundbreaking_test_results.json** - Initial approach testing
- **comprehensive_test_results.json** - System performance metrics

## Development Guidelines

When working on this codebase:
1. **Always verify claims** - Test at least twice before documenting
2. **Use existing YOLOv8 models** - Don't try to train from scratch
3. **Focus on SAM2 + Diffusion** - Only genuinely groundbreaking approach
4. **Respect CPU bottleneck** - GPU utilization is limited to ~46%
5. **Document honestly** - No inflated metrics

## Dependencies

Core requirements (from requirements.txt):
- ultralytics>=8.3.189 (for YOLOv8)
- torch>=2.0.0 with CUDA support
- opencv-python-headless==4.10.0.84
- numpy, scipy, Pillow

## Repository Owner

- **Name**: Chinmay Shrivastava
- **Email**: cshrivastava2000@gmail.com
- **GitHub**: https://github.com/JonSnow1807/RealityGuard

Always attribute commits to the owner when pushing to GitHub.