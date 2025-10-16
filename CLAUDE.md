# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

RealityGuard is an anti-AI privacy protection system implementing patented adversarial pattern technology. The system applies invisible-to-humans but AI-disrupting patterns to video/images, achieving 7-15px pixel differences that defeat facial recognition, deepfakes, and other AI analysis while maintaining visual quality.

## Core Working Implementation

The latest verified working version is `realityguard_ultimate.py` which:
- Applies adversarial patterns with 7.6px average difference (verified on real images)
- Detects 7-9 people per frame using YOLO
- Achieves 14.8 FPS on 800x532 video (below 24 FPS real-time target)
- 100% cache efficiency after warmup
- Successfully implements 5 of 6 patent claims

## Key Development Commands

### Running the Main System
```bash
# Current best implementation (functionally complete, performance needs optimization)
python realityguard_ultimate.py

# Test with real-world data
python test_ultimate_real_world.py
python final_real_test.py

# Video processing tests
python test_video_ultimate.py
python real_world_video_test.py
```

### Testing & Verification
```bash
# Verify all patent claims
python verify_all_claims.py

# Test pattern application effectiveness
python test_pattern_application.py

# Debug detection issues
python debug_detection.py

# Performance testing
python final_real_test.py  # Comprehensive metrics
```

### Critical Testing Notes
**ALWAYS use real images/videos for testing, NOT synthetic shapes**
- YOLO only detects real people in photographs
- Synthetic shapes will trigger fallback detection
- Test images available: real_test_1.jpg, real_test_2.jpg, real_test_3.jpg

## Architecture & Key Issues Resolved

### Pattern Application Fix (Critical)
The system wasn't applying patterns due to incorrect scaling. The fix:
```python
# WRONG (invisible):
roi += pattern  # pattern values: -0.08 to 0.08

# CORRECT (visible):
scaled_pattern = pattern * 255 * strength  # pattern values: -20 to 20
roi += scaled_pattern
```

### Current Configuration
```python
# realityguard_ultimate.py strength settings
min_strength: float = 0.20  # Increased from 0.03
max_strength: float = 0.60  # Increased from 0.12
default_strength: float = 0.35  # Increased from 0.06
```

### System Components

**Core Classes:**
- `RealityGuardUltimate` - Main orchestrator with pattern application
- `UltimateCache` - 3-tier hierarchical cache (L1: exact, L2: similar, L3: universal)
- `UltimateController` - Adaptive strength/strategy controller
- `PatternStrategy` - 5 adversarial pattern strategies

**Pattern Strategies:**
1. `geometric` - Moiré patterns (fastest)
2. `neural` - Edge-focused perturbations
3. `cached` - Pre-computed universal patterns
4. `diffusion` - Multi-scale noise
5. `temporal` - Anti-deepfake artifacts

## Performance Metrics (Verified)

### Real-World Test Results
- **Resolution**: 800x532
- **FPS**: 14.8 (target: ≥24)
- **Pixel Difference**: 7.6px average (target: 2-15px) ✓
- **People Detection**: 100% (7-9 per frame) ✓
- **Cache Efficiency**: 100% ✓
- **Consistency**: ±0.71px std deviation ✓

### Performance Issue
Current bottleneck: Processing 7-9 regions per frame
- YOLO detection: ~20ms
- Pattern generation: ~15ms × 7 regions = 105ms
- Total: ~135ms per frame = 7.4 FPS theoretical

## Patent Claims Status

| Claim | Description | Status | Evidence |
|-------|-------------|---------|----------|
| 1 | Real-time (>24 FPS) | ❌ | 14.8 FPS achieved |
| 2 | Hierarchical cache | ✓ | 100% efficiency |
| 3 | Adaptive control | ✓ | Strength/strategy adaptation |
| 4 | Predictive defense | ✓ | Face region prediction |
| 5 | Multiple strategies | ✓ | 5 strategies implemented |
| 6 | Segmentation | ✓ | YOLO + fallback detection |

## Known Issues & Solutions

### Resolved Issues
- **0.0px difference** → Fixed: Pattern scaling (*255)
- **Cache not working** → Fixed: Proper key generation
- **YOLO not detecting** → Expected: Needs real photos
- **Patterns invisible** → Fixed: Strength increased to 0.20-0.60

### Current Limitations
- **Performance**: 14.8 FPS (needs 24+ for real-time)
- **GPU Utilization**: Low (10-33%), CPU bottlenecked
- **Resolution Impact**: Higher resolution = lower FPS

## File Organization

### Core Implementation
- `realityguard_ultimate.py` - Latest working implementation
- `test_ultimate_real_world.py` - Real-world testing
- `verify_all_claims.py` - Patent claim verification

### Test Results
- `ACTUAL_TEST_RESULTS.md` - Latest verified metrics
- `SOLUTION_SUMMARY.md` - Complete fix documentation
- `FINAL_comparison_frame_*.jpg` - Visual proof of patterns

### Patent Documentation
- `05_patent/` - Patent application materials
- `PROVISIONAL_PATENT_APPLICATION.md` - 6 claims detailed

## Testing Requirements

### Image Requirements
Use real photographs containing:
- Actual people (not drawings)
- Real laptops/screens
- Office/meeting scenes

### Verification Commands
```bash
# Quick test with synthetic data (limited value)
python realityguard_ultimate.py

# Real-world image test (recommended)
python test_ultimate_real_world.py

# Video processing test
python test_video_ultimate.py

# Comprehensive verification
python final_real_test.py
```

## Git Workflow
```bash
# Commit without Claude attribution
git add -A
git commit -m "Your message"  # No emoji, no Claude mention
git push origin main
```

## Dependencies
Core requirements:
- `ultralytics>=8.3.189` (YOLO)
- `torch>=2.0.0` (CUDA support)
- `opencv-python`
- `numpy`

## Technical Environment
- GPU: NVIDIA L4 (22.3 GB VRAM)
- CUDA: 12.8
- Python: 3.8+
- Real-time threshold: 24 FPS

## Repository Owner
- **Name**: Chinmay Shrivastava
- **Email**: cshrivastava2000@gmail.com
- **GitHub**: https://github.com/JonSnow1807/RealityGuard