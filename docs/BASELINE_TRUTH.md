# Baseline Performance - The Actual Truth

After thorough testing to eliminate all misleading metrics, here are the **real numbers**:

## Actual Baseline Performance (Verified)

### Simple Gaussian Blur (31x31 kernel)
- **HD (1280x720)**: 97-139 FPS
- **Full HD (1920x1080)**: 60 FPS
- **VGA (640x480)**: 188 FPS

### MediaPipe Face Detection
- **HD Performance**: 44 FPS
- Requires TensorFlow Lite
- Uses XNNPACK delegate

### Vision Transformer (Simulated)
- **HD Performance**: 32 FPS
- This is WITHOUT actual model weights
- Real ViT would be slower

## The False Claims vs Reality

### What Was Claimed (META_ACQUISITION_PACKAGE.md):
```
- Full HD (1920x1080): 244.7 FPS
- HD (1280x720): 262.4 FPS
- VGA (640x480): 263.2 FPS
```

### What's Actually Possible:
```
- Full HD: 60 FPS (simple blur only)
- HD: 97-139 FPS (simple blur only)
- With actual CV models: ~30-40 FPS
```

## Key Findings

### 1. Blur Detection "Issue" Explained
The 80% success rate in blur detection is **mathematically correct**:
- Linear gradients have no high-frequency content
- Gaussian blur removes high frequencies
- Therefore blur has minimal effect on smooth gradients
- This is EXPECTED behavior, not a bug

### 2. Kernel Size Impact
```
5x5 kernel:   1077 FPS
15x15 kernel:  283 FPS
31x31 kernel:  110 FPS
51x51 kernel:   66 FPS
```

### 3. GPU vs CPU
- CPU Blur: 97-139 FPS
- GPU Blur: 685 FPS (but requires CUDA setup)
- GPU gives 5-7x speedup when properly configured

### 4. Caching Effects
- Cold cache: First run slightly slower
- Warm cache: 1.14x speedup observed
- Not significant enough to inflate benchmarks dramatically

## The Truth About Performance Claims

### Why 244+ FPS is Impossible:
1. **Simple blur alone** only reaches 97-139 FPS on HD
2. **MediaPipe** runs at 44 FPS
3. **Vision Transformers** would run at ~30 FPS
4. **Combined pipeline** would be bottlenecked by slowest component

### What Actually Achieved 1700+ FPS:
The "Neural Approximation" which:
- Downsamples 8x (1280x720 → 160x90)
- Processes 1/64th of pixels
- Upsamples back
- **Trades quality for speed**

## Recommendations

### For Real CV System:
1. **Be honest about metrics**: 30-40 FPS for full pipeline
2. **Use GPU properly**: 5-7x speedup available
3. **Optimize bottlenecks**: Focus on slowest components
4. **Quality matters**: Don't just downsample for speed

### What Meta Actually Needs:
- Real-time performance (30+ FPS) ✓ Achievable
- Production quality ✓ Possible with GPU
- Novel algorithms ✗ Current approach not novel
- Honest benchmarks ✓ Now we have them

## Conclusion

The baseline performs exactly as expected for CPU-based OpenCV operations:
- ~100 FPS for HD blur
- ~40 FPS for MediaPipe
- ~30 FPS for transformer models

Claims of 244+ FPS for full CV pipeline were **misleading**. The only way to achieve such speeds is through aggressive downsampling (neural approximation) which sacrifices quality.

**Bottom Line**: Focus on building something genuinely novel that runs at realistic speeds (30-60 FPS) rather than chasing inflated metrics through quality compromises.