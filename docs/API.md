# RealityGuard API Reference

## Overview
RealityGuard provides a comprehensive API for real-time privacy protection using AI-generated synthetic content. The system achieves 30+ FPS performance while maintaining video utility.

## Quick Start

```python
from realityguard_ai_replacement import RealityGuardAIReplacement, AIReplacementConfig, ReplacementMode

# Initialize system
config = AIReplacementConfig(
    default_mode=ReplacementMode.SYNTHETIC_FACE,
    preserve_context=True
)
system = RealityGuardAIReplacement(config)

# Process video
system.process_video("input.mp4", "output.mp4")
```

## Core Classes

### RealityGuardAIReplacement

Main class for AI-powered privacy protection.

#### Constructor

```python
RealityGuardAIReplacement(config: AIReplacementConfig = None)
```

**Parameters:**
- `config` (AIReplacementConfig, optional): System configuration

#### Methods

##### process_frame
```python
process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]
```

Process a single frame with AI replacement.

**Parameters:**
- `frame`: Input frame (BGR format)

**Returns:**
- `protected_frame`: Frame with AI replacements
- `stats`: Processing statistics including FPS, replacements count

##### process_video
```python
process_video(input_path: str, output_path: str, show_stats: bool = True) -> dict
```

Process entire video file.

**Parameters:**
- `input_path`: Path to input video
- `output_path`: Path to output video
- `show_stats`: Display processing statistics

**Returns:**
- Dictionary with processing metrics

##### process_stream
```python
process_stream(stream_url: str, output_url: str = None) -> None
```

Process live video stream.

**Parameters:**
- `stream_url`: Input stream URL (RTSP, HTTP, etc.)
- `output_url`: Output stream URL (optional)

### AIReplacementConfig

Configuration for AI replacement system.

```python
@dataclass
class AIReplacementConfig:
    default_mode: ReplacementMode = ReplacementMode.SYNTHETIC_FACE
    detection_interval: int = 1  # Detect every N frames
    cache_size: int = 300
    batch_size: int = 8
    use_half_precision: bool = True  # FP16 for modern GPUs
    preserve_context: bool = True
    generation_quality: float = 0.7  # 0.0-1.0
    min_face_size: tuple = (20, 20)
    enable_tracking: bool = True
    adaptive_quality: bool = True
    target_fps: int = 30
```

### ReplacementMode

Available replacement modes:

```python
class ReplacementMode(Enum):
    SYNTHETIC_FACE = "synthetic_face"      # AI-generated faces
    CONTEXTUAL = "contextual"              # Context-aware replacements
    GENERIC_PERSON = "generic_person"      # Privacy-safe silhouettes
    SMART_BLUR = "smart_blur"             # AI-enhanced blur
    ABSTRACT = "abstract"                  # Artistic patterns
```

## GPU Optimized Version

For maximum performance, use the GPU-optimized base class:

```python
from realityguard_gpu_optimized import RealityGuardGPU, GPUConfig

config = GPUConfig(
    batch_size=16,
    use_tensorrt=True,  # If available
    precision="fp16"
)

system = RealityGuardGPU(config)
```

### GPUConfig

```python
@dataclass
class GPUConfig:
    device: str = "cuda:0"
    batch_size: int = 16
    use_tensorrt: bool = False
    precision: str = "fp16"  # fp16, fp32, int8
    memory_fraction: float = 0.8
    enable_cuda_graphs: bool = True
    num_workers: int = 4
```

## Advanced Features

### Custom Replacement Strategies

```python
from realityguard_ai_replacement import ReplacementStrategy

class CustomStrategy(ReplacementStrategy):
    def generate(self, roi: np.ndarray, context: dict) -> np.ndarray:
        # Your custom generation logic
        return synthetic_roi

# Register strategy
system.register_strategy("custom", CustomStrategy())
system.set_mode("custom")
```

### Selective Privacy

Apply different modes to different regions:

```python
# Define privacy zones
zones = {
    "faces": ReplacementMode.SYNTHETIC_FACE,
    "screens": ReplacementMode.SMART_BLUR,
    "background": ReplacementMode.CONTEXTUAL
}

system.set_zone_modes(zones)
```

### Performance Monitoring

```python
# Enable detailed metrics
system.enable_metrics(detailed=True)

# Process with callbacks
def on_frame_complete(frame_num, stats):
    print(f"Frame {frame_num}: {stats['fps']:.1f} FPS")

system.process_video("input.mp4", "output.mp4",
                     callback=on_frame_complete)

# Get performance report
report = system.get_performance_report()
```

## Cache Management

### Hierarchical Cache System

```python
from realityguard_ai_replacement import CacheConfig

cache_config = CacheConfig(
    l1_size=100,  # Exact match cache
    l2_size=200,  # Similar region cache
    l3_size=50,   # Universal pattern cache
    similarity_threshold=0.85
)

system = RealityGuardAIReplacement(
    config=AIReplacementConfig(cache_config=cache_config)
)
```

### Cache Statistics

```python
stats = system.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"L1 hits: {stats['l1_hits']}")
print(f"L2 hits: {stats['l2_hits']}")
print(f"L3 hits: {stats['l3_hits']}")
```

## Batch Processing

Process multiple videos efficiently:

```python
from realityguard_ai_replacement import BatchProcessor

processor = BatchProcessor(num_workers=4, gpu_per_worker=0.25)

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
output_dir = "processed/"

results = processor.process_batch(videos, output_dir)
```

## Integration Examples

### Flask Web Service

```python
from flask import Flask, request, send_file
from realityguard_ai_replacement import RealityGuardAIReplacement

app = Flask(__name__)
system = RealityGuardAIReplacement()

@app.route('/process', methods=['POST'])
def process_video():
    input_file = request.files['video']
    output_path = f"processed_{input_file.filename}"

    # Save and process
    input_file.save("temp.mp4")
    system.process_video("temp.mp4", output_path)

    return send_file(output_path, as_attachment=True)
```

### Real-time Webcam

```python
import cv2
from realityguard_ai_replacement import RealityGuardAIReplacement

system = RealityGuardAIReplacement()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    protected, stats = system.process_frame(frame)

    # Display with stats
    cv2.putText(protected, f"FPS: {stats['fps']:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Protected Stream', protected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### RTSP Stream Processing

```python
from realityguard_ai_replacement import StreamProcessor

processor = StreamProcessor()
processor.add_stream("rtsp://camera1.local:554/stream", "camera1_protected")
processor.add_stream("rtsp://camera2.local:554/stream", "camera2_protected")

processor.start()  # Non-blocking
```

## Error Handling

```python
from realityguard_ai_replacement import RealityGuardError

try:
    system = RealityGuardAIReplacement()
    system.process_video("input.mp4", "output.mp4")
except RealityGuardError as e:
    print(f"Processing error: {e}")
    if e.error_code == "GPU_MEMORY":
        # Reduce batch size and retry
        config = AIReplacementConfig(batch_size=4)
        system = RealityGuardAIReplacement(config)
        system.process_video("input.mp4", "output.mp4")
```

## Performance Tuning

### For Maximum Speed

```python
config = AIReplacementConfig(
    detection_interval=2,       # Detect every 2nd frame
    batch_size=16,             # Large batches
    use_half_precision=True,   # FP16
    cache_size=500,            # Large cache
    adaptive_quality=True,      # Auto-adjust quality
    target_fps=60              # Target high FPS
)
```

### For Maximum Quality

```python
config = AIReplacementConfig(
    detection_interval=1,       # Detect every frame
    generation_quality=1.0,     # Maximum quality
    preserve_context=True,      # Keep context
    use_half_precision=False,  # Full precision
    adaptive_quality=False      # Fixed quality
)
```

### For Low Memory

```python
config = AIReplacementConfig(
    batch_size=1,              # Single item batches
    cache_size=50,             # Small cache
    use_half_precision=True,   # Reduce memory
    detection_interval=3        # Less frequent detection
)
```

## Supported Formats

### Input Formats
- Video: MP4, AVI, MOV, MKV, WebM, RTSP, HTTP streams
- Image: JPG, PNG, BMP, TIFF

### Output Formats
- Video: MP4 (H.264), WebM (VP9), RTSP streams
- Image: JPG, PNG

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA 11.8+
- 4GB+ VRAM
- PyTorch 2.0+

## Thread Safety

The system is thread-safe for read operations. For concurrent processing:

```python
from concurrent.futures import ThreadPoolExecutor
from realityguard_ai_replacement import RealityGuardAIReplacement

def process_video(path):
    system = RealityGuardAIReplacement()  # Create per thread
    return system.process_video(path, f"output_{path}")

with ThreadPoolExecutor(max_workers=3) as executor:
    videos = ["v1.mp4", "v2.mp4", "v3.mp4"]
    results = list(executor.map(process_video, videos))
```

## Benchmarks

| Operation | Time (ms) | FPS |
|-----------|-----------|-----|
| Detection (YOLO) | 8-12 | 83-125 |
| Generation (GAN) | 15-25 | 40-67 |
| Full Pipeline | 25-35 | 29-40 |
| With Cache | 10-15 | 67-100 |

## Support

For issues or questions:
- GitHub Issues: https://github.com/JonSnow1807/RealityGuard/issues
- Email: cshrivastava2000@gmail.com