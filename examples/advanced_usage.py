#!/usr/bin/env python3
"""
RealityGuard Advanced Usage Examples
Demonstrates advanced features like batch processing, custom strategies, and optimization
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from realityguard_ai_replacement import (
    RealityGuardAIReplacement,
    AIReplacementConfig,
    ReplacementMode
)


class CustomReplacementStrategy:
    """Custom replacement strategy example"""

    def __init__(self):
        self.pattern_cache = {}

    def generate(self, roi: np.ndarray, context: dict) -> np.ndarray:
        """Generate custom replacement"""
        h, w = roi.shape[:2]

        # Create custom pattern (e.g., artistic mosaic)
        if (h, w) not in self.pattern_cache:
            # Generate mosaic pattern
            block_size = 8
            mosaic = np.zeros_like(roi)

            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    # Get average color of block
                    block = roi[y:y+block_size, x:x+block_size]
                    avg_color = block.mean(axis=(0, 1))

                    # Fill block with average color
                    mosaic[y:y+block_size, x:x+block_size] = avg_color

            self.pattern_cache[(h, w)] = mosaic

        return self.pattern_cache[(h, w)]


def parallel_video_processing():
    """Process multiple videos in parallel"""
    print("\n=== Parallel Video Processing ===")

    video_files = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]

    def process_single_video(video_path):
        """Process a single video"""
        output_path = f"protected_{Path(video_path).name}"

        config = AIReplacementConfig(
            default_mode=ReplacementMode.SYNTHETIC_FACE,
            batch_size=8,
            cache_size=200
        )

        system = RealityGuardAIReplacement(config)

        try:
            if Path(video_path).exists():
                stats = system.process_video(video_path, output_path, show_stats=False)
                return f"✓ {video_path}: {stats['avg_fps']:.1f} FPS"
            else:
                return f"✗ {video_path}: File not found"
        except Exception as e:
            return f"✗ {video_path}: Error - {e}"

    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_single_video, video_files))

    for result in results:
        print(f"  {result}")


def adaptive_quality_demo():
    """Demonstrate adaptive quality based on performance"""
    print("\n=== Adaptive Quality Demo ===")

    class AdaptiveSystem:
        def __init__(self):
            self.config = AIReplacementConfig(
                adaptive_quality=True,
                target_fps=30
            )
            self.system = RealityGuardAIReplacement(self.config)
            self.quality_levels = [0.3, 0.5, 0.7, 0.9]
            self.current_level = 2

        def process_with_adaptation(self, frame):
            """Process frame with quality adaptation"""
            # Process frame
            protected, stats = self.system.process_frame(frame)

            # Adapt quality based on FPS
            current_fps = stats['fps']

            if current_fps < 25 and self.current_level > 0:
                # Reduce quality
                self.current_level -= 1
                self.config.generation_quality = self.quality_levels[self.current_level]
                print(f"  ↓ Quality reduced to {self.config.generation_quality:.1f} (FPS: {current_fps:.1f})")

            elif current_fps > 40 and self.current_level < len(self.quality_levels) - 1:
                # Increase quality
                self.current_level += 1
                self.config.generation_quality = self.quality_levels[self.current_level]
                print(f"  ↑ Quality increased to {self.config.generation_quality:.1f} (FPS: {current_fps:.1f})")

            return protected, stats

    # Create test scenario
    adaptive_system = AdaptiveSystem()

    # Simulate varying load
    for i in range(10):
        # Create frame with varying complexity
        complexity = 480 + i * 60  # Increasing resolution
        frame = np.random.randint(0, 255, (complexity, complexity * 16//9, 3), dtype=np.uint8)

        print(f"\nFrame {i+1} ({complexity}p)")
        protected, stats = adaptive_system.process_with_adaptation(frame)
        print(f"  Result: {stats['fps']:.1f} FPS, Quality: {adaptive_system.config.generation_quality:.1f}")


def selective_privacy_zones():
    """Apply different privacy modes to different zones"""
    print("\n=== Selective Privacy Zones ===")

    class ZoneManager:
        def __init__(self):
            self.zones = []
            self.config = AIReplacementConfig()
            self.system = RealityGuardAIReplacement(self.config)

        def add_zone(self, x1, y1, x2, y2, mode: ReplacementMode):
            """Add a privacy zone"""
            self.zones.append({
                'bbox': (x1, y1, x2, y2),
                'mode': mode
            })

        def process_frame(self, frame):
            """Process frame with zone-specific privacy"""
            result = frame.copy()
            stats = {'zones_processed': 0}

            for zone in self.zones:
                x1, y1, x2, y2 = zone['bbox']
                roi = frame[y1:y2, x1:x2]

                # Set mode for this zone
                self.config.default_mode = zone['mode']

                # Process ROI
                protected_roi, zone_stats = self.system.process_frame(roi)

                # Replace in result
                result[y1:y2, x1:x2] = protected_roi
                stats['zones_processed'] += 1

            return result, stats

    # Create zone manager
    manager = ZoneManager()

    # Define privacy zones (for 1280x720 frame)
    manager.add_zone(0, 0, 640, 360, ReplacementMode.SYNTHETIC_FACE)  # Top-left: faces
    manager.add_zone(640, 0, 1280, 360, ReplacementMode.SMART_BLUR)  # Top-right: blur
    manager.add_zone(0, 360, 640, 720, ReplacementMode.CONTEXTUAL)  # Bottom-left: context
    manager.add_zone(640, 360, 1280, 720, ReplacementMode.ABSTRACT)  # Bottom-right: abstract

    # Process test frame
    test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    result, stats = manager.process_frame(test_frame)

    print(f"  Zones processed: {stats['zones_processed']}")
    print(f"  Output saved as: selective_privacy_demo.jpg")
    cv2.imwrite("selective_privacy_demo.jpg", result)


def cache_optimization_demo():
    """Demonstrate cache optimization benefits"""
    print("\n=== Cache Optimization Demo ===")

    # Test with and without cache
    configs = [
        ("No Cache", AIReplacementConfig(cache_size=0)),
        ("Small Cache", AIReplacementConfig(cache_size=50)),
        ("Large Cache", AIReplacementConfig(cache_size=500))
    ]

    # Create repeating test scenario
    test_frames = []
    for i in range(5):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * (50 + i * 40)
        test_frames.append(frame)

    # Repeat frames to test cache
    test_sequence = test_frames * 4  # 20 frames total

    for name, config in configs:
        print(f"\n{name}:")
        system = RealityGuardAIReplacement(config)

        start_time = time.time()
        fps_values = []

        for frame in test_sequence:
            _, stats = system.process_frame(frame)
            fps_values.append(stats['fps'])

        elapsed = time.time() - start_time

        # Get cache stats
        if hasattr(system.cache, 'get_stats'):
            cache_stats = system.cache.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0)
        else:
            hit_rate = 0

        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average FPS: {sum(fps_values)/len(fps_values):.1f}")
        print(f"  Cache hit rate: {hit_rate:.1%}")


def gpu_memory_management():
    """Demonstrate GPU memory management"""
    print("\n=== GPU Memory Management ===")

    if not torch.cuda.is_available():
        print("  GPU not available. Skipping...")
        return

    def get_gpu_memory():
        """Get current GPU memory usage"""
        return torch.cuda.memory_allocated() / 1024**3  # GB

    print(f"  Initial GPU memory: {get_gpu_memory():.2f} GB")

    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]

    for batch_size in batch_sizes:
        # Clear GPU cache
        torch.cuda.empty_cache()

        config = AIReplacementConfig(
            batch_size=batch_size,
            use_half_precision=True
        )

        system = RealityGuardAIReplacement(config)

        # Process test frame
        test_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        _, stats = system.process_frame(test_frame)

        memory_used = get_gpu_memory()
        print(f"\n  Batch size {batch_size}:")
        print(f"    GPU memory: {memory_used:.2f} GB")
        print(f"    FPS: {stats['fps']:.1f}")

    # Clear GPU cache
    torch.cuda.empty_cache()
    print(f"\n  Final GPU memory: {get_gpu_memory():.2f} GB")


def streaming_pipeline():
    """Example of streaming pipeline with buffering"""
    print("\n=== Streaming Pipeline ===")

    from queue import Queue
    import threading

    class StreamProcessor:
        def __init__(self, buffer_size=10):
            self.input_queue = Queue(maxsize=buffer_size)
            self.output_queue = Queue(maxsize=buffer_size)
            self.config = AIReplacementConfig(
                batch_size=4,
                cache_size=200
            )
            self.system = RealityGuardAIReplacement(self.config)
            self.running = False

        def process_worker(self):
            """Worker thread for processing"""
            while self.running:
                if not self.input_queue.empty():
                    frame = self.input_queue.get()
                    if frame is None:
                        break

                    protected, stats = self.system.process_frame(frame)
                    self.output_queue.put((protected, stats))

        def start(self):
            """Start processing pipeline"""
            self.running = True
            self.worker = threading.Thread(target=self.process_worker)
            self.worker.start()
            print("  Pipeline started")

        def stop(self):
            """Stop processing pipeline"""
            self.running = False
            self.input_queue.put(None)
            self.worker.join()
            print("  Pipeline stopped")

        def add_frame(self, frame):
            """Add frame to processing queue"""
            if not self.input_queue.full():
                self.input_queue.put(frame)
                return True
            return False

        def get_result(self):
            """Get processed frame"""
            if not self.output_queue.empty():
                return self.output_queue.get()
            return None, None

    # Demo the pipeline
    processor = StreamProcessor()
    processor.start()

    # Simulate streaming
    for i in range(20):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * (i * 10)
        if processor.add_frame(frame):
            print(f"  Added frame {i}")

        # Get results
        result = processor.get_result()
        if result[0] is not None:
            protected, stats = result
            print(f"    Processed: {stats['fps']:.1f} FPS")

        time.sleep(0.03)  # Simulate 30 FPS input

    processor.stop()


def profiling_example():
    """Profile system performance"""
    print("\n=== Performance Profiling ===")

    import cProfile
    import pstats
    from io import StringIO

    config = AIReplacementConfig(
        default_mode=ReplacementMode.SYNTHETIC_FACE,
        batch_size=8
    )

    system = RealityGuardAIReplacement(config)

    # Create profiler
    profiler = cProfile.Profile()

    # Profile frame processing
    test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128

    print("  Profiling 10 frames...")
    profiler.enable()

    for _ in range(10):
        system.process_frame(test_frame)

    profiler.disable()

    # Analyze results
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

    # Extract key metrics
    print("\n  Top time-consuming functions:")
    lines = stream.getvalue().split('\n')
    for line in lines[5:15]:  # Skip header, show top functions
        if line.strip():
            print(f"    {line[:100]}")


def main():
    """Run advanced examples"""
    print("=" * 60)
    print("RealityGuard Advanced Usage Examples")
    print("=" * 60)

    try:
        # Run examples
        parallel_video_processing()
        adaptive_quality_demo()
        selective_privacy_zones()
        cache_optimization_demo()
        gpu_memory_management()
        streaming_pipeline()
        profiling_example()

    except Exception as e:
        print(f"\nError in advanced examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()