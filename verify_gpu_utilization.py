#!/usr/bin/env python3
"""
Verify GPU utilization during inference
Monitor real-time GPU usage while running inference
"""

import torch
import numpy as np
from ultralytics import YOLO
import time
import subprocess
import threading
import queue

class GPUMonitor:
    """Monitor GPU utilization in real-time."""

    def __init__(self):
        self.gpu_stats = queue.Queue()
        self.monitoring = False

    def get_gpu_stats(self):
        """Get current GPU utilization."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                stats = result.stdout.strip().split(', ')
                return {
                    'gpu_util': float(stats[0]),
                    'mem_used': float(stats[1]),
                    'mem_total': float(stats[2]),
                    'temp': float(stats[3]),
                    'power': float(stats[4]) if stats[4] != '[N/A]' else 0
                }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        return None

    def monitor_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring:
            stats = self.get_gpu_stats()
            if stats:
                self.gpu_stats.put(stats)
            time.sleep(0.1)  # Sample every 100ms

    def start_monitoring(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.thread = threading.Thread(target=self.monitor_loop)
        self.thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        self.thread.join()

        # Collect all stats
        all_stats = []
        while not self.gpu_stats.empty():
            all_stats.append(self.gpu_stats.get())

        if all_stats:
            # Calculate statistics
            gpu_utils = [s['gpu_util'] for s in all_stats]
            mem_used = [s['mem_used'] for s in all_stats]
            temps = [s['temp'] for s in all_stats]
            powers = [s['power'] for s in all_stats if s['power'] > 0]

            return {
                'avg_gpu_util': np.mean(gpu_utils),
                'max_gpu_util': np.max(gpu_utils),
                'min_gpu_util': np.min(gpu_utils),
                'avg_mem_mb': np.mean(mem_used),
                'max_mem_mb': np.max(mem_used),
                'avg_temp': np.mean(temps),
                'avg_power': np.mean(powers) if powers else 0,
                'samples': len(all_stats)
            }
        return None


def test_gpu_utilization():
    """Test actual GPU utilization during inference."""

    print("="*80)
    print("GPU UTILIZATION VERIFICATION TEST")
    print("="*80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")

    # Load model
    print("\nLoading YOLOv8 model...")
    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    # Warmup
    print("Warming up model...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model(dummy, verbose=False, device='cuda')

    print("\n" + "="*80)
    print("MONITORING GPU UTILIZATION DURING INFERENCE")
    print("="*80)

    # Test different scenarios
    test_scenarios = [
        ("Single Image", 1, 100),
        ("Small Batch", 4, 50),
        ("Medium Batch", 16, 30),
        ("Large Batch", 32, 20),
    ]

    for scenario_name, batch_size, iterations in test_scenarios:
        print(f"\n{scenario_name} (Batch size: {batch_size})")
        print("-"*50)

        # Create batch
        batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(batch_size)]

        # Start monitoring
        monitor = GPUMonitor()
        monitor.start_monitoring()

        # Run inference
        print(f"Running {iterations} iterations...")
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(iterations):
            with torch.cuda.amp.autocast():
                _ = model(batch, verbose=False, device='cuda')

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # Stop monitoring and get stats
        time.sleep(0.5)  # Let last samples come in
        stats = monitor.stop_monitoring()

        # Calculate performance
        total_images = iterations * batch_size
        fps = total_images / elapsed

        # Display results
        print(f"\nPerformance:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Images processed: {total_images}")

        if stats:
            print(f"\nGPU Utilization:")
            print(f"  Average: {stats['avg_gpu_util']:.1f}%")
            print(f"  Maximum: {stats['max_gpu_util']:.1f}%")
            print(f"  Minimum: {stats['min_gpu_util']:.1f}%")

            print(f"\nMemory Usage:")
            print(f"  Average: {stats['avg_mem_mb']:.0f} MB")
            print(f"  Maximum: {stats['max_mem_mb']:.0f} MB")

            print(f"\nThermal:")
            print(f"  Temperature: {stats['avg_temp']:.1f}°C")
            if stats['avg_power'] > 0:
                print(f"  Power: {stats['avg_power']:.1f}W")

            # Analysis
            if stats['avg_gpu_util'] > 80:
                print("\n✅ HIGH GPU UTILIZATION - Efficiently using GPU")
            elif stats['avg_gpu_util'] > 50:
                print("\n✓ MODERATE GPU UTILIZATION - Good usage")
            else:
                print("\n⚠️ LOW GPU UTILIZATION - May be CPU bottlenecked")

    # Test sustained load
    print("\n" + "="*80)
    print("SUSTAINED LOAD TEST (30 seconds)")
    print("="*80)

    batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            for _ in range(16)]

    monitor = GPUMonitor()
    monitor.start_monitoring()

    print("Running sustained inference...")
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < 30:
        with torch.cuda.amp.autocast():
            _ = model(batch, verbose=False, device='cuda')
        iterations += 1

        # Print progress
        if iterations % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s: {iterations} iterations")

    total_time = time.time() - start_time
    stats = monitor.stop_monitoring()

    print(f"\nResults:")
    print(f"  Duration: {total_time:.1f}s")
    print(f"  Iterations: {iterations}")
    print(f"  Images: {iterations * 16}")
    print(f"  FPS: {iterations * 16 / total_time:.1f}")

    if stats:
        print(f"\nGPU Statistics over 30 seconds:")
        print(f"  Average utilization: {stats['avg_gpu_util']:.1f}%")
        print(f"  Peak utilization: {stats['max_gpu_util']:.1f}%")
        print(f"  Average memory: {stats['avg_mem_mb']:.0f} MB")
        print(f"  Peak memory: {stats['max_mem_mb']:.0f} MB")
        print(f"  Average temperature: {stats['avg_temp']:.1f}°C")

        if stats['avg_gpu_util'] > 70:
            print("\n✅ CONFIRMED: GPU IS BEING HEAVILY UTILIZED")
        else:
            print("\n⚠️ GPU utilization lower than expected")

    # Final verification
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    # Check if model is on GPU
    model_device = next(model.model.parameters()).device
    print(f"Model device: {model_device}")

    # Check CUDA memory
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"CUDA memory allocated: {allocated:.2f} GB")
    print(f"CUDA memory reserved: {reserved:.2f} GB")

    # Run a quick GPU operation to verify
    print("\nTesting raw GPU compute...")
    x = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gflops = (2 * 1000**3 * 1000) / (elapsed * 1e9)
    print(f"Raw compute: {gflops:.1f} GFLOPS")

    if model_device.type == 'cuda' and allocated > 0 and gflops > 100:
        print("\n✅ CONFIRMED: GPU IS BEING USED FOR COMPUTATION")
    else:
        print("\n❌ WARNING: GPU may not be properly utilized")


if __name__ == "__main__":
    test_gpu_utilization()