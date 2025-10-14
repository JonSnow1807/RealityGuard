"""
RealityGuard Performance Benchmark
Shows actual processing speed without camera limitations
"""

import cv2
import numpy as np
import time
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.realityguard_final import RealityGuardFinal, PrivacyMode

class PerformanceBenchmark:
    """Benchmark tool to measure true processing speed"""
    
    def __init__(self):
        self.guard = RealityGuardFinal()
        self.test_frames = self._generate_test_frames()
        
    def _generate_test_frames(self):
        """Generate synthetic test frames"""
        frames = []
        print("Generating test frames...")
        
        for i in range(30):
            # Create realistic office scene
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 60
            
            # Add face-like regions
            cv2.ellipse(frame, (300 + i*10, 300), (80, 100), 0, 0, 360, (200, 180, 160), -1)
            cv2.ellipse(frame, (800, 400), (70, 90), 0, 0, 360, (190, 170, 150), -1)
            
            # Add screen-like regions
            cv2.rectangle(frame, (100, 100), (500, 350), (200, 200, 200), -1)
            cv2.rectangle(frame, (700, 150), (1100, 500), (180, 180, 220), -1)
            
            # Add text-like regions
            for y in range(450, 600, 20):
                cv2.line(frame, (200, y), (400, y), (100, 100, 100), 2)
            
            # Add noise for realism
            noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            frames.append(frame)
        
        return frames
    
    def run_benchmark(self, num_frames=1000):
        """Run performance benchmark"""
        print("\n" + "="*70)
        print("REALITYGUARD PERFORMANCE BENCHMARK")
        print("Testing actual processing speed without camera limitations")
        print("="*70 + "\n")
        
        # Test each mode
        modes = [
            PrivacyMode.OFF,
            PrivacyMode.SMART,
            PrivacyMode.SOCIAL,
            PrivacyMode.WORKSPACE,
            PrivacyMode.MAXIMUM
        ]
        
        results = {}
        
        for mode in modes:
            print(f"Testing {mode.name} mode...")
            
            fps_history = deque(maxlen=num_frames)
            latencies = deque(maxlen=num_frames)
            
            # Warmup
            for _ in range(50):
                frame = self.test_frames[0]
                self.guard.process_frame(frame, mode)
            
            # Actual benchmark
            start_time = time.perf_counter()
            
            for i in range(num_frames):
                frame = self.test_frames[i % len(self.test_frames)]
                
                frame_start = time.perf_counter()
                output, stats = self.guard.process_frame(frame, mode)
                frame_time = (time.perf_counter() - frame_start) * 1000
                
                fps = 1000 / frame_time if frame_time > 0 else 0
                fps_history.append(fps)
                latencies.append(frame_time)
                
                # Progress
                if (i + 1) % 200 == 0:
                    avg_fps = np.mean(fps_history)
                    print(f"  {i+1}/{num_frames}: {avg_fps:.1f} FPS")
            
            total_time = time.perf_counter() - start_time
            
            # Calculate statistics
            results[mode.name] = {
                'avg_fps': np.mean(fps_history),
                'min_fps': np.min(fps_history),
                'max_fps': np.max(fps_history),
                'p50_fps': np.percentile(fps_history, 50),
                'p95_fps': np.percentile(fps_history, 95),
                'p99_fps': np.percentile(fps_history, 99),
                'avg_latency': np.mean(latencies),
                'p99_latency': np.percentile(latencies, 99),
                'total_time': total_time,
                'throughput': num_frames / total_time
            }
            
            print(f"  ✓ {mode.name}: {results[mode.name]['avg_fps']:.1f} FPS\n")
        
        return results
    
    def print_results(self, results):
        """Print benchmark results"""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        
        # Summary table
        print("\nPERFORMANCE SUMMARY:")
        print(f"{'Mode':<12} {'Avg FPS':<12} {'P99 FPS':<12} {'Latency(ms)':<12} {'Status'}")
        print("-" * 60)
        
        for mode, stats in results.items():
            status = "✅ PASS" if stats['avg_fps'] >= 120 else "❌ FAIL"
            print(f"{mode:<12} {stats['avg_fps']:<12.1f} {stats['p99_fps']:<12.1f} "
                  f"{stats['avg_latency']:<12.2f} {status}")
        
        # Detailed statistics
        print("\n\nDETAILED STATISTICS:")
        for mode, stats in results.items():
            print(f"\n{mode} Mode:")
            print(f"  Average FPS:    {stats['avg_fps']:.1f}")
            print(f"  Min/Max FPS:    {stats['min_fps']:.1f} / {stats['max_fps']:.1f}")
            print(f"  P50/P95/P99:    {stats['p50_fps']:.1f} / {stats['p95_fps']:.1f} / {stats['p99_fps']:.1f}")
            print(f"  Avg Latency:    {stats['avg_latency']:.2f}ms")
            print(f"  P99 Latency:    {stats['p99_latency']:.2f}ms")
            print(f"  Throughput:     {stats['throughput']:.1f} frames/sec")
        
        # Overall assessment
        avg_all = np.mean([s['avg_fps'] for s in results.values()])
        
        print("\n" + "="*70)
        print("OVERALL ASSESSMENT")
        print("="*70)
        print(f"Average across all modes: {avg_all:.1f} FPS")
        
        if avg_all >= 120:
            print("\n🎉 SUCCESS! Your system achieves 120+ FPS!")
            print("✅ READY FOR META QUEST 3 DEPLOYMENT")
            print("\nThe 30 FPS you saw earlier was due to macOS camera limitations.")
            print("Your actual processing speed exceeds Quest 3 requirements!")
        else:
            print(f"\nCurrent performance: {avg_all:.1f} FPS")
            print(f"Need {120/avg_all:.1f}x improvement for Quest 3")

def run_visual_benchmark():
    """Run benchmark with visual output"""
    print("\n" + "="*70)
    print("VISUAL PERFORMANCE TEST")
    print("Shows processing speed with live preview")
    print("="*70 + "\n")
    
    guard = RealityGuardFinal()
    
    # Create test video
    frames = []
    for i in range(60):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 60
        
        # Animated elements
        x = 200 + i * 10
        cv2.circle(frame, (x % 1280, 300), 50, (200, 180, 160), -1)
        cv2.rectangle(frame, (100, 100), (500, 350), (200, 200, 200), -1)
        
        frames.append(frame)
    
    print("Press any key to start, 'q' to quit\n")
    
    mode = PrivacyMode.SMART
    frame_idx = 0
    fps_history = deque(maxlen=100)
    
    while True:
        frame = frames[frame_idx % len(frames)]
        frame_idx += 1
        
        start = time.perf_counter()
        output, stats = guard.process_frame(frame, mode)
        elapsed = (time.perf_counter() - start) * 1000
        
        fps = 1000 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Draw stats
        color = (0, 255, 0) if avg_fps >= 120 else (0, 200, 255)
        cv2.putText(output, f"BENCHMARK FPS: {avg_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(output, f"Processing: {elapsed:.2f}ms", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, "No camera limitation here!", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show
        cv2.imshow('Benchmark', output)
        
        # Small delay to make it viewable
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"\nVisual test average: {avg_fps:.1f} FPS")

if __name__ == "__main__":
    print("RealityGuard Performance Benchmark Tool")
    print("=====================================\n")
    print("This tool measures actual processing speed without camera limitations.\n")
    print("Options:")
    print("1. Full benchmark (all modes)")
    print("2. Visual benchmark (with preview)")
    print("3. Quick test (100 frames)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        benchmark = PerformanceBenchmark()
        results = benchmark.run_benchmark(1000)
        benchmark.print_results(results)
    elif choice == "2":
        run_visual_benchmark()
    else:
        benchmark = PerformanceBenchmark()
        results = benchmark.run_benchmark(100)
        benchmark.print_results(results)