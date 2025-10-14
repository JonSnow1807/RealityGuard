"""
RealityGuard Realistic Performance Benchmark
Accurate measurement of actual processing capabilities
"""

import cv2
import numpy as np
import time
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.realityguard_final import RealityGuardFinal, PrivacyMode

class RealisticBenchmark:
    """Realistic benchmark with proper measurements"""
    
    def __init__(self):
        self.guard = RealityGuardFinal()
        # Generate realistic test data
        self.test_frames = self._generate_realistic_frames()
        
    def _generate_realistic_frames(self):
        """Generate realistic test frames at different resolutions"""
        frames = []
        resolutions = [
            (640, 480),   # VGA
            (1280, 720),  # 720p (Quest 3 passthrough)
            (1920, 1080), # 1080p
        ]
        
        print("Generating realistic test frames...")
        for res in resolutions:
            for i in range(10):
                # Create realistic scene
                frame = np.random.randint(50, 200, (*res[::-1], 3), dtype=np.uint8)
                
                # Add face-like regions (skin tones)
                for _ in range(3):
                    x, y = np.random.randint(100, res[0]-100), np.random.randint(100, res[1]-100)
                    cv2.ellipse(frame, (x, y), (60, 80), 0, 0, 360, 
                               (np.random.randint(150, 200), 
                                np.random.randint(130, 180), 
                                np.random.randint(100, 150)), -1)
                
                # Add screen-like bright regions
                for _ in range(2):
                    x1, y1 = np.random.randint(0, res[0]//2), np.random.randint(0, res[1]//2)
                    x2, y2 = x1 + np.random.randint(200, 400), y1 + np.random.randint(150, 300)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                 (np.random.randint(180, 220),
                                  np.random.randint(180, 220),
                                  np.random.randint(180, 220)), -1)
                
                frames.append(frame)
        
        return frames
    
    def benchmark_single_frame(self, frame, mode, iterations=100):
        """Benchmark single frame processing"""
        times = []
        
        # Warmup
        for _ in range(10):
            self.guard.process_frame(frame, mode)
        
        # Actual measurement
        for _ in range(iterations):
            start = time.perf_counter()
            output, stats = self.guard.process_frame(frame, mode)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times)
        }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark"""
        print("\n" + "="*70)
        print("REALISTIC PERFORMANCE BENCHMARK")
        print("="*70)
        
        modes = [PrivacyMode.OFF, PrivacyMode.SMART, PrivacyMode.SOCIAL, 
                PrivacyMode.WORKSPACE, PrivacyMode.MAXIMUM]
        
        resolutions = {
            'VGA (640x480)': (640, 480),
            '720p (1280x720)': (1280, 720),
            '1080p (1920x1080)': (1920, 1080)
        }
        
        results = {}
        
        for res_name, resolution in resolutions.items():
            print(f"\nTesting {res_name}:")
            print("-" * 40)
            
            # Get frames for this resolution
            test_frame = None
            for frame in self.test_frames:
                if frame.shape[:2] == resolution[::-1]:
                    test_frame = frame
                    break
            
            if test_frame is None:
                test_frame = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)
            
            for mode in modes:
                print(f"  {mode.name} mode...", end='')
                
                # Run benchmark
                stats = self.benchmark_single_frame(test_frame, mode, iterations=100)
                
                key = f"{res_name}_{mode.name}"
                results[key] = stats
                
                print(f" {stats['fps']:.1f} FPS ({stats['mean_ms']:.2f}ms)")
        
        return results
    
    def measure_component_performance(self):
        """Measure individual component performance"""
        print("\n" + "="*70)
        print("COMPONENT PERFORMANCE ANALYSIS")
        print("="*70)
        
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Measure face detection
        start = time.perf_counter()
        for _ in range(100):
            self.guard.detect_faces(frame)
        face_time = (time.perf_counter() - start) / 100 * 1000
        
        # Measure screen detection
        start = time.perf_counter()
        for _ in range(100):
            self.guard.detect_screens(frame)
        screen_time = (time.perf_counter() - start) / 100 * 1000
        
        # Measure blur application
        detections = self.guard.detect_faces(frame)
        start = time.perf_counter()
        for _ in range(100):
            self.guard.apply_blur(frame, detections, PrivacyMode.SMART)
        blur_time = (time.perf_counter() - start) / 100 * 1000
        
        print(f"\nComponent Timings (720p):")
        print(f"  Face Detection:   {face_time:.2f}ms ({1000/face_time:.1f} FPS)")
        print(f"  Screen Detection: {screen_time:.2f}ms ({1000/screen_time:.1f} FPS)")
        print(f"  Blur Application: {blur_time:.2f}ms ({1000/blur_time:.1f} FPS)")
        print(f"  Total Pipeline:   {face_time + screen_time + blur_time:.2f}ms")
        
        return {
            'face_detection_ms': face_time,
            'screen_detection_ms': screen_time,
            'blur_application_ms': blur_time,
            'total_ms': face_time + screen_time + blur_time
        }
    
    def test_quest3_scenario(self):
        """Test specific Quest 3 scenario"""
        print("\n" + "="*70)
        print("QUEST 3 SPECIFIC BENCHMARK (720p @ 120Hz)")
        print("="*70)
        
        # Quest 3 passthrough is 1280x720
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Add realistic content
        cv2.ellipse(frame, (400, 300), (70, 90), 0, 0, 360, (180, 160, 140), -1)
        cv2.rectangle(frame, (700, 200), (1100, 500), (200, 200, 200), -1)
        
        print("\nSimulating 1000 frames at Quest 3 resolution...")
        
        modes_to_test = [PrivacyMode.SMART, PrivacyMode.SOCIAL]
        
        for mode in modes_to_test:
            fps_history = []
            latencies = []
            
            # Process 1000 frames
            start_time = time.perf_counter()
            
            for i in range(1000):
                frame_start = time.perf_counter()
                output, stats = self.guard.process_frame(frame, mode)
                frame_time = time.perf_counter() - frame_start
                
                fps = 1 / frame_time
                fps_history.append(fps)
                latencies.append(frame_time * 1000)
                
                if (i + 1) % 200 == 0:
                    avg_fps = np.mean(fps_history)
                    print(f"  {mode.name}: Frame {i+1}/1000 - {avg_fps:.1f} FPS")
            
            total_time = time.perf_counter() - start_time
            
            # Results
            avg_fps = np.mean(fps_history)
            p99_latency = np.percentile(latencies, 99)
            
            print(f"\n{mode.name} Mode Results:")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  P99 Latency: {p99_latency:.2f}ms")
            print(f"  Can handle Quest 3 120Hz: {'✅ YES' if avg_fps >= 120 else '❌ NO'}")
            
            if avg_fps >= 120:
                headroom = (avg_fps / 120 - 1) * 100
                print(f"  Performance headroom: {headroom:.1f}%")
    
    def print_final_assessment(self, results):
        """Print final assessment"""
        print("\n" + "="*70)
        print("FINAL REALISTIC ASSESSMENT")
        print("="*70)
        
        # Check 720p performance (Quest 3 resolution)
        quest3_modes = ['720p (1280x720)_SMART', '720p (1280x720)_SOCIAL', 
                       '720p (1280x720)_WORKSPACE']
        
        quest3_fps = []
        for key in quest3_modes:
            if key in results:
                quest3_fps.append(results[key]['fps'])
        
        if quest3_fps:
            avg_quest3_fps = np.mean(quest3_fps)
            
            print(f"\nQuest 3 Resolution (720p) Performance:")
            print(f"  Average FPS across modes: {avg_quest3_fps:.1f}")
            
            if avg_quest3_fps >= 120:
                print(f"  ✅ ACHIEVES 120 FPS for Quest 3!")
                print(f"  Performance margin: {(avg_quest3_fps/120 - 1)*100:.1f}%")
            else:
                print(f"  Current: {avg_quest3_fps:.1f} FPS")
                print(f"  Need {120/avg_quest3_fps:.1f}x improvement")
        
        print("\n" + "="*70)
        print("This is the REAL performance of your system.")
        print("Previous inflated numbers were due to measurement error.")
        print("="*70)

def main():
    """Run realistic benchmark"""
    print("\nREALISTIC REALITYGUARD BENCHMARK")
    print("=" * 40)
    print("This benchmark shows ACTUAL performance")
    print("with proper frame processing.\n")
    
    benchmark = RealisticBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Component analysis
    benchmark.measure_component_performance()
    
    # Quest 3 specific test
    benchmark.test_quest3_scenario()
    
    # Final assessment
    benchmark.print_final_assessment(results)

if __name__ == "__main__":
    main()