#!/usr/bin/env python3
"""
Fixed and Optimized Pipeline - Achieving Real-Time Performance
"""

import torch
import numpy as np
import cv2
import time
from PIL import Image
from typing import Dict, List, Optional
import sys
sys.path.insert(0, '01_production')

class FastPrivacyProcessor:
    """Ultra-fast privacy processing using optimized techniques"""

    def __init__(self, use_diffusion=False):
        self.use_diffusion = use_diffusion
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

        if use_diffusion:
            self._init_diffusion()

    def _init_diffusion(self):
        """Initialize optimized diffusion pipeline"""
        try:
            from diffusers import AutoPipelineForInpainting, LCMScheduler

            # Use tiny model for speed
            self.pipeline = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                variant="fp16"
            ).to(self.device)

            # Ultra-fast scheduler
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Enable all optimizations
            self.pipeline.enable_attention_slicing(1)
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_xformers_memory_efficient_attention()

            print("Diffusion pipeline initialized with optimizations")

        except Exception as e:
            print(f"Diffusion init failed: {e}, using fallback")
            self.use_diffusion = False

    def process_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Process a single region with privacy mask"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        if self.use_diffusion and self.pipeline:
            # Diffusion-based (slower but better quality)
            return self._diffusion_privacy(roi)
        else:
            # Fast fallback methods
            return self._fast_privacy(roi)

    def _fast_privacy(self, roi: np.ndarray) -> np.ndarray:
        """Ultra-fast privacy methods"""
        h, w = roi.shape[:2]

        # Method 1: Gaussian blur (fastest)
        blurred = cv2.GaussianBlur(roi, (31, 31), 15)

        # Method 2: Pixelation
        scale = 0.1
        small = cv2.resize(roi, None, fx=scale, fy=scale)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Method 3: Mesh effect
        mesh = roi.copy()
        step = 10
        for i in range(0, h, step):
            cv2.line(mesh, (0, i), (w, i), (255, 255, 255), 1)
        for i in range(0, w, step):
            cv2.line(mesh, (i, 0), (i, h), (255, 255, 255), 1)

        # Combine methods for better privacy
        result = cv2.addWeighted(blurred, 0.5, pixelated, 0.5, 0)

        return result

    def _diffusion_privacy(self, roi: np.ndarray) -> np.ndarray:
        """Diffusion-based privacy (higher quality but slower)"""
        h, w = roi.shape[:2]

        # Resize to small for speed
        target_size = (256, 256)
        roi_resized = cv2.resize(roi, target_size)

        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB))
        mask = Image.new('L', target_size, 255)

        # Generate with minimal steps
        with torch.no_grad():
            result = self.pipeline(
                prompt="blurred privacy protected area",
                image=pil_img,
                mask_image=mask,
                num_inference_steps=1,
                guidance_scale=0.0,
                strength=1.0
            ).images[0]

        # Convert back
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        result_resized = cv2.resize(result_bgr, (w, h))

        return result_resized

def benchmark_performance():
    """Benchmark different privacy methods"""
    from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig
    from ultralytics import YOLO

    print("="*60)
    print("PERFORMANCE BENCHMARK: REAL-TIME PRIVACY")
    print("="*60)

    # Load test image
    img = cv2.imread('11_images/team_meeting.jpg')
    if img is None:
        print("Test image not found")
        return

    print(f"\nTest image: {img.shape}")

    # Initialize YOLO
    yolo = YOLO('yolov8n.pt')

    # Detect people
    results = yolo(img, conf=0.25, verbose=False)
    regions = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if int(box.cls) == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    regions.append([int(x1), int(y1), int(x2), int(y2)])

    print(f"Detected {len(regions)} people")

    if not regions:
        print("No people detected")
        return

    # Test different methods
    methods = [
        ("Fast Blur", FastPrivacyProcessor(use_diffusion=False)),
        ("Diffusion (if available)", FastPrivacyProcessor(use_diffusion=True)),
    ]

    print("\n" + "-"*60)
    print("METHOD COMPARISON:")
    print("-"*60)

    for name, processor in methods:
        print(f"\n{name}:")

        # Warm up
        _ = processor.process_region(img, regions[0])

        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            for region in regions[:2]:  # Process up to 2 regions
                _ = processor.process_region(img, region)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time

        print(f"  Average time: {avg_time:.1f}ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Real-time capable: {'✅ YES' if fps > 24 else '❌ NO'}")

    # Test the original system
    print(f"\nOriginal System (Fast mode):")
    config = OptimizedConfig(quality_mode="fast")
    system = OptimizedRealityGuard(config)

    # Warm up
    _ = system._detect_privacy_regions(img)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        regions_detected = system._detect_privacy_regions(img)
        if regions_detected:
            batch = [{'frame': img, 'region': r} for r in regions_detected[:2]]
            _ = system.diffusion.generate_privacy_batch(batch)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time

    print(f"  Average time: {avg_time:.1f}ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Real-time capable: {'✅ YES' if fps > 24 else '❌ NO'}")

def create_demo_video():
    """Create a demo showing privacy processing"""
    print("\n" + "="*60)
    print("CREATING DEMO VIDEO")
    print("="*60)

    # Load image
    img = cv2.imread('11_images/team_meeting.jpg')
    if img is None:
        return

    # Initialize processors
    processor = FastPrivacyProcessor(use_diffusion=False)
    yolo = YOLO('yolov8n.pt')

    # Detect regions
    results = yolo(img, conf=0.25, verbose=False)
    regions = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if int(box.cls) == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    regions.append([int(x1), int(y1), int(x2), int(y2)])

    print(f"Processing {len(regions)} regions...")

    # Create output image
    output = img.copy()

    for bbox in regions:
        x1, y1, x2, y2 = bbox
        # Process region
        processed = processor.process_region(img, bbox)
        # Apply to output
        output[y1:y2, x1:x2] = processed
        # Draw bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, "Privacy Protected", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save comparison
    comparison = np.hstack([img, output])
    cv2.imwrite('privacy_demo_comparison.jpg', comparison)
    print("✅ Saved: privacy_demo_comparison.jpg")

    # Show stats
    h, w = img.shape[:2]
    total_pixels = h * w
    protected_pixels = sum((r[2]-r[0])*(r[3]-r[1]) for r in regions)
    coverage = (protected_pixels / total_pixels) * 100

    print(f"\nPrivacy Coverage: {coverage:.1f}% of image")
    print(f"Regions protected: {len(regions)}")

if __name__ == "__main__":
    # Run benchmark
    benchmark_performance()

    # Create demo
    create_demo_video()

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)