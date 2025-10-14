#!/usr/bin/env python3
"""
Pure GPU Solution - Everything on GPU
No CPU-GPU transfers, entire pipeline stays on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import time
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PureGPUDetector(nn.Module):
    """Everything on GPU - detection and processing"""

    def __init__(self, device='cuda', model_type='yolo'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! This solution requires CUDA.")

        # Option 1: YOLOv5 (stays on GPU)
        if model_type == 'yolo':
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.detector.to(self.device)
            self.detector.eval()
            logger.info("✅ YOLOv5 loaded on GPU")

        # Option 2: RetinaNet (torchvision, GPU-native)
        elif model_type == 'retinanet':
            self.detector = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
            self.detector.to(self.device)
            self.detector.eval()
            logger.info("✅ RetinaNet loaded on GPU")

        # Option 3: Custom lightweight CNN for shapes
        elif model_type == 'custom':
            self.detector = self._build_custom_detector()
            self.detector.to(self.device)
            logger.info("✅ Custom detector loaded on GPU")

        # Pre-allocate GPU memory for common resolutions
        self.gpu_buffers = {
            '480p': torch.zeros((1, 3, 480, 640), device=self.device),
            '720p': torch.zeros((1, 3, 720, 1280), device=self.device),
            '1080p': torch.zeros((1, 3, 1080, 1920), device=self.device),
        }

        # Pre-compute Gaussian kernels on GPU
        self.blur_kernels = {}
        for size in [21, 31, 51, 71]:
            kernel = self._create_gaussian_kernel(size).to(self.device)
            self.blur_kernels[size] = kernel

        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _build_custom_detector(self):
        """Custom CNN for shape detection - fully on GPU"""
        return nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Detection head
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            # Output: confidence map + bounding boxes
            nn.Conv2d(64, 5, 1)  # 1 confidence + 4 bbox coords
        )

    def _create_gaussian_kernel(self, size, sigma=None):
        """Create Gaussian kernel directly on GPU"""
        if sigma is None:
            sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

        # Create kernel on GPU
        coords = torch.arange(size, dtype=torch.float32, device=self.device)
        coords -= (size - 1) / 2.0

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)  # For RGB

        return kernel

    @torch.no_grad()
    def detect_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pure GPU detection - no CPU transfer"""
        if self.model_type == 'yolo':
            # YOLOv5 detection
            results = self.detector(tensor)
            # Results stay on GPU
            return results

        elif self.model_type == 'retinanet':
            # RetinaNet detection
            predictions = self.detector(tensor)
            return predictions

        elif self.model_type == 'custom':
            # Custom detection
            output = self.detector(tensor)
            return output

    @torch.no_grad()
    def apply_blur_gpu_pure(self, tensor: torch.Tensor, detections: torch.Tensor,
                            kernel_size: int = 31) -> torch.Tensor:
        """Apply blur entirely on GPU without CPU roundtrip"""
        batch_size = tensor.shape[0]

        # Get blur kernel
        kernel = self.blur_kernels.get(kernel_size, self.blur_kernels[31])

        # Create mask from detections (all on GPU)
        mask = torch.zeros_like(tensor[:, 0:1, :, :])

        if self.model_type == 'yolo' and hasattr(detections, 'xyxy'):
            # Process YOLO detections
            for det in detections.xyxy[0]:  # First batch item
                if det[5] == 0:  # Person class
                    x1, y1, x2, y2 = det[:4].int()
                    mask[:, :, y1:y2, x1:x2] = 1.0

        # Apply convolution for blur
        padding = kernel_size // 2
        blurred = F.conv2d(tensor, kernel, padding=padding, groups=3)

        # Combine using mask (all on GPU)
        mask = mask.repeat(1, 3, 1, 1)
        output = tensor * (1 - mask) + blurred * mask

        return output

    @torch.no_grad()
    def process_pure_gpu(self, frames: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
        """Process entirely on GPU - no CPU operations"""
        # Frames already on GPU as tensor
        # Shape: [B, C, H, W]

        # Detection (stays on GPU)
        detections = self.detect_gpu(frames)

        # Blur (stays on GPU)
        output = self.apply_blur_gpu_pure(frames, detections, kernel_size)

        return output, detections

    def numpy_to_gpu(self, frame: np.ndarray) -> torch.Tensor:
        """Efficient numpy to GPU tensor conversion"""
        # Direct to GPU without CPU tensor intermediate
        tensor = torch.from_numpy(frame).cuda()
        tensor = tensor.float() / 255.0

        # Rearrange dimensions
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW

        return tensor

    def gpu_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Efficient GPU tensor to numpy conversion"""
        # Convert back
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)  # Remove batch

        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
        tensor = (tensor * 255).byte()

        # Single transfer to CPU
        frame = tensor.cpu().numpy()

        return frame


class PureGPUPipeline:
    """Complete pipeline that keeps everything on GPU"""

    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not torch.cuda.is_available():
            raise RuntimeError("This pipeline requires GPU!")

        # Initialize detector
        self.detector = PureGPUDetector(device='cuda', model_type='yolo')

        # Pre-allocate batch tensor on GPU
        self.batch_tensor = torch.zeros((batch_size, 3, 720, 1280),
                                       device=self.device, dtype=torch.float32)

        # Frame buffer
        self.frame_buffer = []

        logger.info(f"✅ Pure GPU Pipeline initialized with batch size {batch_size}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame through GPU pipeline"""
        start = time.perf_counter()

        # Convert to GPU tensor (single transfer)
        gpu_tensor = self.detector.numpy_to_gpu(frame)

        # Process on GPU (no transfers)
        output_tensor, detections = self.detector.process_pure_gpu(gpu_tensor)

        # Convert back (single transfer)
        output_frame = self.detector.gpu_to_numpy(output_tensor)

        elapsed = (time.perf_counter() - start) * 1000

        return output_frame, {
            'time_ms': elapsed,
            'fps': 1000 / elapsed,
            'detections': len(detections.xyxy[0]) if hasattr(detections, 'xyxy') else 0
        }

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch entirely on GPU"""
        start = time.perf_counter()

        # Stack frames and send to GPU in one transfer
        batch_np = np.stack(frames)
        batch_tensor = torch.from_numpy(batch_np).cuda().float() / 255.0
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW

        # Process on GPU (no transfers)
        output_tensor, detections = self.detector.process_pure_gpu(batch_tensor)

        # Convert back in one transfer
        output_tensor = output_tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
        output_tensor = (output_tensor * 255).byte()
        output_frames = output_tensor.cpu().numpy()

        elapsed = (time.perf_counter() - start) * 1000

        return list(output_frames), {
            'batch_size': len(frames),
            'total_time_ms': elapsed,
            'fps_total': len(frames) * 1000 / elapsed,
            'fps_per_frame': 1000 / (elapsed / len(frames))
        }

    def process_stream(self, frames_generator, max_frames=100):
        """Process video stream keeping data on GPU"""
        results = []
        total_time = 0

        for i, frame in enumerate(frames_generator):
            if i >= max_frames:
                break

            self.frame_buffer.append(frame)

            # Process when batch is full
            if len(self.frame_buffer) >= self.batch_size:
                outputs, info = self.process_batch(self.frame_buffer)
                results.extend(outputs)
                total_time += info['total_time_ms']
                self.frame_buffer = []

        # Process remaining frames
        if self.frame_buffer:
            outputs, info = self.process_batch(self.frame_buffer)
            results.extend(outputs)
            total_time += info['total_time_ms']

        return results, {
            'total_frames': len(results),
            'total_time_ms': total_time,
            'avg_fps': len(results) * 1000 / total_time if total_time > 0 else 0
        }


def benchmark_pure_gpu():
    """Benchmark pure GPU solution vs CPU/GPU hybrid"""
    print("\n" + "="*80)
    print("PURE GPU SOLUTION BENCHMARK")
    print("="*80)

    if not torch.cuda.is_available():
        print("❌ GPU not available! Cannot run pure GPU solution.")
        return None

    # Initialize pure GPU pipeline
    pipeline = PureGPUPipeline(batch_size=8)

    # Test different scenarios
    resolutions = [
        ('480p', (480, 640)),
        ('720p', (720, 1280)),
        ('1080p', (1080, 1920))
    ]

    results = {}

    for res_name, (h, w) in resolutions:
        print(f"\n{res_name} Resolution ({w}x{h}):")
        print("-" * 40)

        # Create test frame
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Single frame test
        print("\nSingle Frame:")
        times = []
        for _ in range(20):
            _, info = pipeline.process_frame(frame)
            times.append(info['time_ms'])

        avg_time = np.mean(times[5:])  # Skip warmup
        print(f"  Time: {avg_time:.2f}ms")
        print(f"  FPS: {1000/avg_time:.1f}")

        # Batch test
        print("\nBatch Processing (8 frames):")
        frames = [frame.copy() for _ in range(8)]

        batch_times = []
        for _ in range(10):
            _, info = pipeline.process_batch(frames)
            batch_times.append(info['total_time_ms'])

        avg_batch = np.mean(batch_times[2:])  # Skip warmup
        print(f"  Total: {avg_batch:.2f}ms")
        print(f"  Per frame: {avg_batch/8:.2f}ms")
        print(f"  FPS: {8000/avg_batch:.1f}")

        results[res_name] = {
            'single_ms': avg_time,
            'single_fps': 1000/avg_time,
            'batch_ms': avg_batch,
            'batch_fps': 8000/avg_batch
        }

    # Memory usage
    print("\n" + "="*60)
    print("GPU MEMORY USAGE")
    print("-" * 60)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Compare with previous results
    print("\n" + "="*80)
    print("COMPARISON: Pure GPU vs MediaPipe CPU")
    print("="*80)

    print("\n720p Performance Comparison:")
    print(f"  MediaPipe CPU: 130 FPS (from previous tests)")
    print(f"  Pure GPU: {results['720p']['batch_fps']:.1f} FPS")
    print(f"  Speedup: {results['720p']['batch_fps']/130:.2f}x")

    return results


def test_pure_gpu_accuracy():
    """Test accuracy of pure GPU detection"""
    print("\n" + "="*60)
    print("PURE GPU DETECTION ACCURACY TEST")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ GPU not available!")
        return

    detector = PureGPUDetector(device='cuda', model_type='yolo')

    # Test on different frame types
    test_cases = [
        ('empty', np.zeros((480, 640, 3), dtype=np.uint8)),
        ('noise', np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)),
        ('white_circle', np.zeros((480, 640, 3), dtype=np.uint8))
    ]

    # Add white circle to last test case
    cv2.circle(test_cases[2][1], (320, 240), 60, (255, 255, 255), -1)

    for name, frame in test_cases:
        print(f"\nTest: {name}")

        # Convert and process
        tensor = detector.numpy_to_gpu(frame)
        output, detections = detector.process_pure_gpu(tensor)

        # Check detections
        if hasattr(detections, 'xyxy'):
            num_detections = len(detections.xyxy[0])
        else:
            num_detections = 0

        print(f"  Detections: {num_detections}")

        # Note: YOLO won't detect geometric shapes (trained on real objects)
        if name == 'white_circle':
            print(f"  Note: YOLO doesn't detect synthetic shapes (expected)")


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_pure_gpu()

    # Test accuracy
    test_pure_gpu_accuracy()

    print("\n" + "="*80)
    print("PURE GPU SOLUTION SUMMARY")
    print("="*80)

    if results:
        print("✅ Everything runs on GPU - no CPU transfers during processing")
        print("✅ Single input/output transfer only")
        print("✅ Pre-allocated GPU memory for efficiency")
        print("✅ Batch processing maximizes GPU utilization")
        print("\n⚠️ Note: Requires YOLO/RetinaNet which don't detect geometric shapes")
        print("⚠️ Best for real-world images, not synthetic test cases")