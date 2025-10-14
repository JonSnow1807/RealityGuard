#!/usr/bin/env python3
"""
Test ONNX Runtime optimization as alternative to TensorRT
"""

import numpy as np
import time
import torch
from ultralytics import YOLO
import onnxruntime as ort


def test_onnx_runtime():
    """Test ONNX Runtime performance."""
    print("="*80)
    print("ONNX RUNTIME GPU OPTIMIZATION TEST")
    print("="*80)

    # Create baseline model
    base_model = YOLO('yolov8n-seg.pt')
    base_model.to('cuda')

    # Create ONNX Runtime session
    print("\nSetting up ONNX Runtime...")
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cuda_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]

    session = ort.InferenceSession('yolov8n-seg.onnx', providers=providers)
    print(f"Active providers: {session.get_providers()}")

    # Test batch
    batch_size = 8
    test_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                  for _ in range(batch_size)]

    print("\n1. BASELINE PYTORCH PERFORMANCE")
    print("-"*40)

    # Warmup
    for _ in range(10):
        _ = base_model(test_batch[0], verbose=False, device='cuda')

    # Test PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 50
    for _ in range(iterations):
        _ = base_model(test_batch[0], verbose=False, device='cuda')

    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start
    pytorch_fps = iterations / pytorch_time

    print(f"PyTorch FPS: {pytorch_fps:.1f}")

    print("\n2. ONNX RUNTIME PERFORMANCE")
    print("-"*40)

    # Prepare input for ONNX
    def preprocess_onnx(image):
        """Preprocess image for ONNX."""
        # Convert to float and normalize
        img = image.astype(np.float32) / 255.0
        # Change from HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    # Warmup ONNX
    input_name = session.get_inputs()[0].name
    for _ in range(10):
        test_input = preprocess_onnx(test_batch[0])
        _ = session.run(None, {input_name: test_input})

    # Test ONNX Runtime
    start = time.perf_counter()

    for _ in range(iterations):
        test_input = preprocess_onnx(test_batch[0])
        _ = session.run(None, {input_name: test_input})

    onnx_time = time.perf_counter() - start
    onnx_fps = iterations / onnx_time

    print(f"ONNX Runtime FPS: {onnx_fps:.1f}")

    # Batch processing test
    print("\n3. BATCH PROCESSING COMPARISON")
    print("-"*40)

    # PyTorch batch
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(10):
        _ = base_model(test_batch, verbose=False, device='cuda')

    torch.cuda.synchronize()
    pytorch_batch_time = time.perf_counter() - start
    pytorch_batch_fps = (10 * batch_size) / pytorch_batch_time

    print(f"PyTorch Batch FPS: {pytorch_batch_fps:.1f}")

    # ONNX batch
    batch_input = np.stack([preprocess_onnx(img)[0] for img in test_batch])

    start = time.perf_counter()

    for _ in range(10):
        _ = session.run(None, {input_name: batch_input})

    onnx_batch_time = time.perf_counter() - start
    onnx_batch_fps = (10 * batch_size) / onnx_batch_time

    print(f"ONNX Batch FPS: {onnx_batch_fps:.1f}")

    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)

    single_speedup = onnx_fps / pytorch_fps
    batch_speedup = onnx_batch_fps / pytorch_batch_fps

    print(f"Single image speedup: {single_speedup:.2f}x")
    print(f"Batch speedup: {batch_speedup:.2f}x")

    if single_speedup > 1.5:
        print("✅ Significant optimization achieved with ONNX Runtime!")
    elif single_speedup > 1.1:
        print("✓ Modest optimization with ONNX Runtime")
    else:
        print("⚠️ ONNX Runtime not faster than PyTorch")

    return {
        'pytorch_fps': pytorch_fps,
        'onnx_fps': onnx_fps,
        'pytorch_batch_fps': pytorch_batch_fps,
        'onnx_batch_fps': onnx_batch_fps,
        'single_speedup': single_speedup,
        'batch_speedup': batch_speedup
    }


if __name__ == "__main__":
    results = test_onnx_runtime()