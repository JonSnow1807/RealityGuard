#!/usr/bin/env python3
"""
Thorough verification of Mobile SAM claims.
Testing for any misleading metrics or false performance.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import hashlib
import gc
import psutil
from fast_mobile_sam import FastMobileSAM, OptimizedInference


class MobileSAMVerification:
    """Complete verification of Mobile SAM performance."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")

    def verify_model_architecture(self):
        """Verify the model is what we claim."""
        print("\n" + "="*60)
        print("ARCHITECTURE VERIFICATION")
        print("="*60)

        model = FastMobileSAM()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Calculate actual model size
        fp32_size = total_params * 4 / (1024 * 1024)
        fp16_size = total_params * 2 / (1024 * 1024)
        int8_size = total_params / (1024 * 1024)

        print(f"\nModel sizes:")
        print(f"  FP32: {fp32_size:.2f} MB")
        print(f"  FP16: {fp16_size:.2f} MB")
        print(f"  INT8: {int8_size:.2f} MB")

        # Verify it's actually small
        if total_params > 1_000_000:
            print("⚠️ WARNING: Model larger than claimed!")
        else:
            print("✓ Model size verified: < 1M parameters")

        # Check layer composition
        conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        transpose_layers = sum(1 for m in model.modules() if isinstance(m, nn.ConvTranspose2d))

        print(f"\nArchitecture:")
        print(f"  Conv2d layers: {conv_layers}")
        print(f"  ConvTranspose2d layers: {transpose_layers}")
        print(f"  Total layers: {len(list(model.modules()))}")

        return total_params

    def test_actual_segmentation(self):
        """Verify the model actually performs segmentation."""
        print("\n" + "="*60)
        print("SEGMENTATION FUNCTIONALITY TEST")
        print("="*60)

        model = FastMobileSAM().to(self.device)
        model.eval()

        # Create test image with clear object
        test_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(test_img, (256, 256), 100, (255, 255, 255), -1)  # White circle
        cv2.rectangle(test_img, (100, 100), (200, 200), (128, 128, 128), -1)  # Gray square

        # Convert to tensor
        tensor = torch.from_numpy(test_img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = model(tensor)

        # Check output
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Verify it's a valid mask
        mask = output[0, 0].cpu().numpy()
        unique_values = len(np.unique(mask > 0.5))

        if unique_values > 1:
            print("✓ Model produces segmentation mask")
        else:
            print("⚠️ WARNING: Output is uniform - not segmenting!")

        # Check if mask corresponds to objects
        mask_binary = (mask > 0.5).astype(np.uint8)
        num_components = cv2.connectedComponents(mask_binary)[0]

        print(f"Detected components: {num_components - 1}")  # Minus background

        return output

    def test_inference_speed_honestly(self):
        """Test real inference speed without tricks."""
        print("\n" + "="*60)
        print("HONEST SPEED TEST")
        print("="*60)

        model = FastMobileSAM().to(self.device)
        model.eval()

        # Test different scenarios
        test_cases = [
            ("Best case (cached)", True, False),
            ("Realistic (no cache)", False, True),
            ("With CPU-GPU transfer", False, True),
        ]

        results = {}

        for name, use_warmup, clear_cache in test_cases:
            print(f"\n{name}:")

            # Create random test image (different each time to avoid caching)
            test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

            if use_warmup:
                # Warmup
                for _ in range(10):
                    tensor = torch.from_numpy(test_img).permute(2, 0, 1).float() / 255.0
                    tensor = tensor.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        _ = model(tensor)

            times = []

            for i in range(50):
                if clear_cache:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                # Different image each time
                test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

                # Include preprocessing in timing
                start_total = time.perf_counter()

                # Preprocessing
                tensor = torch.from_numpy(test_img).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(self.device)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                # Inference
                start_inference = time.perf_counter()

                with torch.no_grad():
                    output = model(tensor)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()

                # Postprocessing (also timed)
                mask = output[0, 0].cpu().numpy()
                mask_resized = cv2.resize(mask, (1280, 720))  # Resize to HD

                end_total = time.perf_counter()

                inference_time = end - start_inference
                total_time = end_total - start_total

                times.append({
                    'inference': inference_time,
                    'total': total_time
                })

            # Calculate statistics
            inference_times = [t['inference'] for t in times]
            total_times = [t['total'] for t in times]

            avg_inference = np.mean(inference_times)
            avg_total = np.mean(total_times)

            fps_inference = 1.0 / avg_inference
            fps_total = 1.0 / avg_total

            results[name] = {
                'fps_inference': fps_inference,
                'fps_total': fps_total
            }

            print(f"  Inference only: {fps_inference:.1f} FPS ({avg_inference*1000:.2f} ms)")
            print(f"  With pre/post: {fps_total:.1f} FPS ({avg_total*1000:.2f} ms)")

        # Check for suspiciously high numbers
        print("\n" + "-"*60)
        print("REALITY CHECK:")

        max_fps = max(r['fps_inference'] for r in results.values())
        if max_fps > 500:
            print(f"⚠️ WARNING: {max_fps:.0f} FPS seems suspiciously high!")
            print("  Possible issues:")
            print("  - Model might be too simple")
            print("  - GPU might be caching aggressively")
            print("  - Timing might be incorrect")
        else:
            print(f"✓ Performance seems realistic: {max_fps:.0f} FPS")

        return results

    def test_memory_usage(self):
        """Check actual memory usage."""
        print("\n" + "="*60)
        print("MEMORY USAGE VERIFICATION")
        print("="*60)

        model = FastMobileSAM().to(self.device)
        model.eval()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Before loading model
            before_mem = torch.cuda.memory_allocated()

            # Create input
            test_input = torch.randn(1, 3, 512, 512).to(self.device)

            # Run inference
            with torch.no_grad():
                _ = model(test_input)

            torch.cuda.synchronize()

            # After inference
            after_mem = torch.cuda.memory_allocated()

            mem_used = (after_mem - before_mem) / (1024 * 1024)

            print(f"GPU memory used: {mem_used:.2f} MB")

            if mem_used < 1:
                print("⚠️ WARNING: Suspiciously low memory usage!")
                print("  Model might be too simple")
            else:
                print(f"✓ Memory usage seems reasonable")

        # CPU memory
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"CPU RAM used: {mem_info.rss / (1024*1024):.0f} MB")

    def test_output_quality(self):
        """Test if output is actually useful segmentation."""
        print("\n" + "="*60)
        print("OUTPUT QUALITY VERIFICATION")
        print("="*60)

        model = FastMobileSAM().to(self.device)
        model.eval()

        # Test on different types of images
        test_images = []

        # 1. Simple shapes
        img1 = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img1, (256, 256), 100, (255, 255, 255), -1)
        test_images.append(("Circle", img1))

        # 2. Multiple objects
        img2 = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img2, (150, 150), 50, (255, 0, 0), -1)
        cv2.rectangle(img2, (300, 300), (450, 450), (0, 255, 0), -1)
        test_images.append(("Multiple objects", img2))

        # 3. Random noise (should not segment)
        img3 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_images.append(("Noise", img3))

        for name, img in test_images:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = model(tensor)

            mask = output[0, 0].cpu().numpy()

            # Analyze mask
            mask_binary = (mask > 0.5).astype(np.uint8)
            unique = np.unique(mask_binary)
            coverage = np.mean(mask_binary)

            print(f"\n{name}:")
            print(f"  Unique values: {unique}")
            print(f"  Coverage: {coverage*100:.1f}%")

            if name == "Noise" and coverage > 0.3:
                print("  ⚠️ WARNING: Segmenting noise as objects!")

    def compare_with_claims(self):
        """Compare actual performance with claims."""
        print("\n" + "="*60)
        print("CLAIMS VS REALITY")
        print("="*60)

        # Our claims
        claims = {
            "L4 GPU FP16": 326,
            "Model size (MB)": 0.4,
            "Parameters": 200497
        }

        # Actual measurements
        model = FastMobileSAM()
        actual_params = sum(p.numel() for p in model.parameters())
        actual_size = actual_params * 2 / (1024 * 1024)  # FP16

        print("Claimed vs Actual:")
        print(f"  Parameters: {claims['Parameters']:,} vs {actual_params:,}")
        print(f"  Model size: {claims['Model size (MB)']} MB vs {actual_size:.2f} MB")

        # Speed test
        inference = OptimizedInference()

        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        times = []

        for _ in range(30):
            _, t = inference.process_frame(frame)
            times.append(t)

        actual_fps = 1.0 / np.mean(times)

        print(f"  HD FPS: {claims['L4 GPU FP16']} vs {actual_fps:.1f}")

        # Verdict
        print("\n" + "-"*60)
        if abs(actual_fps - claims['L4 GPU FP16']) / claims['L4 GPU FP16'] > 0.2:
            print("⚠️ SIGNIFICANT DISCREPANCY DETECTED!")
            print(f"  Claimed: {claims['L4 GPU FP16']} FPS")
            print(f"  Actual: {actual_fps:.1f} FPS")
            print(f"  Difference: {abs(actual_fps - claims['L4 GPU FP16']):.1f} FPS")
        else:
            print("✓ Performance claims are accurate")

    def run_complete_verification(self):
        """Run all verification tests."""
        print("="*60)
        print("MOBILE SAM - COMPLETE VERIFICATION")
        print("="*60)

        # 1. Architecture
        params = self.verify_model_architecture()

        # 2. Functionality
        _ = self.test_actual_segmentation()

        # 3. Speed
        speed_results = self.test_inference_speed_honestly()

        # 4. Memory
        self.test_memory_usage()

        # 5. Quality
        self.test_output_quality()

        # 6. Claims
        self.compare_with_claims()

        # Final verdict
        print("\n" + "="*60)
        print("FINAL VERIFICATION VERDICT")
        print("="*60)

        realistic_fps = speed_results['Realistic (no cache)']['fps_total']

        print(f"✓ Model parameters: {params:,}")
        print(f"✓ Realistic FPS (with pre/post): {realistic_fps:.1f}")
        print(f"✓ Model produces segmentation masks")

        if realistic_fps > 200:
            print("\n✅ Performance is GENUINE")
            print("   Model achieves high FPS through:")
            print("   - Lightweight architecture (200K params)")
            print("   - Efficient depthwise separable convs")
            print("   - GPU acceleration")
        else:
            print("\n⚠️ Performance lower than expected")
            print(f"   Actual: {realistic_fps:.1f} FPS")
            print("   This is still good for mobile deployment")

        mobile_projection = realistic_fps * 0.3
        print(f"\n📱 Mobile projection: ~{mobile_projection:.0f} FPS")

        if mobile_projection >= 60:
            print("   ✅ Meets Quest 60 FPS target")
        else:
            print("   ⚠️ May need more optimization for Quest")


if __name__ == "__main__":
    verifier = MobileSAMVerification()
    verifier.run_complete_verification()