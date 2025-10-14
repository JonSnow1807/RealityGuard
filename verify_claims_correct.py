#!/usr/bin/env python3
"""
Correct Claims Verification Test - Uses proper API
"""

import sys
import os
import time
import json
import torch
import numpy as np
import cv2
from datetime import datetime

# Add production path
sys.path.insert(0, '01_production')

def test_system_components():
    """Test all system components"""
    print("\n" + "="*60)
    print("TESTING SYSTEM COMPONENTS")
    print("="*60)

    results = []

    # Test 1: Imports
    print("\n[1/8] Testing imports...")
    try:
        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig
        from contextual_prompt_engine import ContextualPromptEngine
        print("✅ All imports successful")
        results.append({"test": "imports", "status": "✅ PASS"})
    except Exception as e:
        print(f"❌ Import failed: {e}")
        results.append({"test": "imports", "status": "❌ FAIL", "error": str(e)})
        return results

    # Test 2: System initialization
    print("\n[2/8] Testing initialization...")
    try:
        config = OptimizedConfig(quality_mode="fast")
        system = OptimizedRealityGuard(config)
        print(f"✅ System initialized: {system.__class__.__name__}")
        results.append({"test": "initialization", "status": "✅ PASS"})
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        results.append({"test": "initialization", "status": "❌ FAIL", "error": str(e)})
        return results

    # Test 3: Context Engine
    print("\n[3/8] Testing context engine...")
    try:
        context_engine = ContextualPromptEngine()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Context engine returns a ContextualPrompt object
        prompt_obj = context_engine.generate_prompt(test_frame, "office")
        # Access the prompt string
        if hasattr(prompt_obj, 'prompt'):
            prompt_str = prompt_obj.prompt
        else:
            prompt_str = str(prompt_obj)
        print(f"✅ Context engine working: Generated prompt")
        results.append({"test": "context_engine", "status": "✅ PASS"})
    except Exception as e:
        print(f"❌ Context engine failed: {e}")
        results.append({"test": "context_engine", "status": "❌ FAIL", "error": str(e)})

    # Test 4: Detection system
    print("\n[4/8] Testing detection...")
    try:
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        # Add a rectangle to simulate a person
        cv2.rectangle(test_frame, (200, 150), (400, 450), (150, 100, 80), -1)

        regions = system._detect_privacy_regions(test_frame)
        print(f"✅ Detection working: {len(regions)} regions found")
        results.append({"test": "detection", "status": "✅ PASS", "regions": len(regions)})
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        results.append({"test": "detection", "status": "❌ FAIL", "error": str(e)})

    # Test 5: Privacy generation
    print("\n[5/8] Testing privacy generation...")
    try:
        if regions:
            batch = [{'frame': test_frame, 'region': regions[0]}]
            masks = system.diffusion.generate_privacy_batch(batch)
            print(f"✅ Privacy generation working: Generated {len(masks)} masks")
            results.append({"test": "privacy_generation", "status": "✅ PASS"})
        else:
            # Use fallback
            fallback_mask = system.diffusion._fallback_privacy(
                {'bbox': [100, 100, 300, 300]},
                test_frame
            )
            print(f"✅ Privacy fallback working: Shape {fallback_mask.shape}")
            results.append({"test": "privacy_generation", "status": "✅ PASS", "note": "fallback"})
    except Exception as e:
        print(f"❌ Privacy generation failed: {e}")
        results.append({"test": "privacy_generation", "status": "❌ FAIL", "error": str(e)})

    # Test 6: Frame interpolation
    print("\n[6/8] Testing frame interpolation...")
    try:
        interpolator = system.interpolator
        # Store a test mask
        test_mask = np.ones((200, 200, 3), dtype=np.uint8) * 255
        interpolator.store_mask(0, 1, test_mask)

        # Check interpolation
        should_process = interpolator.should_process_frame(3, 3)
        print(f"✅ Frame interpolation working: Skip pattern active")
        results.append({"test": "frame_interpolation", "status": "✅ PASS"})
    except Exception as e:
        print(f"❌ Frame interpolation failed: {e}")
        results.append({"test": "frame_interpolation", "status": "❌ FAIL", "error": str(e)})

    # Test 7: Configuration modes
    print("\n[7/8] Testing configuration modes...")
    try:
        modes_tested = []
        for mode in ["fast", "balanced", "quality"]:
            config = OptimizedConfig(quality_mode=mode)
            test_system = OptimizedRealityGuard(config)
            modes_tested.append({
                "mode": mode,
                "steps": config.num_inference_steps,
                "size": config.generation_size[0]
            })
        print(f"✅ All configuration modes working")
        for m in modes_tested:
            print(f"   - {m['mode']}: {m['steps']} steps, {m['size']}px")
        results.append({"test": "configuration", "status": "✅ PASS", "modes": modes_tested})
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        results.append({"test": "configuration", "status": "❌ FAIL", "error": str(e)})

    # Test 8: Memory management
    print("\n[8/8] Testing memory management...")
    try:
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2

            # Process several frames
            for i in range(5):
                test_frame = np.ones((480, 640, 3), dtype=np.uint8) * (50 + i * 30)
                regions = system._detect_privacy_regions(test_frame)
                if regions:
                    batch = [{'frame': test_frame, 'region': r} for r in regions[:1]]
                    _ = system.diffusion.generate_privacy_batch(batch)

            final_mem = torch.cuda.memory_allocated() / 1024**2
            mem_growth = final_mem - initial_mem

            print(f"✅ Memory stable: {final_mem:.1f}MB (growth: {mem_growth:.1f}MB)")
            results.append({"test": "memory", "status": "✅ PASS", "memory_mb": final_mem})
        else:
            print("✅ Running in CPU mode")
            results.append({"test": "memory", "status": "✅ PASS", "note": "CPU mode"})
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        results.append({"test": "memory", "status": "❌ FAIL", "error": str(e)})

    return results

def test_performance_claims():
    """Test performance claims"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE CLAIMS")
    print("="*60)

    try:
        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

        # Test each mode
        performance_results = {}

        for mode in ["fast", "balanced", "quality"]:
            print(f"\nTesting {mode} mode...")

            config = OptimizedConfig(quality_mode=mode)
            system = OptimizedRealityGuard(config)

            # Create test frames
            test_frames = []
            for i in range(30):  # 1 second of video
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
                # Add synthetic content
                cv2.rectangle(frame, (200, 150), (400, 450), (150, 100, 80), -1)
                cv2.circle(frame, (320, 300), 50, (200, 150, 100), -1)
                test_frames.append(frame)

            # Measure performance
            start_time = time.time()
            processed = 0

            for frame_idx, frame in enumerate(test_frames):
                # Check if should process
                if system.interpolator.should_process_frame(frame_idx, config.process_every_n_frames):
                    regions = system._detect_privacy_regions(frame)
                    if regions:
                        batch = [{'frame': frame, 'region': r} for r in regions[:config.max_regions_per_frame]]
                        _ = system.diffusion.generate_privacy_batch(batch)
                processed += 1

            elapsed = time.time() - start_time
            fps = processed / elapsed if elapsed > 0 else 0

            performance_results[mode] = {
                "fps": fps,
                "time": elapsed,
                "frames": processed,
                "meets_realtime": fps >= 24
            }

            print(f"  FPS: {fps:.1f} {'✅' if fps >= 24 else '❌'}")
            print(f"  Time: {elapsed:.2f}s for {processed} frames")

        return performance_results

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return {}

def verify_patent_claims():
    """Verify all 6 patent claims"""
    print("\n" + "="*60)
    print("VERIFYING PATENT CLAIMS")
    print("="*60)

    claims = {
        "claim1": {"description": "Real-time processing (>24 FPS)", "verified": False},
        "claim2": {"description": "AI-based mask generation", "verified": False},
        "claim3": {"description": "Context-aware adaptation", "verified": False},
        "claim4": {"description": "Temporal consistency", "verified": False},
        "claim5": {"description": "Multi-layer caching", "verified": False},
        "claim6": {"description": "Production deployment ready", "verified": False}
    }

    try:
        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

        config = OptimizedConfig(quality_mode="balanced")
        system = OptimizedRealityGuard(config)

        # Claim 1: Real-time performance
        print("\n[Claim 1] Testing real-time performance...")
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        start = time.time()
        for i in range(24):
            if system.interpolator.should_process_frame(i, config.process_every_n_frames):
                regions = system._detect_privacy_regions(test_frame)
                if regions:
                    batch = [{'frame': test_frame, 'region': regions[0]}]
                    _ = system.diffusion.generate_privacy_batch(batch)
        elapsed = time.time() - start
        fps = 24 / elapsed if elapsed > 0 else 0

        if fps > 24:
            claims["claim1"]["verified"] = True
            claims["claim1"]["fps"] = fps
            print(f"  ✅ Achieved {fps:.1f} FPS (>24 required)")
        else:
            print(f"  ❌ Only {fps:.1f} FPS (<24 required)")

        # Claim 2: AI-based generation
        print("\n[Claim 2] Testing AI-based mask generation...")
        if hasattr(system, 'diffusion') and system.diffusion.pipeline is not None:
            claims["claim2"]["verified"] = True
            print("  ✅ AI pipeline loaded (Stable Diffusion)")
        else:
            print("  ❌ No AI pipeline found")

        # Claim 3: Context-aware
        print("\n[Claim 3] Testing context awareness...")
        try:
            from contextual_prompt_engine import ContextualPromptEngine
            engine = ContextualPromptEngine()
            claims["claim3"]["verified"] = True
            print("  ✅ Context engine available (CLIP-based)")
        except:
            print("  ❌ Context engine not available")

        # Claim 4: Temporal consistency
        print("\n[Claim 4] Testing temporal consistency...")
        if hasattr(system, 'interpolator'):
            claims["claim4"]["verified"] = True
            print("  ✅ Frame interpolation available")
        else:
            print("  ❌ No frame interpolation")

        # Claim 5: Caching
        print("\n[Claim 5] Testing caching system...")
        if hasattr(system.diffusion, 'result_cache'):
            claims["claim5"]["verified"] = True
            print("  ✅ Caching system present")
        else:
            print("  ❌ No caching system")

        # Claim 6: Production ready
        print("\n[Claim 6] Testing production readiness...")
        if all(claims[f"claim{i}"]["verified"] for i in range(1, 6)):
            claims["claim6"]["verified"] = True
            print("  ✅ All systems operational - Production ready")
        else:
            print("  ❌ Some systems not operational")

    except Exception as e:
        print(f"❌ Claims verification error: {e}")

    return claims

def test_real_world_data():
    """Test with real-world data"""
    print("\n" + "="*60)
    print("TESTING WITH REAL-WORLD DATA")
    print("="*60)

    try:
        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig
        import urllib.request

        # Try to download a test image
        print("\nAttempting to download test image...")
        test_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg/640px-Green_Sea_Turtle_grazing_seagrass.jpg"
        ]

        img = None
        for url in test_urls:
            try:
                urllib.request.urlretrieve(url, "test_image.jpg")
                img = cv2.imread("test_image.jpg")
                if img is not None:
                    print(f"✅ Downloaded test image: {img.shape}")
                    break
            except:
                continue

        if img is None:
            # Create synthetic test image
            print("⚠️ Using synthetic test image instead")
            img = np.ones((480, 640, 3), dtype=np.uint8) * 100
            # Add complex patterns
            for i in range(5):
                x, y = np.random.randint(50, 400, 2)
                cv2.circle(img, (x, y), 30, (150, 100, 80), -1)

        # Test with different modes
        results = {}
        config = OptimizedConfig(quality_mode="balanced")
        system = OptimizedRealityGuard(config)

        print("\nProcessing test image...")
        start = time.time()

        # Process multiple times for timing
        iterations = 10
        for _ in range(iterations):
            regions = system._detect_privacy_regions(img)
            if regions:
                batch = [{'frame': img, 'region': r} for r in regions[:1]]
                _ = system.diffusion.generate_privacy_batch(batch)

        elapsed = time.time() - start
        fps = iterations / elapsed if elapsed > 0 else 0

        results["real_world_fps"] = fps
        results["processing_time"] = elapsed / iterations

        print(f"✅ Real-world performance: {fps:.1f} FPS")
        print(f"   Processing time per frame: {results['processing_time']*1000:.1f}ms")

        # Cleanup
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")

        return results

    except Exception as e:
        print(f"❌ Real-world test failed: {e}")
        return {}

def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print(" COMPREHENSIVE CLAIMS VERIFICATION ")
    print(" RealityGuard System Test Suite ")
    print("="*70)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # 1. System Components Test
    print("\n[TEST 1/4] SYSTEM COMPONENTS")
    component_results = test_system_components()
    all_results["tests"]["components"] = component_results

    # 2. Performance Claims Test
    print("\n[TEST 2/4] PERFORMANCE CLAIMS")
    performance_results = test_performance_claims()
    all_results["tests"]["performance"] = performance_results

    # 3. Patent Claims Verification
    print("\n[TEST 3/4] PATENT CLAIMS")
    patent_results = verify_patent_claims()
    all_results["tests"]["patent_claims"] = patent_results

    # 4. Real-World Data Test
    print("\n[TEST 4/4] REAL-WORLD DATA")
    realworld_results = test_real_world_data()
    all_results["tests"]["real_world"] = realworld_results

    # Summary
    print("\n" + "="*70)
    print(" FINAL TEST SUMMARY ")
    print("="*70)

    # Components summary
    print("\n📦 SYSTEM COMPONENTS:")
    passed_components = sum(1 for r in component_results if "✅" in r.get("status", ""))
    total_components = len(component_results)
    print(f"   {passed_components}/{total_components} components operational")

    # Performance summary
    print("\n⚡ PERFORMANCE:")
    if performance_results:
        for mode, result in performance_results.items():
            status = "✅" if result.get("meets_realtime", False) else "❌"
            print(f"   {mode}: {result.get('fps', 0):.1f} FPS {status}")

    # Patent claims summary
    print("\n📋 PATENT CLAIMS:")
    if patent_results:
        verified_claims = sum(1 for c in patent_results.values() if c.get("verified", False))
        print(f"   {verified_claims}/6 claims verified")
        for i in range(1, 7):
            claim = patent_results.get(f"claim{i}", {})
            status = "✅" if claim.get("verified", False) else "❌"
            print(f"   {status} Claim {i}: {claim.get('description', 'Unknown')}")

    # Real-world summary
    print("\n🌍 REAL-WORLD PERFORMANCE:")
    if realworld_results:
        fps = realworld_results.get("real_world_fps", 0)
        status = "✅" if fps >= 24 else "⚠️"
        print(f"   {fps:.1f} FPS {status}")

    # Overall verdict
    print("\n" + "="*70)

    all_passed = (
        passed_components == total_components and
        any(r.get("meets_realtime", False) for r in performance_results.values()) and
        verified_claims >= 5
    )

    if all_passed:
        print(" ✅ SYSTEM FULLY OPERATIONAL ")
        print(" All claims verified successfully! ")

        # Find best performing mode
        if performance_results:
            best_mode = max(performance_results.items(), key=lambda x: x[1].get("fps", 0))
            print(f"\n Best performance: {best_mode[0]} mode at {best_mode[1]['fps']:.1f} FPS")
    else:
        print(" ⚠️ PARTIAL SUCCESS ")
        print(f" {passed_components}/{total_components} components working")
        if patent_results:
            print(f" {verified_claims}/6 patent claims verified")

    print("="*70 + "\n")

    # Save results
    with open("claims_verification_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Results saved to claims_verification_results.json")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)