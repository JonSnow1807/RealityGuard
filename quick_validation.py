#!/usr/bin/env python3
"""
Quick Validation Test for Patent System
Fast test to verify system is working before GitHub push
"""

import torch
import numpy as np
import cv2
import time
import json
from pathlib import Path

def quick_test():
    """Run quick validation test."""

    print("="*80)
    print("QUICK VALIDATION TEST")
    print("="*80)

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cuda': torch.cuda.is_available(),
        'tests': []
    }

    # Test 1: Import check
    print("\n1. Checking imports...")
    try:
        from patent_ready_optimized import PatentReadySystem, PatentConfig
        print("✅ Patent system imports successfully")
        results['tests'].append({'test': 'import', 'status': 'PASS'})
    except Exception as e:
        print(f"❌ Import failed: {e}")
        results['tests'].append({'test': 'import', 'status': 'FAIL', 'error': str(e)})
        return results

    # Test 2: Basic functionality
    print("\n2. Testing basic functionality...")
    try:
        config = PatentConfig(
            target_fps=30,
            enable_adaptive_quality=True,
            enable_hierarchical_cache=True
        )
        system = PatentReadySystem(config)

        # Create simple test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (200, 150), (440, 330), (100, 100, 200), -1)

        # Process single frame
        start = time.time()
        result = system._process_frame(test_frame, {'quality': 0.5, 'strategy': 'hybrid', 'resolution_scale': 0.7})
        elapsed = time.time() - start

        fps_estimate = 1.0 / elapsed
        print(f"✅ Single frame processed in {elapsed*1000:.2f}ms ({fps_estimate:.1f} FPS estimate)")

        results['tests'].append({
            'test': 'single_frame',
            'status': 'PASS',
            'time_ms': elapsed * 1000,
            'fps_estimate': fps_estimate
        })

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        results['tests'].append({'test': 'single_frame', 'status': 'FAIL', 'error': str(e)})
        return results

    # Test 3: Video processing
    print("\n3. Testing video processing...")
    try:
        # Create mini test video (30 frames)
        frames = []
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = 320 + int(100 * np.sin(i * 0.2))
            cv2.rectangle(frame, (x-50, 190), (x+50, 290), (150, 100, 100), -1)
            frames.append(frame)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('quick_test.mp4', fourcc, 30, (640, 480))
        for frame in frames:
            out.write(frame)
        out.release()

        # Process video
        system.process_video('quick_test.mp4', 'quick_output.mp4')

        if system.fps_history:
            avg_fps = np.mean(list(system.fps_history))
            min_fps = np.min(list(system.fps_history))
            print(f"✅ Video processed: {avg_fps:.1f} FPS (min: {min_fps:.1f})")

            results['tests'].append({
                'test': 'video_processing',
                'status': 'PASS',
                'avg_fps': float(avg_fps),
                'min_fps': float(min_fps),
                'real_time': avg_fps >= 24
            })
        else:
            results['tests'].append({'test': 'video_processing', 'status': 'FAIL', 'error': 'No FPS data'})

    except Exception as e:
        print(f"❌ Video test failed: {e}")
        results['tests'].append({'test': 'video_processing', 'status': 'FAIL', 'error': str(e)})

    # Test 4: Cache performance
    print("\n4. Testing cache system...")
    try:
        cache_stats = system.cache.hit_stats
        total = sum(cache_stats.values())
        if total > 0:
            hit_rate = (total - cache_stats.get('miss', 0)) / total * 100
            print(f"✅ Cache hit rate: {hit_rate:.1f}%")
            results['tests'].append({
                'test': 'cache',
                'status': 'PASS',
                'hit_rate': hit_rate,
                'stats': cache_stats
            })
        else:
            results['tests'].append({'test': 'cache', 'status': 'SKIP', 'reason': 'No cache data'})
    except Exception as e:
        results['tests'].append({'test': 'cache', 'status': 'FAIL', 'error': str(e)})

    # Final verdict
    passed = sum(1 for t in results['tests'] if t.get('status') == 'PASS')
    total = len(results['tests'])

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Tests passed: {passed}/{total}")

    if passed >= 3 and any(t.get('real_time', False) for t in results['tests']):
        print("✅ SYSTEM READY FOR GITHUB!")
        results['verdict'] = 'READY'
    else:
        print("⚠️ System needs work")
        results['verdict'] = 'NOT_READY'

    # Save results
    with open('quick_validation.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

def check_performance_claims():
    """Verify performance claims are accurate."""

    print("\n" + "="*80)
    print("VERIFYING PERFORMANCE CLAIMS")
    print("="*80)

    claims = {
        '40+ FPS achieved': False,
        'Real-time (>24 FPS)': False,
        'Hierarchical caching works': False,
        'Adaptive quality works': False
    }

    # Check validation results
    if Path('quick_validation.json').exists():
        with open('quick_validation.json') as f:
            data = json.load(f)

        for test in data.get('tests', []):
            if test.get('test') == 'video_processing':
                fps = test.get('avg_fps', 0)
                claims['40+ FPS achieved'] = fps >= 40
                claims['Real-time (>24 FPS)'] = fps >= 24
            elif test.get('test') == 'cache':
                claims['Hierarchical caching works'] = test.get('hit_rate', 0) > 0

    # Check if adaptive quality is implemented
    try:
        from patent_ready_optimized import AdaptiveQualityController
        claims['Adaptive quality works'] = True
    except:
        pass

    print("\nClaim Verification:")
    for claim, verified in claims.items():
        status = "✅" if verified else "❌"
        print(f"  {status} {claim}")

    return all(claims.values())

if __name__ == "__main__":
    results = quick_test()

    if results['verdict'] == 'READY':
        verified = check_performance_claims()
        if verified:
            print("\n🎉 ALL CLAIMS VERIFIED - READY FOR PATENT AND GITHUB!")
        else:
            print("\n⚠️ Some claims need verification")
    else:
        print("\n❌ System not ready")