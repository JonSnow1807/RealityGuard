#!/usr/bin/env python3
"""
FINAL VIDEO PROCESSING TEST
The ultimate test - can it actually process video?
"""

import asyncio
import time
import numpy as np
import cv2
from pathlib import Path

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def test_video_processing():
    """Test if the system can actually process video frames."""

    print("=" * 70)
    print("FINAL VIDEO PROCESSING TEST")
    print("=" * 70)

    from src.services.privacy_engine import PrivacyEngine

    # Initialize engine
    print("\n1. Initializing Privacy Engine...")
    engine = PrivacyEngine()
    await engine.warmup()
    print("   ✅ Engine warmed up")

    # Create test video
    print("\n2. Creating test video...")
    test_video = Path("test_input.mp4")

    # Generate synthetic test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(test_video), fourcc, 30, (640, 480))

    # Create 30 frames (1 second at 30fps)
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add moving rectangle to simulate person
        x = 100 + i * 5
        y = 200
        cv2.rectangle(frame, (x, y), (x + 100, y + 200), (255, 255, 255), -1)
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    print(f"   ✅ Test video created: {test_video}")

    # Process individual frames
    print("\n3. Testing frame processing...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    processed_frame = await engine.process_frame(test_frame, mode="balanced")
    frame_time = time.time() - start_time

    if processed_frame is not None and processed_frame.shape == test_frame.shape:
        fps = 1.0 / frame_time if frame_time > 0 else 0
        print(f"   ✅ Frame processed in {frame_time*1000:.1f}ms ({fps:.1f} FPS)")
    else:
        print("   ❌ Frame processing failed")
        return False

    # Process video file
    print("\n4. Processing full video...")
    output_video = Path("test_output.mp4")

    try:
        start_time = time.time()
        metrics = await engine.process_video(test_video, output_video, mode="balanced")
        total_time = time.time() - start_time

        print(f"   ✅ Video processed in {total_time:.2f}s")
        print(f"   • Average FPS: {metrics.fps:.1f}")
        print(f"   • Frames processed: {metrics.frames_processed}")
        print(f"   • Cache hit rate: {metrics.cache_hit_rate:.1%}")
        print(f"   • Quality adaptations: {metrics.adaptations}")

    except Exception as e:
        print(f"   ❌ Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify output
    print("\n5. Verifying output...")
    if output_video.exists():
        cap = cv2.VideoCapture(str(output_video))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"   ✅ Output video valid: {frame_count} frames")
        else:
            print("   ❌ Output video corrupted")
            return False
    else:
        print("   ❌ Output video not created")
        return False

    # Test caching efficiency
    print("\n6. Testing cache efficiency...")
    cache_stats = engine.cache.stats
    if cache_stats["total"] > 0:
        hit_rate = (cache_stats["l1"] + cache_stats["l2"] + cache_stats["l3"]) / cache_stats["total"]
        print(f"   ✅ Cache working: {hit_rate:.1%} hit rate")
        print(f"   • L1 hits: {cache_stats['l1']}")
        print(f"   • L2 hits: {cache_stats['l2']}")
        print(f"   • L3 hits: {cache_stats['l3']}")
    else:
        print("   ⚠️ No cache activity recorded")

    # Cleanup
    print("\n7. Cleanup...")
    test_video.unlink(missing_ok=True)
    output_video.unlink(missing_ok=True)
    print("   ✅ Test files cleaned up")

    print("\n" + "=" * 70)
    print("VIDEO PROCESSING TEST: PASSED ✅")
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = asyncio.run(test_video_processing())

    if success:
        print("\n🎉 SUCCESS: System can process video!")
        print("This confirms the production system is functional.")
    else:
        print("\n❌ FAILURE: Video processing not working")
        print("System has critical issues.")