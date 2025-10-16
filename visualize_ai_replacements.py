#!/usr/bin/env python3
"""
Visualize the AI replacements to see what's actually happening
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_visual_comparison():
    """Create side-by-side comparison of original vs AI replaced"""

    # Load original and replaced images
    original = cv2.imread('real_test_1.jpg')
    synthetic = cv2.imread('ai_test_synthetic_face_real_test_1.jpg')
    blur = cv2.imread('ai_test_smart_blur_real_test_1.jpg')

    if original is None or synthetic is None:
        print("Could not load test images")
        return

    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    synthetic_rgb = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB) if synthetic is not None else None
    blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB) if blur is not None else None

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Synthetic face replacement
    if synthetic_rgb is not None:
        axes[0, 1].imshow(synthetic_rgb)
        diff_synthetic = np.abs(original.astype(float) - synthetic.astype(float))
        avg_diff_s = np.mean(diff_synthetic)
        axes[0, 1].set_title(f'Synthetic Face (diff: {avg_diff_s:.1f}px)')
        axes[0, 1].axis('off')

        # Difference heatmap
        diff_gray_s = np.mean(diff_synthetic, axis=2)
        im1 = axes[0, 2].imshow(diff_gray_s, cmap='hot', vmin=0, vmax=50)
        axes[0, 2].set_title('Difference Map (Synthetic)')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2])

    # Smart blur
    if blur_rgb is not None:
        axes[1, 0].imshow(blur_rgb)
        diff_blur = np.abs(original.astype(float) - blur.astype(float))
        avg_diff_b = np.mean(diff_blur)
        axes[1, 0].set_title(f'Smart Blur (diff: {avg_diff_b:.1f}px)')
        axes[1, 0].axis('off')

        # Difference heatmap
        diff_gray_b = np.mean(diff_blur, axis=2)
        im2 = axes[1, 1].imshow(diff_gray_b, cmap='hot', vmin=0, vmax=50)
        axes[1, 1].set_title('Difference Map (Blur)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])

    # Zoom on face region (if detected)
    h, w = original.shape[:2]
    # Approximate face region
    face_y1, face_y2 = h//4, h//2
    face_x1, face_x2 = w//3, 2*w//3

    # Zoomed comparison
    zoom_orig = original_rgb[face_y1:face_y2, face_x1:face_x2]
    axes[1, 2].imshow(zoom_orig)
    axes[1, 2].set_title('Original (Zoomed)')
    axes[1, 2].axis('off')

    plt.suptitle('AI Replacement System - Visual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ai_replacement_visual_analysis.jpg', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVisual analysis saved to: ai_replacement_visual_analysis.jpg")

    # Print detailed statistics
    print("\nDetailed Statistics:")
    if synthetic is not None:
        print(f"Synthetic Face:")
        print(f"  Average difference: {avg_diff_s:.1f}px")
        print(f"  Max difference: {np.max(diff_synthetic):.1f}px")
        print(f"  Pixels > 10: {np.sum(diff_gray_s > 10) / diff_gray_s.size * 100:.1f}%")
        print(f"  Pixels > 20: {np.sum(diff_gray_s > 20) / diff_gray_s.size * 100:.1f}%")

    if blur is not None:
        print(f"Smart Blur:")
        print(f"  Average difference: {avg_diff_b:.1f}px")
        print(f"  Max difference: {np.max(diff_blur):.1f}px")
        print(f"  Pixels > 5: {np.sum(diff_gray_b > 5) / diff_gray_b.size * 100:.1f}%")
        print(f"  Pixels > 10: {np.sum(diff_gray_b > 10) / diff_gray_b.size * 100:.1f}%")


def check_video_frames():
    """Check a few frames from the output video"""

    cap_orig = cv2.VideoCapture('real_people_video.mp4')
    cap_synth = cv2.VideoCapture('ai_output_synthetic_face_real_people_video.mp4')

    if not cap_orig.isOpened() or not cap_synth.isOpened():
        print("Could not open videos")
        return

    # Read first frame
    ret1, frame_orig = cap_orig.read()
    ret2, frame_synth = cap_synth.read()

    if ret1 and ret2:
        diff = np.mean(np.abs(frame_orig.astype(float) - frame_synth.astype(float)))
        print(f"\nVideo frame difference: {diff:.1f}px")

        # Save frame comparison
        combined = np.hstack([frame_orig, frame_synth])
        cv2.imwrite('ai_video_frame_comparison.jpg', combined)
        print("Frame comparison saved to: ai_video_frame_comparison.jpg")

    cap_orig.release()
    cap_synth.release()


if __name__ == "__main__":
    print("Analyzing AI replacement results...")
    create_visual_comparison()
    check_video_frames()