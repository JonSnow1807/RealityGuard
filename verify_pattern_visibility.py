#!/usr/bin/env python3
"""
Verify if patterns are ACTUALLY visible in the output
Compare original vs processed images side-by-side
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_pattern_visibility(original_path, processed_path, output_name):
    """Analyze and visualize pattern application"""

    # Load images
    original = cv2.imread(original_path)
    processed = cv2.imread(processed_path)

    if original is None or processed is None:
        print(f"Could not load images: {original_path} or {processed_path}")
        return None

    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # Calculate difference
    diff = np.abs(original.astype(float) - processed.astype(float))
    pixel_diff = np.mean(diff)
    max_diff = np.max(diff)

    # Create difference heatmap
    diff_gray = np.mean(diff, axis=2)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Processed
    axes[0, 1].imshow(processed_rgb)
    axes[0, 1].set_title(f'Processed (Avg diff: {pixel_diff:.2f}px)')
    axes[0, 1].axis('off')

    # Difference heatmap
    im = axes[0, 2].imshow(diff_gray, cmap='hot', vmin=0, vmax=50)
    axes[0, 2].set_title(f'Difference Heatmap (Max: {max_diff:.1f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # Zoomed regions (center)
    h, w = original.shape[:2]
    cy, cx = h//2, w//2
    crop_size = 150

    # Original crop
    crop_orig = original_rgb[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    axes[1, 0].imshow(crop_orig)
    axes[1, 0].set_title('Original (Zoomed)')
    axes[1, 0].axis('off')

    # Processed crop
    crop_proc = processed_rgb[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    axes[1, 1].imshow(crop_proc)
    axes[1, 1].set_title('Processed (Zoomed)')
    axes[1, 1].axis('off')

    # Difference crop
    crop_diff = diff_gray[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    axes[1, 2].imshow(crop_diff, cmap='hot', vmin=0, vmax=50)
    axes[1, 2].set_title('Difference (Zoomed)')
    axes[1, 2].axis('off')

    # Add statistics
    stats_text = f"""
    Statistics:
    - Average pixel difference: {pixel_diff:.2f}
    - Max pixel difference: {max_diff:.1f}
    - Std deviation: {np.std(diff):.2f}
    - Pixels changed (>5): {np.sum(diff_gray > 5) / diff_gray.size * 100:.1f}%
    - Pixels changed (>10): {np.sum(diff_gray > 10) / diff_gray.size * 100:.1f}%
    """

    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')

    plt.suptitle(f'Pattern Visibility Analysis: {output_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'visibility_analysis_{output_name}.jpg', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'avg_diff': pixel_diff,
        'max_diff': max_diff,
        'std_diff': np.std(diff),
        'pixels_changed_5': np.sum(diff_gray > 5) / diff_gray.size * 100,
        'pixels_changed_10': np.sum(diff_gray > 10) / diff_gray.size * 100
    }


def main():
    """Analyze all test results"""

    print("="*80)
    print("PATTERN VISIBILITY VERIFICATION")
    print("="*80)

    # Find test result pairs
    test_cases = [
        ('test_adversarial_v1.jpg', 'test_result_ultimate_test_adversarial_v1.jpg', 'ultimate'),
        ('test_adversarial_v1.jpg', 'test_result_gpu_test_adversarial_v1.jpg', 'gpu'),
        ('test_adversarial_v1.jpg', 'test_result_patent_test_adversarial_v1.jpg', 'patent'),
    ]

    results = {}

    for orig, proc, impl_name in test_cases:
        if os.path.exists(orig) and os.path.exists(proc):
            print(f"\nAnalyzing {impl_name} implementation:")
            print(f"  Original: {orig}")
            print(f"  Processed: {proc}")

            stats = analyze_pattern_visibility(orig, proc, impl_name)
            if stats:
                results[impl_name] = stats

                print(f"  Results:")
                print(f"    Average difference: {stats['avg_diff']:.2f}px")
                print(f"    Max difference: {stats['max_diff']:.1f}px")
                print(f"    Pixels changed (>5): {stats['pixels_changed_5']:.1f}%")
                print(f"    Pixels changed (>10): {stats['pixels_changed_10']:.1f}%")

                # Assessment
                if stats['avg_diff'] < 2:
                    print("    ⚠️ WARNING: Patterns may be too weak or not applied!")
                elif stats['avg_diff'] > 15:
                    print("    ⚠️ WARNING: Patterns may be too strong!")
                else:
                    print("    ✓ Patterns are in optimal range (2-15px)")

    # Summary
    print("\n" + "="*80)
    print("VISIBILITY VERIFICATION SUMMARY")
    print("="*80)

    for impl_name, stats in results.items():
        print(f"\n{impl_name.upper()}:")
        if stats['avg_diff'] >= 2 and stats['avg_diff'] <= 15:
            print(f"  ✓ VISIBLE AND EFFECTIVE: {stats['avg_diff']:.2f}px average difference")
            print(f"    {stats['pixels_changed_5']:.1f}% of pixels modified")
        else:
            print(f"  ✗ INEFFECTIVE: {stats['avg_diff']:.2f}px average difference")
            if stats['avg_diff'] < 2:
                print(f"    Patterns too weak or not being applied")
            else:
                print(f"    Patterns too strong, may degrade quality")

    print("\n" + "="*80)
    print("Visual comparison images saved as visibility_analysis_*.jpg")
    print("="*80)


if __name__ == "__main__":
    main()