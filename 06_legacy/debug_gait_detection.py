#!/usr/bin/env python3
"""
Debug why gait detection isn't working
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal


def debug_gait_detection():
    """Figure out why gait detection fails."""
    print("DEBUGGING GAIT DETECTION")
    print("=" * 60)

    # Generate clear gait signal
    duration = 10.0
    fs = 1000  # Sample rate
    t = np.linspace(0, duration, int(fs * duration))

    # Create VERY clear gait pattern
    gait_freq = 1.2  # Hz
    amplitude = 50
    y_gait = 240 + amplitude * np.sin(2 * np.pi * gait_freq * t)

    # Add small noise
    y_gait_noisy = y_gait + np.random.normal(0, 2, len(t))

    # Create random signal
    y_random = 240 + np.random.normal(0, amplitude, len(t))

    # Method 1: Simple FFT
    print("\nMethod 1: Simple FFT")
    print("-" * 40)

    def simple_fft_detection(y, fs):
        # Remove DC component
        y_centered = y - np.mean(y)

        # FFT
        fft = np.fft.rfft(y_centered)
        freqs = np.fft.rfftfreq(len(y_centered), 1/fs)

        # Find peaks
        peaks, _ = signal.find_peaks(np.abs(fft), height=len(y)/10)

        if len(peaks) > 0:
            peak_freqs = freqs[peaks]
            # Check if any peak in gait range
            gait_peaks = peak_freqs[(peak_freqs > 0.8) & (peak_freqs < 1.5)]
            return len(gait_peaks) > 0, peak_freqs
        return False, []

    has_gait_fft, peaks_fft = simple_fft_detection(y_gait_noisy, fs)
    no_gait_fft, _ = simple_fft_detection(y_random, fs)

    print(f"  Gait signal detected: {has_gait_fft}")
    if len(peaks_fft) > 0:
        print(f"  Peak frequencies: {peaks_fft[:3]}")
    print(f"  Random signal detected: {no_gait_fft}")

    # Method 2: Welch's method (more robust)
    print("\nMethod 2: Welch's Method")
    print("-" * 40)

    def welch_detection(y, fs):
        # Welch's method for PSD
        freqs, psd = signal.welch(y, fs, nperseg=2048)

        # Find dominant frequency
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]
        peak_power = psd[peak_idx]

        # Check if in gait range with sufficient power
        in_gait_range = 0.8 <= peak_freq <= 1.5
        sufficient_power = peak_power > np.mean(psd) * 10

        return in_gait_range and sufficient_power, peak_freq, peak_power

    has_gait_welch, freq_welch, power_welch = welch_detection(y_gait_noisy, fs)
    no_gait_welch, freq_random, power_random = welch_detection(y_random, fs)

    print(f"  Gait signal: Detected={has_gait_welch}, Freq={freq_welch:.2f} Hz")
    print(f"  Random signal: Detected={no_gait_welch}, Freq={freq_random:.2f} Hz")

    # Method 3: Autocorrelation (detect periodicity)
    print("\nMethod 3: Autocorrelation")
    print("-" * 40)

    def autocorr_detection(y, fs):
        # Normalize
        y = y - np.mean(y)
        y = y / np.std(y)

        # Autocorrelation
        autocorr = np.correlate(y, y, mode='same')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags

        # Find peaks in autocorrelation
        peaks, properties = signal.find_peaks(autocorr, height=0.3, distance=fs//3)

        if len(peaks) > 0:
            # First peak gives period
            period_samples = peaks[0]
            frequency = fs / period_samples
            in_gait = 0.8 <= frequency <= 1.5
            return in_gait, frequency
        return False, 0

    has_gait_auto, freq_auto = autocorr_detection(y_gait_noisy, fs)
    no_gait_auto, _ = autocorr_detection(y_random, fs)

    print(f"  Gait signal: Detected={has_gait_auto}, Freq={freq_auto:.2f} Hz")
    print(f"  Random signal: Detected={no_gait_auto}")

    # Method 4: Simplified for event data
    print("\nMethod 4: Event-Specific Detection")
    print("-" * 40)

    def event_gait_detection(y_positions, sample_rate=1000):
        """Simplified detection for event camera data."""
        # Smooth the signal first
        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y_positions, sigma=10)

        # Remove mean
        y_centered = y_smooth - np.mean(y_smooth)

        # Simple FFT
        fft = np.fft.rfft(y_centered)
        freqs = np.fft.rfftfreq(len(y_centered), 1/sample_rate)
        magnitudes = np.abs(fft)

        # Find power in gait band
        gait_band = (freqs >= 0.8) & (freqs <= 1.5)
        gait_power = np.sum(magnitudes[gait_band])
        total_power = np.sum(magnitudes[1:])  # Exclude DC

        # Threshold
        ratio = gait_power / total_power if total_power > 0 else 0
        has_gait = ratio > 0.2  # 20% of power in gait band

        # Find peak
        peak_idx = np.argmax(magnitudes[1:]) + 1
        peak_freq = freqs[peak_idx]

        return has_gait, peak_freq, ratio

    has_gait_event, freq_event, ratio_event = event_gait_detection(y_gait_noisy, fs)
    no_gait_event, _, ratio_random = event_gait_detection(y_random, fs)

    print(f"  Gait signal: Detected={has_gait_event}, Freq={freq_event:.2f} Hz, Ratio={ratio_event:.3f}")
    print(f"  Random signal: Detected={no_gait_event}, Ratio={ratio_random:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    methods_work = 0
    if has_gait_fft and not no_gait_fft:
        print("  ✓ Simple FFT works")
        methods_work += 1
    else:
        print("  ✗ Simple FFT failed")

    if has_gait_welch and not no_gait_welch:
        print("  ✓ Welch's method works")
        methods_work += 1
    else:
        print("  ✗ Welch's method failed")

    if has_gait_auto and not no_gait_auto:
        print("  ✓ Autocorrelation works")
        methods_work += 1
    else:
        print("  ✗ Autocorrelation failed")

    if has_gait_event and not no_gait_event:
        print("  ✓ Event-specific works")
        methods_work += 1
    else:
        print("  ✗ Event-specific failed")

    print(f"\n  Methods working: {methods_work}/4")

    if methods_work >= 2:
        print("\n✅ GAIT DETECTION IS POSSIBLE")
        print("   Multiple methods can detect gait patterns.")
        print("   Implementation needs correct parameters.")
    else:
        print("\n⚠️  GAIT DETECTION CHALLENGING")
        print("   May need more sophisticated methods.")

    return methods_work >= 2


if __name__ == "__main__":
    success = debug_gait_detection()