#!/usr/bin/env python3
"""
sound2video.py — REVERSE STRIPE ENGINE

The inverse of video2dnb_v8: takes a WAV file and reconstructs
a video from its audio content.

Since v8 mapped:
  - Rows → frequency bands (bottom=low, top=high)
  - Columns → time slices within each frame
  - Red → bass (30-300 Hz)
  - Green → mid (300-3000 Hz)
  - Blue → treble (3000-14000 Hz)
  - Stripe spatial frequencies → audio tones
  - Brightness transitions → transients

We reverse all of that:
  - Compute STFT spectrogram of the audio
  - Split into 3 frequency bands → R, G, B channels
  - Each frame = one time window of the spectrogram
  - Frequency bins map to pixel rows (low=bottom, high=top)
  - Amplitude → pixel brightness
  - Transient energy → horizontal edge lines (stripe artifacts)
  - Stereo difference → left/right pixel brightness shift

The result won't be the original video, but a visual "ghost"
that mirrors its structure — stripes, color bands, brightness
patterns all reconstructed from the audio they created.

Usage:
    python sound2video.py audio.wav --output output.mp4
    python sound2video.py audio.wav --fps 50 --width 704 --height 576
    python sound2video.py audio.wav --original video.mov  # side-by-side!
"""

import numpy as np
import soundfile as sf
from scipy import signal
import cv2
import argparse
import os
import sys
import time


# ─── Constants (matching v8) ─────────────────────────────────────────────────

BASS_RANGE = (30, 300)       # → Red channel
MID_RANGE = (300, 3000)      # → Green channel  
TREBLE_RANGE = (3000, 14000) # → Blue channel


def audio_to_video(wav_path, output_path=None, fps=50, width=704, height=576,
                   original_path=None, side_by_side=False, brightness=2.0,
                   colorize=True, add_stripes=True):
    """
    Reverse-engineer a video from audio using inverse spectrogram mapping.
    """
    print(f"""
============================================================
  SOUND2VIDEO — REVERSE STRIPE ENGINE
  Audio:  {wav_path}
  Output: {output_path or 'auto'}
  FPS:    {fps}
  Size:   {width}x{height}
============================================================
""")

    # ─── Load audio ───────────────────────────────────────────────────────
    print("[1/4] Loading audio...")
    audio, sr = sf.read(wav_path)
    if audio.ndim == 2:
        left = audio[:, 0]
        right = audio[:, 1]
        mono = (left + right) / 2
    else:
        mono = audio
        left = audio
        right = audio

    duration = len(mono) / sr
    total_frames = int(duration * fps)
    samples_per_frame = int(sr / fps)

    print(f"  Duration: {duration:.1f}s, SR: {sr}, Frames: {total_frames}")

    # ─── Compute spectrograms ────────────────────────────────────────────
    print("[2/4] Computing spectrograms...")

    # STFT parameters — window size matches frame duration for exact sync
    nperseg = min(samples_per_frame, 4096)
    noverlap = nperseg // 2
    
    # Full spectrogram
    freqs, times, Zxx_L = signal.stft(left, sr, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_R = signal.stft(right, sr, nperseg=nperseg, noverlap=noverlap)
    
    mag_L = np.abs(Zxx_L)
    mag_R = np.abs(Zxx_R)
    mag_mono = (mag_L + mag_R) / 2

    # Find frequency bin indices for each RGB band
    bass_mask = (freqs >= BASS_RANGE[0]) & (freqs <= BASS_RANGE[1])
    mid_mask = (freqs >= MID_RANGE[0]) & (freqs <= MID_RANGE[1])
    treble_mask = (freqs >= TREBLE_RANGE[0]) & (freqs <= TREBLE_RANGE[1])

    bass_bins = mag_mono[bass_mask, :]
    mid_bins = mag_mono[mid_mask, :]
    treble_bins = mag_mono[treble_mask, :]

    # Compute transient energy (spectral flux) for stripe generation
    spectral_flux = np.zeros(mag_mono.shape[1])
    for t in range(1, mag_mono.shape[1]):
        diff = mag_mono[:, t] - mag_mono[:, t - 1]
        spectral_flux[t] = np.sum(np.maximum(diff, 0))
    
    # Normalize
    sf_max = np.max(spectral_flux)
    if sf_max > 0:
        spectral_flux /= sf_max

    # Stereo width per time step
    stereo_diff = np.zeros(mag_mono.shape[1])
    for t in range(mag_mono.shape[1]):
        stereo_diff[t] = np.mean(np.abs(mag_L[:, t] - mag_R[:, t]))
    sd_max = np.max(stereo_diff)
    if sd_max > 0:
        stereo_diff /= sd_max

    print(f"  Spectrogram: {mag_mono.shape[0]} freq bins × {mag_mono.shape[1]} time steps")
    print(f"  Bass bins: {bass_bins.shape[0]}, Mid: {mid_bins.shape[0]}, Treble: {treble_bins.shape[0]}")

    # ─── Load original video for side-by-side (optional) ─────────────────
    orig_cap = None
    if original_path and os.path.exists(original_path):
        orig_cap = cv2.VideoCapture(original_path)
        print(f"  Original video loaded for comparison: {original_path}")

    # ─── Generate video ──────────────────────────────────────────────────
    print("[3/4] Generating frames...")

    if output_path is None:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        output_path = os.path.join(os.path.dirname(wav_path), f"{base}_reversed.mp4")

    out_width = width * 2 if (side_by_side and orig_cap) else width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    if not out.isOpened():
        print(f"Error: Cannot create video writer for {output_path}")
        sys.exit(1)

    for fi in range(total_frames):
        # Find corresponding spectrogram time index
        t_sec = fi / fps
        t_idx = int(t_sec / (duration + 1e-10) * (mag_mono.shape[1] - 1))
        t_idx = min(t_idx, mag_mono.shape[1] - 1)

        # Get a window of spectrogram columns for this frame
        # (use multiple columns for horizontal detail)
        window_width = max(1, mag_mono.shape[1] // total_frames * 2)
        t_start = max(0, t_idx - window_width // 2)
        t_end = min(mag_mono.shape[1], t_start + window_width)

        # ─── Build RGB channels from frequency bands ──────────────────
        def band_to_image(band_data, t0, t1, target_h, target_w):
            """Convert a frequency band slice to a grayscale image."""
            chunk = band_data[:, t0:t1]
            if chunk.size == 0:
                return np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Flip vertically: low freq at bottom, high at top
            chunk = chunk[::-1, :]
            
            # Normalize with some headroom
            p99 = np.percentile(chunk, 99) if chunk.size > 0 else 1.0
            if p99 > 0:
                chunk = chunk / p99
            chunk = np.clip(chunk * brightness, 0, 1)
            
            # Resize to target dimensions
            img = cv2.resize(chunk, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return (img * 255).astype(np.uint8)

        # Each channel gets 1/3 of the vertical space, or full height
        red_img = band_to_image(bass_bins, t_start, t_end, height, width)
        green_img = band_to_image(mid_bins, t_start, t_end, height, width)
        blue_img = band_to_image(treble_bins, t_start, t_end, height, width)

        # ─── Add stripe artifacts (reverse of v8's transition detection) ─
        if add_stripes:
            flux = spectral_flux[t_idx]
            if flux > 0.15:
                # Create horizontal stripes proportional to transient energy
                n_stripes = int(flux * 20) + 2
                stripe_intensity = int(flux * 180)
                for s in range(n_stripes):
                    y = np.random.randint(0, height)
                    thickness = np.random.randint(1, max(2, int(flux * 6)))
                    y_end = min(y + thickness, height)
                    # Stripes primarily in red channel (matching VHS artifacts)
                    red_img[y:y_end, :] = np.clip(
                        red_img[y:y_end, :].astype(int) + stripe_intensity, 0, 255
                    ).astype(np.uint8)
                    # Some in blue too (CRT color bleeding)
                    if np.random.random() < 0.4:
                        blue_img[y:y_end, :] = np.clip(
                            blue_img[y:y_end, :].astype(int) + stripe_intensity // 2, 0, 255
                        ).astype(np.uint8)

        # ─── Stereo → left/right brightness shift ────────────────────────
        sw = stereo_diff[t_idx]
        if sw > 0.1:
            shift = int(sw * 30)
            # Shift red slightly left, blue slightly right
            if shift > 0 and shift < width:
                red_shifted = np.zeros_like(red_img)
                red_shifted[:, :width - shift] = red_img[:, shift:]
                red_img = red_shifted
                
                blue_shifted = np.zeros_like(blue_img)
                blue_shifted[:, shift:] = blue_img[:, :width - shift]
                blue_img = blue_shifted

        # ─── Combine into BGR frame ──────────────────────────────────────
        frame = cv2.merge([blue_img, green_img, red_img])

        # ─── Optional: add scanline effect ────────────────────────────────
        if colorize:
            # Add subtle CRT scanline effect (darken every other row)
            scanline_mask = np.ones((height, width), dtype=np.float32)
            scanline_mask[1::2, :] = 0.85
            for c in range(3):
                frame[:, :, c] = (frame[:, :, c].astype(float) * scanline_mask).astype(np.uint8)

        # ─── Side-by-side with original ───────────────────────────────────
        if orig_cap:
            ret, orig_frame = orig_cap.read()
            if ret:
                orig_resized = cv2.resize(orig_frame, (width, height))
                if side_by_side:
                    # Add labels
                    cv2.putText(orig_resized, "ORIGINAL", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, "FROM AUDIO", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frame = np.hstack([orig_resized, frame])
                else:
                    # Blend 
                    frame = cv2.addWeighted(orig_resized, 0.3, frame, 0.7, 0)

        out.write(frame)

        if fi % 100 == 0:
            pct = fi / total_frames * 100
            print(f"\r  Generating [{('#' * int(pct / 2.5)):40s}] {pct:.0f}%", end="", flush=True)

    print(f"\r  Generating [{'#' * 40}] 100%")
    out.release()
    if orig_cap:
        orig_cap.release()

    # ─── Add audio to the video ───────────────────────────────────────────
    print("\n[4/4] Adding audio track...")
    final_path = output_path.replace('.mp4', '_final.mp4')
    os.system(f'ffmpeg -y -i "{output_path}" -i "{wav_path}" '
              f'-c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 '
              f'-shortest "{final_path}" 2>/dev/null')

    if os.path.exists(final_path):
        os.replace(final_path, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"""
============================================================
  Output:     {output_path}
  Duration:   {duration:.1f}s
  Frames:     {total_frames} at {fps} fps
  Size:       {size_mb:.1f} MB
  ──────────────────────────────────────────────────
  AUDIO → VIDEO (reverse spectrogram)
  Bass → Red | Mid → Green | Treble → Blue
  Transients → stripe artifacts
  Stereo width → RGB channel shift
============================================================""")

    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='sound2video — Reverse Stripe Engine. Reconstruct video from audio.')
    parser.add_argument('audio', help='Input WAV file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=50, help='Output frame rate')
    parser.add_argument('--width', type=int, default=704, help='Output width')
    parser.add_argument('--height', type=int, default=576, help='Output height')
    parser.add_argument('--original', type=str, default=None,
                        help='Original video for side-by-side comparison')
    parser.add_argument('--side-by-side', action='store_true',
                        help='Show original and reconstructed side by side')
    parser.add_argument('--brightness', type=float, default=2.5,
                        help='Brightness multiplier (default 2.5)')
    parser.add_argument('--no-stripes', action='store_true',
                        help='Disable synthetic stripe artifacts')
    parser.add_argument('--no-scanlines', action='store_true',
                        help='Disable CRT scanline effect')
    args = parser.parse_args()

    t0 = time.time()
    audio_to_video(
        wav_path=args.audio,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        original_path=args.original,
        side_by_side=args.side_by_side,
        brightness=args.brightness,
        add_stripes=not args.no_stripes,
        colorize=not args.no_scanlines,
    )
    print(f"  Render time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
