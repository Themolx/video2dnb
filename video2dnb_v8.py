#!/usr/bin/env python3
"""
video2dnb_v8.py — STRIPE SONIFICATION ENGINE

The image IS the synthesizer. No preset sounds. Every audio sample
is derived directly from pixel values.

Core principles:
  1. STRIPE → TONE: Horizontal stripes (like CRT/VHS artifacts) are detected
     via FFT of vertical pixel columns. Stripe spatial frequency → audio pitch.
     Stripe amplitude → loudness. The stripes literally become tones.

  2. FRAME = SPECTROGRAM: Each frame is read as a spectrogram:
     - Y-axis (rows) = frequency (bottom=low, top=high)
     - X-axis (columns) = time within one beat
     - Pixel brightness = amplitude at that freq/time point
     This means the IMAGE directly defines the sound, not statistics about it.

  3. RGB = THREE FREQUENCY BANDS:
     - Red channel → bass layer (30-300 Hz)
     - Green channel → mid layer (300-3000 Hz)
     - Blue channel → treble layer (3000-14000 Hz)
     Each channel is independently sonified.

  4. TRANSITIONS = TRANSIENTS: Sharp brightness changes between adjacent
     rows create impulse-like transients — NOT preset kick/snare sounds,
     but the actual derivative of the image brightness curve.

  5. COLOR EDGES = GLITCH EVENTS: Where RGB channels diverge sharply
     (like red/blue stripe boundaries) → resonant filter sweeps.

  6. BPM GRID: Still quantized to musical time for structure.

Usage:
    python video2dnb_v8.py <video_path> [--bpm 174] [--output out.wav]
    python video2dnb_v8.py video.mp4 --brightness 0.8  # sonification brightness
    python video2dnb_v8.py video.mp4 --transient 0.7   # transient emphasis
"""

import cv2
import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import os
import sys
import time


# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 48000

# Frequency ranges for RGB channel sonification
BASS_RANGE = (30, 300)      # Red channel
MID_RANGE = (300, 3000)     # Green channel
TREBLE_RANGE = (3000, 14000)  # Blue channel

# Spectrogram sonification: how many frequency bands per channel
N_BANDS_BASS = 12
N_BANDS_MID = 24
N_BANDS_TREBLE = 24

# Generate frequency grids for each channel
BASS_FREQS = np.logspace(np.log10(BASS_RANGE[0]), np.log10(BASS_RANGE[1]), N_BANDS_BASS)
MID_FREQS = np.logspace(np.log10(MID_RANGE[0]), np.log10(MID_RANGE[1]), N_BANDS_MID)
TREBLE_FREQS = np.logspace(np.log10(TREBLE_RANGE[0]), np.log10(TREBLE_RANGE[1]), N_BANDS_TREBLE)


# ─── Image-to-Sound Primitives ───────────────────────────────────────────────

def frame_to_spectrogram_bands(channel, n_bands):
    """
    Treat a single-channel image as a spectrogram.
    Rows map to frequency bands, columns map to time steps.
    Returns [n_bands, n_cols] amplitude matrix.
    """
    h, w = channel.shape
    # Resample rows to n_bands (bottom=low freq, top=high freq)
    band_indices = np.linspace(h - 1, 0, n_bands).astype(int)
    bands = channel[band_indices, :].astype(float) / 255.0
    return bands


def stripe_detect(channel):
    """
    Detect horizontal stripe patterns via FFT of vertical columns.
    Returns dominant spatial frequencies and their amplitudes.
    
    This is THE key innovation: CRT/VHS horizontal stripes have
    specific spatial periodicities. We find them and map to audio.
    """
    h, w = channel.shape
    # Sample several columns across the width
    n_sample_cols = min(32, w)
    col_indices = np.linspace(0, w - 1, n_sample_cols).astype(int)
    
    # Average FFT across sampled columns for robust stripe detection
    fft_accum = np.zeros(h // 2)
    for ci in col_indices:
        col_data = channel[:, ci].astype(float) / 255.0
        col_data -= np.mean(col_data)  # remove DC
        fft_mag = np.abs(np.fft.rfft(col_data))[:h // 2]
        fft_accum += fft_mag
    fft_accum /= n_sample_cols
    
    # Normalize
    peak = np.max(fft_accum)
    if peak > 0:
        fft_accum /= peak
    
    # Find peaks (dominant stripe frequencies)
    # Spatial frequency in cycles/pixel → we map to audio Hz
    spatial_freqs = np.arange(len(fft_accum)) / h  # cycles per pixel
    
    # Find the top N peaks
    n_peaks = 8
    if len(fft_accum) > 3:
        # Simple peak finding: find local maxima above threshold
        peaks = []
        for i in range(1, len(fft_accum) - 1):
            if fft_accum[i] > fft_accum[i-1] and fft_accum[i] > fft_accum[i+1]:
                if fft_accum[i] > 0.1:  # significance threshold
                    peaks.append((spatial_freqs[i], float(fft_accum[i])))
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:n_peaks]
    else:
        peaks = []
    
    return peaks, spatial_freqs, fft_accum


def spatial_freq_to_audio_hz(spatial_freq, base_freq=55.0, scale=2000.0):
    """
    Map spatial frequency (cycles/pixel) to audio frequency (Hz).
    Higher spatial freq (tighter stripes) → higher pitch.
    """
    return base_freq + spatial_freq * scale


def transitions_to_impulses(channel, sr, n_samples):
    """
    Sharp brightness transitions between adjacent rows → transient impulses.
    This replaces preset kick/snare — the image gradient IS the percussion.
    """
    h, w = channel.shape
    # Compute vertical gradient (row-to-row brightness change)
    gray_float = channel.astype(float) / 255.0
    # Average across columns for a 1D vertical profile
    profile = np.mean(gray_float, axis=1)
    # Derivative = rate of brightness change
    derivative = np.diff(profile)
    # Absolute value — both dark→light and light→dark transitions matter
    abs_deriv = np.abs(derivative)
    
    # Map derivative peaks to time positions in the audio buffer
    # Each row position maps to a position in the n_samples buffer
    impulse_buf = np.zeros(n_samples)
    
    for i in range(len(abs_deriv)):
        if abs_deriv[i] > 0.05:  # threshold for "sharp transition"
            # Map row position to sample position
            t_pos = int(i / len(abs_deriv) * n_samples)
            t_pos = min(t_pos, n_samples - 1)
            # Impulse strength proportional to transition sharpness
            strength = min(abs_deriv[i] * 3.0, 1.0)
            # Create a short click/impulse (not a preset sound!)
            click_len = min(int(sr * 0.003), n_samples - t_pos)
            if click_len > 1:
                t = np.arange(click_len) / sr
                # The click's character comes from the local pixel color
                click = strength * np.exp(-t * 800)
                impulse_buf[t_pos:t_pos + click_len] += click
    
    return impulse_buf


def color_edge_resonance(frame, sr, n_samples):
    """
    Where RGB channels diverge sharply → resonant sweep.
    This detects glitch boundaries where red and blue separate.
    """
    b, g, r = frame[:, :, 0].astype(float), frame[:, :, 1].astype(float), frame[:, :, 2].astype(float)
    
    # RGB divergence per row (averaged across columns)
    rg_diff = np.mean(np.abs(r - g), axis=1) / 255.0
    rb_diff = np.mean(np.abs(r - b), axis=1) / 255.0
    gb_diff = np.mean(np.abs(g - b), axis=1) / 255.0
    
    divergence = (rg_diff + rb_diff + gb_diff) / 3.0
    
    # Map to audio: high divergence = resonant tone
    buf = np.zeros(n_samples)
    for i in range(len(divergence)):
        if divergence[i] > 0.15:  # significant color separation
            t_pos = int(i / len(divergence) * n_samples)
            # Resonant frequency from the row position (higher rows = higher freq)
            freq = 100 + (i / len(divergence)) * 4000
            ring_len = min(int(sr * 0.01 * divergence[i] * 5), n_samples - t_pos)
            if ring_len > 1:
                t = np.arange(ring_len) / sr
                ring = np.sin(2 * np.pi * freq * t) * np.exp(-t * 200) * divergence[i] * 0.3
                buf[t_pos:t_pos + ring_len] += ring
    
    return buf


def synthesize_frame_audio(frame, sr, n_samples, phases, stripe_emphasis=0.5,
                           transient_gain=0.5, spectro_gain=0.5):
    """
    THE CORE: Convert one video frame directly into audio.
    No preset sounds. The image IS the instrument.
    
    Returns (audio_L, audio_R, updated_phases)
    """
    h, w = frame.shape[:2]
    
    # Split channels
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 1: SPECTROGRAM SONIFICATION
    # Each channel treated as a spectrogram → additive synthesis
    # ═══════════════════════════════════════════════════════════════════════
    
    # Red → Bass spectrogram
    bass_bands = frame_to_spectrogram_bands(red, N_BANDS_BASS)  # [n_bands, w]
    # Green → Mid spectrogram
    mid_bands = frame_to_spectrogram_bands(green, N_BANDS_MID)
    # Blue → Treble spectrogram
    treble_bands = frame_to_spectrogram_bands(blue, N_BANDS_TREBLE)
    
    # Synthesize via additive oscillators with time-varying amplitude
    audio_L = np.zeros(n_samples)
    audio_R = np.zeros(n_samples)
    
    total_bands = N_BANDS_BASS + N_BANDS_MID + N_BANDS_TREBLE
    all_freqs = np.concatenate([BASS_FREQS, MID_FREQS, TREBLE_FREQS])
    all_bands = np.vstack([bass_bands, mid_bands, treble_bands])  # [total, w]
    all_gains = np.concatenate([
        np.ones(N_BANDS_BASS) * 0.5,    # bass louder
        np.ones(N_BANDS_MID) * 0.25,     # mid moderate
        np.ones(N_BANDS_TREBLE) * 0.15,  # treble quieter
    ]) * spectro_gain
    
    for bi in range(total_bands):
        freq = all_freqs[bi]
        if freq > sr / 2:
            continue
        
        gain = all_gains[bi]
        amp_profile = all_bands[bi, :]  # amplitude over time (from image columns)
        
        # Skip silent bands
        if np.max(amp_profile) < 0.02:
            continue
        
        # Resample amplitude profile to audio length (smooth)
        amp_env = np.interp(
            np.linspace(0, 1, n_samples),
            np.linspace(0, 1, len(amp_profile)),
            amp_profile
        )
        
        # Generate oscillator with continuous phase
        phase_inc = 2 * np.pi * freq / sr
        phase_arr = phases[bi] + np.cumsum(np.ones(n_samples) * phase_inc)
        osc = np.sin(phase_arr)
        phases[bi] = phase_arr[-1] % (2 * np.pi)
        
        # Apply amplitude envelope from image
        signal_out = osc * amp_env * gain
        
        # Stereo: bands from left side of image → left channel, etc.
        # Use the column-weighted average position for panning
        col_weights = amp_profile / (np.sum(amp_profile) + 1e-10)
        col_positions = np.linspace(0, 1, len(amp_profile))
        pan = float(np.dot(col_weights, col_positions))  # 0=left, 1=right
        
        audio_L += signal_out * (1.0 - pan * 0.6)
        audio_R += signal_out * (0.4 + pan * 0.6)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 2: STRIPE TONES
    # FFT of columns to find horizontal stripe frequencies → pitched tones
    # ═══════════════════════════════════════════════════════════════════════
    
    stripe_audio = np.zeros(n_samples)
    
    for ch, ch_data, base_oct in [(red, 'r', 55), (green, 'g', 110), (blue, 'b', 220)]:
        peaks, _, _ = stripe_detect(ch)
        for sf_val, amp in peaks:
            if sf_val < 0.01:
                continue
            audio_hz = spatial_freq_to_audio_hz(sf_val, base_freq=base_oct)
            if audio_hz > sr / 2:
                continue
            t = np.arange(n_samples) / sr
            tone = np.sin(2 * np.pi * audio_hz * t) * amp * stripe_emphasis * 0.15
            stripe_audio += tone
    
    audio_L += stripe_audio * 0.6
    audio_R += stripe_audio * 0.4
    
    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 3: TRANSITION IMPULSES (percussion from brightness gradient)
    # ═══════════════════════════════════════════════════════════════════════
    
    # From grayscale (main transients)
    impulses = transitions_to_impulses(gray, sr, n_samples) * transient_gain
    
    # Red channel transitions → low thump
    red_imp = transitions_to_impulses(red, sr, n_samples)
    # Make red impulses "thumpy" by lowpass filtering
    if n_samples > 100:
        sos = signal.butter(2, min(200, sr/2 - 100), btype='low', fs=sr, output='sos')
        red_imp_lp = signal.sosfilt(sos, red_imp) * transient_gain * 2.0
    else:
        red_imp_lp = red_imp * transient_gain
    
    # Blue channel transitions → high click
    blue_imp = transitions_to_impulses(blue, sr, n_samples)
    if n_samples > 100:
        sos = signal.butter(2, max(2000, min(8000, sr/2 - 100)), btype='high', fs=sr, output='sos')
        blue_imp_hp = signal.sosfilt(sos, blue_imp) * transient_gain * 0.5
    else:
        blue_imp_hp = blue_imp * transient_gain * 0.5
    
    audio_L += impulses + red_imp_lp + blue_imp_hp * 0.7
    audio_R += impulses + red_imp_lp + blue_imp_hp * 1.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # LAYER 4: COLOR EDGE RESONANCE (glitch boundaries)
    # ═══════════════════════════════════════════════════════════════════════
    
    glitch_reso = color_edge_resonance(frame, sr, n_samples)
    audio_L += glitch_reso * 0.5
    audio_R += glitch_reso * 0.5
    
    return audio_L, audio_R, phases


# ─── Low-level audio helpers ─────────────────────────────────────────────────

def apply_filter(audio, cutoff_hz, btype='low', sr=SAMPLE_RATE):
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype=btype, fs=sr, output='sos')
    return signal.sosfilt(sos, audio)


def comb_reverb(audio, sr=SAMPLE_RATE, decay=0.3, delays_ms=(23, 37, 49)):
    out = audio.copy()
    for d_ms in delays_ms:
        d = int(d_ms * sr / 1000)
        if d < len(audio):
            delayed = np.zeros_like(audio)
            delayed[d:] = audio[:-d]
            out += delayed * decay * (1.0 / len(delays_ms))
    return out


def place(buf, sound, pos):
    end = min(pos + len(sound), len(buf))
    length = end - pos
    if length > 0 and pos >= 0:
        buf[pos:end] += sound[:length]


# ─── Video Analysis ──────────────────────────────────────────────────────────

def analyze_video_stats(video_path):
    """Quick pass to get video metadata and global statistics."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample some frames for global stats
    sample_indices = np.linspace(0, total_frames - 1, min(50, total_frames)).astype(int)
    stripe_counts = []
    avg_brightness = []
    avg_divergence = []
    
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        avg_brightness.append(float(np.mean(gray)) / 255.0)
        
        # Count stripe peaks
        peaks, _, _ = stripe_detect(gray)
        stripe_counts.append(len(peaks))
        
        # RGB divergence
        b, g, r = small[:,:,0].astype(float), small[:,:,1].astype(float), small[:,:,2].astype(float)
        div = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / (3*255)
        avg_divergence.append(float(div))
    
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'width': w,
        'height': h,
        'avg_brightness': float(np.mean(avg_brightness)) if avg_brightness else 0.5,
        'avg_stripes': float(np.mean(stripe_counts)) if stripe_counts else 0,
        'max_stripes': int(np.max(stripe_counts)) if stripe_counts else 0,
        'avg_divergence': float(np.mean(avg_divergence)) if avg_divergence else 0,
    }


# ─── Main Render Engine ──────────────────────────────────────────────────────

def render(video_path, bpm=174, output_path=None, brightness_gain=0.7,
           transient_gain=0.6, stripe_emphasis=0.5, stems=False,
           sr=SAMPLE_RATE):

    print(f"""
============================================================
  VIDEO2DNB v8 — STRIPE SONIFICATION ENGINE
  Video: {video_path}
  BPM:   {bpm}
  SR:    {sr}
============================================================
""")

    # ─── Quick analysis ───────────────────────────────────────────────────
    print("[1/4] Analyzing video...")
    stats = analyze_video_stats(video_path)
    
    total_frames = stats['total_frames']
    fps = stats['fps']
    duration = stats['duration']
    total_samples = int(duration * sr)
    
    beat_dur = 60.0 / bpm
    s16 = beat_dur / 4  # 16th note duration in seconds
    s16_samples = int(s16 * sr)
    
    # Process resolution — balance quality vs speed
    proc_w, proc_h = 160, 120
    
    print(f"""
  ──────────────────────────────────────────────────
  VIDEO INFO
  ──────────────────────────────────────────────────
  Frames:      {total_frames} at {fps:.1f} fps
  Duration:    {duration:.1f}s
  Resolution:  {stats['width']}x{stats['height']} → {proc_w}x{proc_h}
  ──────────────────────────────────────────────────
  DETECTED FEATURES
  ──────────────────────────────────────────────────
  Avg brightness:    {stats['avg_brightness']:.3f}
  Avg stripe peaks:  {stats['avg_stripes']:.1f}
  Max stripe peaks:  {stats['max_stripes']}
  RGB divergence:    {stats['avg_divergence']:.3f}
  ──────────────────────────────────────────────────
  SONIFICATION
  ──────────────────────────────────────────────────
  Spectro gain:  {brightness_gain:.2f}
  Transient:     {transient_gain:.2f}
  Stripe:        {stripe_emphasis:.2f}
  ──────────────────────────────────────────────────
  The image IS the synthesizer.
  Stripes → tones. Rows → frequencies. RGB → bands.
  Transitions → transients. No preset sounds.
  ──────────────────────────────────────────────────
""")

    # ─── Render ───────────────────────────────────────────────────────────
    print("[2/4] Rendering (image → audio)...")
    
    # Allocate output buffers
    mix_L = np.zeros(total_samples + sr)  # +1s tail
    mix_R = np.zeros(total_samples + sr)
    
    # For stems
    if stems:
        spectro_L = np.zeros(total_samples + sr)
        spectro_R = np.zeros(total_samples + sr)
        stripe_stem = np.zeros(total_samples + sr)
        transient_stem = np.zeros(total_samples + sr)
        glitch_stem = np.zeros(total_samples + sr)
    
    # Phase accumulators for continuous oscillators
    total_bands = N_BANDS_BASS + N_BANDS_MID + N_BANDS_TREBLE
    phases = np.zeros(total_bands)
    
    # Open video for frame-by-frame processing
    cap = cv2.VideoCapture(video_path)
    
    # Process in chunks aligned to beat grid
    # Each frame gets one "grain" of audio
    samples_per_frame = int(sr / fps) if fps > 0 else int(sr / 30)
    
    # We quantize to 16th notes: group frames into 16th-note windows
    # Each 16th note window picks the frame nearest to it
    frame_idx = 0
    prev_frame = None
    frame_cache = {}
    
    # Read all frames into a lightweight cache (resized)
    print("  Reading frames...")
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (proc_w, proc_h))
        all_frames.append(small)
    cap.release()
    num_frames = len(all_frames)
    print(f"  {num_frames} frames loaded")
    
    # ─── MAIN SYNTHESIS LOOP: frame by frame ──────────────────────────────
    print("  Synthesizing...")
    
    for fi in range(num_frames):
        frame = all_frames[fi]
        pos = int(fi * samples_per_frame)
        if pos >= total_samples:
            break
        
        grain_len = min(samples_per_frame, total_samples - pos)
        if grain_len < 64:
            continue
        
        # THE CORE: frame → audio
        audio_L, audio_R, phases = synthesize_frame_audio(
            frame, sr, grain_len, phases,
            stripe_emphasis=stripe_emphasis,
            transient_gain=transient_gain,
            spectro_gain=brightness_gain,
        )
        
        # Cross-fade grains to avoid clicks (32 sample overlap)
        fade = min(32, grain_len // 4)
        if fade > 1:
            audio_L[:fade] *= np.linspace(0, 1, fade)
            audio_L[-fade:] *= np.linspace(1, 0, fade)
            audio_R[:fade] *= np.linspace(0, 1, fade)
            audio_R[-fade:] *= np.linspace(1, 0, fade)
        
        mix_L[pos:pos + grain_len] += audio_L
        mix_R[pos:pos + grain_len] += audio_R
        
        if fi % 100 == 0:
            pct = fi / num_frames * 100
            print(f"\r  Synthesizing [{'#' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)
    
    print(f"\r  Synthesizing [{'#' * 40}] 100%")
    
    # ─── Add beat-grid-aligned sub pulse from red channel ─────────────────
    # Not a preset kick — the amplitude and pitch come from the frame's
    # red channel brightness at that moment
    print("\n[3/4] Adding beat-aligned sub pulse...")
    
    pos = 0
    step = 0
    while pos < total_samples:
        # Get frame at this position
        fi = min(int(pos / sr * fps), num_frames - 1)
        frame = all_frames[fi]
        red = frame[:, :, 2].astype(float) / 255.0
        
        # Only pulse on strong beats (1 and 3 of each bar = steps 0, 8)
        # And sometimes on step 4/12 if red is bright
        step_in_bar = step % 16
        red_brightness = float(np.mean(red))
        
        do_pulse = False
        if step_in_bar == 0:
            do_pulse = True
        elif step_in_bar == 8 and red_brightness > 0.3:
            do_pulse = True
        elif step_in_bar in (4, 12) and red_brightness > 0.6:
            do_pulse = True
        elif step_in_bar in (6, 10, 14) and red_brightness > 0.8:
            do_pulse = True
        
        if do_pulse:
            # Sub pulse: pitch from red intensity (not a preset!)
            sub_freq = 30 + red_brightness * 30  # 30-60 Hz
            pulse_len = min(int(sr * 0.08), total_samples - pos)
            if pulse_len > 10:
                t = np.arange(pulse_len) / sr
                # Pitch sweep from image brightness
                sweep = sub_freq * (1 + red_brightness * np.exp(-t * 20))
                phase = np.cumsum(2 * np.pi * sweep / sr)
                sub_pulse = np.sin(phase) * np.exp(-t * (12 + red_brightness * 8))
                sub_pulse *= red_brightness * 0.6
                # Saturate slightly
                sub_pulse = np.tanh(sub_pulse * 2.0) * 0.5
                place(mix_L, sub_pulse, pos)
                place(mix_R, sub_pulse, pos)
        
        # High-frequency transient on beat from blue channel  
        blue_brightness = float(np.mean(frame[:, :, 0].astype(float) / 255.0))
        if step_in_bar in (4, 12):  # "snare" positions
            green_brightness = float(np.mean(frame[:, :, 1].astype(float) / 255.0))
            if green_brightness > 0.2:
                snap_len = min(int(sr * 0.02), total_samples - pos)
                if snap_len > 10:
                    t = np.arange(snap_len) / sr
                    # Noise burst shaped by green channel
                    np.random.seed(int(pos) % (2**31))
                    noise = np.random.randn(snap_len)
                    snap = noise * np.exp(-t * (50 + green_brightness * 100))
                    snap *= green_brightness * 0.3
                    # Bandpass from frame brightness
                    if snap_len > 20:
                        center = 2000 + blue_brightness * 6000
                        lo = max(20, center - 2000)
                        hi = min(sr/2 - 100, center + 2000)
                        if hi > lo + 100:
                            sos = signal.butter(2, [lo, hi], btype='band', fs=sr, output='sos')
                            snap = signal.sosfilt(sos, snap)
                    place(mix_L, snap * 0.8, pos)
                    place(mix_R, snap, pos)
        
        # Hat-like from blue channel on 8ths/16ths
        if blue_brightness > 0.25:
            hat_prob = blue_brightness * 0.4
            np.random.seed((int(pos) + 7) % (2**31))
            if np.random.random() < hat_prob:
                hat_len = min(int(sr * (0.005 + blue_brightness * 0.015)), total_samples - pos)
                if hat_len > 5:
                    t = np.arange(hat_len) / sr
                    np.random.seed((int(pos) + 13) % (2**31))
                    hat_noise = np.random.randn(hat_len)
                    hat = hat_noise * np.exp(-t * (80 + blue_brightness * 120))
                    hat *= blue_brightness * 0.12
                    if hat_len > 20:
                        sos = signal.butter(2, min(6000, sr/2-100), btype='high', fs=sr, output='sos')
                        hat = signal.sosfilt(sos, hat)
                    place(mix_L, hat * 0.7, pos)
                    place(mix_R, hat, pos)
        
        step += 1
        pos += s16_samples
    
    # ─── Post-processing ──────────────────────────────────────────────────
    print("\n[4/4] Mastering...")
    
    # Trim to final length
    mix_L = mix_L[:total_samples]
    mix_R = mix_R[:total_samples]
    
    # Gentle highpass to remove DC
    mix_L = apply_filter(mix_L, 25, 'high', sr)
    mix_R = apply_filter(mix_R, 25, 'high', sr)
    
    # Add reverb to high frequencies only
    mix_L_hi = apply_filter(mix_L, 2000, 'high', sr)
    mix_R_hi = apply_filter(mix_R, 2000, 'high', sr)
    mix_L_hi_verb = comb_reverb(mix_L_hi, sr, decay=0.2, delays_ms=(43, 59, 73))
    mix_R_hi_verb = comb_reverb(mix_R_hi, sr, decay=0.2, delays_ms=(59, 73, 89))
    mix_L += mix_L_hi_verb * 0.15
    mix_R += mix_R_hi_verb * 0.15
    
    # Stereo mix
    stereo = np.column_stack([mix_L, mix_R])
    
    # Normalize
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.95
    
    # Soft compression
    threshold = 0.5
    ratio = 3.0
    for ch in range(2):
        above = np.abs(stereo[:, ch]) > threshold
        stereo[above, ch] = np.sign(stereo[above, ch]) * (
            threshold + (np.abs(stereo[above, ch]) - threshold) / ratio
        )
    
    # Final limiter
    peak = np.max(np.abs(stereo))
    if peak > 0.98:
        stereo = stereo / peak * 0.96
    
    # Sanitize
    stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ─── Output ───────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_v8_stripe.wav"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    sf.write(output_path, stereo, sr, subtype='PCM_24')
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output: {output_path} ({size_mb:.1f} MB)")
    
    # ─── Stems ────────────────────────────────────────────────────────────
    if stems:
        stem_dir = os.path.splitext(output_path)[0] + '_stems'
        os.makedirs(stem_dir, exist_ok=True)
        # Re-render with isolated layers would go here
        print(f"  (Stems not yet implemented for v8 — all layers are interleaved)")
    
    print(f"""
============================================================
  Output:     {output_path}
  Duration:   {duration:.1f}s
  Format:     {sr}Hz / 24-bit / Stereo
  ──────────────────────────────────────────────────
  THE IMAGE IS THE SYNTHESIZER.
  Stripes → tones | RGB → bands | Transitions → transients
  No preset kick. No preset snare. Pure pixel sound.
============================================================""")
    
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='video2dnb v8 — Stripe Sonification Engine. The image IS the synthesizer.')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('--bpm', type=int, default=174, help='BPM')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output WAV path')
    parser.add_argument('--brightness', type=float, default=0.7,
                        help='Spectrogram sonification gain (0-2)')
    parser.add_argument('--transient', type=float, default=0.6,
                        help='Transition/impulse gain (0-2)')
    parser.add_argument('--stripe', type=float, default=0.5,
                        help='Stripe detection tone emphasis (0-2)')
    parser.add_argument('--stems', action='store_true',
                        help='Export stems (experimental)')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()

    t0 = time.time()
    render(
        video_path=args.video,
        bpm=args.bpm,
        output_path=args.output,
        brightness_gain=args.brightness,
        transient_gain=args.transient,
        stripe_emphasis=args.stripe,
        stems=args.stems,
        sr=args.sample_rate,
    )
    print(f"  Render time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
