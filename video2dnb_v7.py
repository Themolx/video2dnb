#!/usr/bin/env python3
"""
video2dnb_v7.py — SCANLINE GLITCH RENDERER

Middle ground between v1 (hardcoded patterns) and v6 (pure chaos).
Patterns are IMAGE-DERIVED via scanline analysis, placed on a BPM grid.

Design:
  SCANLINE RHYTHM:
  - Each frame is resized to 16 columns (= 16 steps in a bar)
  - Column brightness profile → kick pattern (bright columns = hits)
  - Column edge density → snare pattern
  - Column blue channel → hat pattern
  - Column hue variance → tonal triggers
  - This creates IMAGE-DERIVED patterns on a MUSICAL GRID

  HUE-DRIVEN HARMONY (glitch art focus):
  - Dominant hue → musical key (12 keys from 360°)
  - Hue spread → chord complexity (narrow = simple, wide = complex)
  - Hue shift frame-to-frame → melodic movement
  - Saturation → filter resonance / harmonic richness
  - Glitch artifacts (sudden hue jumps) → musical events

  VISUAL PATTERN DETECTION:
  - Autocorrelation of column brightness → repeating pattern detection
  - Repeating visual patterns → loop/phrase repetition
  - Pattern strength → how strictly the rhythm follows the image

  LOOSE GRID:
  - BPM-locked 16th note positions
  - Micro-timing swing derived from brightness gradient across columns
  - Image-modulated synthesis (same as v6 — no fixed sounds)

Usage:
    python video2dnb_v7.py <video_path> [--bpm 174] [--output out.wav]
    python video2dnb_v7.py video.mp4 --chaos 0.5   # 0=strict grid, 1=loose
    python video2dnb_v7.py video.mp4 --density 0.6  # event density scale
    python video2dnb_v7.py video.mp4 --stems        # export stems
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

# Chromatic frequencies (semitone index → Hz)
CHROMATIC = {}
for i in range(48):  # C1 to C5 — 48 semitones
    CHROMATIC[i] = 32.70 * (2.0 ** (i / 12.0))

# HUE → musical key mapping (12 keys around the color wheel)
# Each key = (root_semitone_offset, scale_intervals, name)
# Hue 0°=Red, 30°=Orange, 60°=Yellow, 120°=Green, 180°=Cyan, 240°=Blue, 300°=Magenta
HUE_KEYS = {
    0:  (0,  [0,2,3,5,7,8,10], "Cm"),    # Red → C minor
    1:  (2,  [0,2,3,5,7,8,10], "Dm"),     # Orange-Red → D minor
    2:  (4,  [0,2,4,5,7,9,11], "E"),      # Orange → E major
    3:  (5,  [0,2,3,5,7,8,10], "Fm"),     # Yellow-Orange → F minor
    4:  (7,  [0,2,3,5,7,8,10], "Gm"),     # Yellow → G minor
    5:  (9,  [0,2,3,5,7,8,10], "Am"),     # Yellow-Green → A minor
    6:  (10, [0,2,3,5,7,8,10], "Bbm"),    # Green → Bb minor
    7:  (0,  [0,2,4,5,7,9,11], "C"),      # Cyan-Green → C major
    8:  (3,  [0,2,3,5,7,8,10], "Ebm"),    # Cyan → Eb minor
    9:  (5,  [0,2,4,5,7,9,11], "F"),      # Blue-Cyan → F major
    10: (8,  [0,2,3,5,7,8,10], "Abm"),    # Blue → Ab minor
    11: (10, [0,2,4,5,7,9,11], "Bb"),     # Magenta → Bb major
}

# Scanline thresholds for pattern extraction
SCAN_KICK_THRESH = 0.55    # column brightness > this → kick candidate
SCAN_SNARE_THRESH = 0.12   # column edge density > this → snare candidate
SCAN_HAT_THRESH = 0.35     # column blue > this → hat candidate


# ─── Synthesis Primitives ─────────────────────────────────────────────────────

def synth_kick(sr, brightness=0.5, motion=0.5):
    """Kick whose character varies with frame data. No fixed sound."""
    dur = 0.06 + brightness * 0.12  # brighter frames = longer kick
    n = int(sr * dur); t = np.arange(n) / sr
    # Pitch sweep range from motion
    hi_pitch = 80 + motion * 200
    lo_pitch = 30 + brightness * 20
    pitch = hi_pitch * np.exp(-t * (20 + motion * 40)) + lo_pitch
    phase = np.cumsum(2 * np.pi * pitch / sr)
    body = np.sin(phase) * np.exp(-t * (8 + motion * 15))
    click = np.exp(-t * (100 + brightness * 200)) * (0.3 + motion * 0.5)
    return np.tanh((body + click) * (1.5 + brightness)) * 0.85


def synth_snare(sr, edge_density=0.5, contrast=0.5):
    """Snare shaped by edge density and contrast."""
    dur = 0.04 + edge_density * 0.12
    n = int(sr * dur); t = np.arange(n) / sr
    noise = np.random.randn(n)
    # More edges = brighter snare
    lo_cut = 1000 + edge_density * 4000
    hi_cut = min(lo_cut + 3000 + contrast * 5000, sr / 2 - 100)
    lo_cut = min(lo_cut, hi_cut - 100)
    sos = signal.butter(2, [max(20, lo_cut), hi_cut], btype='band', fs=sr, output='sos')
    noise_f = signal.sosfilt(sos, noise) * np.exp(-t * (15 + contrast * 20))
    # Tone body
    tone_freq = 150 + contrast * 150
    tone = np.sin(2 * np.pi * tone_freq * t) * np.exp(-t * (20 + edge_density * 30))
    return np.tanh((noise_f * 0.7 + tone * 0.4) * 2.0) * 0.7


def synth_hat(sr, blue=0.5, saturation=0.5):
    """Hat whose brightness comes from blue channel."""
    dur = 0.02 + blue * 0.08
    n = int(sr * dur); t = np.arange(n) / sr
    noise = np.random.randn(n)
    decay = 20 + (1 - blue) * 60
    env = np.exp(-t * decay)
    # Cutoff from saturation
    cutoff = max(4000 + saturation * 8000, 4100)
    cutoff = min(cutoff, sr / 2 - 100)
    sos = signal.butter(2, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, noise) * env * (0.2 + blue * 0.3)


def synth_drone(sr, freq, n_samples, red=0.5, saturation=0.5):
    """Low drone — pitch from red, harmonics from saturation."""
    t = np.arange(n_samples) / sr
    # Fundamental
    wave = np.sin(2 * np.pi * freq * t)
    # Add harmonics based on saturation
    n_harm = int(1 + saturation * 8)
    for k in range(2, n_harm + 2):
        if k * freq > sr / 2:
            break
        wave += np.sin(2 * np.pi * k * freq * t) / (k ** (1.5 - saturation * 0.5))
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak
    return wave * (0.2 + red * 0.4)


def synth_texture_grain(sr, freqs, amps, n_samples, phases):
    """Additive grain from image row brightness → oscillator bank."""
    grain = np.zeros(n_samples)
    for i in range(len(freqs)):
        if amps[i] < 0.03 or freqs[i] > sr / 2:
            continue
        ph_inc = 2 * np.pi * freqs[i] / sr
        ph = phases[i] + np.cumsum(np.ones(n_samples) * ph_inc)
        grain += np.sin(ph) * amps[i]
        phases[i] = ph[-1] % (2 * np.pi)
    return grain, phases


def synth_tone(sr, freq, n_samples, brightness=0.5, waveshape='sine'):
    """Single tone with variable waveshape."""
    t = np.arange(n_samples) / sr
    if waveshape == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif waveshape == 'saw':
        wave = 2.0 * (t * freq - np.floor(t * freq + 0.5))
    elif waveshape == 'tri':
        wave = 2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0
    elif waveshape == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        wave = np.sin(2 * np.pi * freq * t)
    return wave


def apply_lp(audio, cutoff_hz, sr=SAMPLE_RATE):
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype='low', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)


def apply_hp(audio, cutoff_hz, sr=SAMPLE_RATE):
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)


def comb_reverb(audio, sr=SAMPLE_RATE, decay=0.3, delays_ms=(23, 37, 49, 61)):
    out = audio.copy()
    for d_ms in delays_ms:
        d = int(d_ms * sr / 1000)
        if d < len(audio):
            delayed = np.zeros_like(audio)
            delayed[d:] = audio[:-d]
            out += delayed * decay * 0.25
    return out


def place(buf, sound, pos):
    end = min(pos + len(sound), len(buf))
    length = end - pos
    if length > 0 and pos >= 0:
        buf[pos:end] += sound[:length]


# ─── Frame Analysis ──────────────────────────────────────────────────────────

def scanline_pattern(frame, gray, hsv):
    """
    SCANLINE: Read frame as 16 columns → 16-step rhythm pattern.
    Each column's statistics determine whether kick/snare/hat fires at that step.
    Returns dict with per-step arrays and pattern metadata.
    """
    h, w = gray.shape
    n_steps = 16

    # Resize to exactly 16 columns for clean scanline mapping
    gray_16 = cv2.resize(gray, (n_steps, h))
    frame_16 = cv2.resize(frame, (n_steps, h))
    hsv_16 = cv2.resize(hsv, (n_steps, h))

    # Per-column statistics (each column = one 16th note step)
    col_brightness = np.mean(gray_16, axis=0) / 255.0          # [16]
    col_blue = np.mean(frame_16[:, :, 0], axis=0) / 255.0      # [16]
    col_red = np.mean(frame_16[:, :, 2], axis=0) / 255.0       # [16]
    col_green = np.mean(frame_16[:, :, 1], axis=0) / 255.0     # [16]
    col_hue = np.mean(hsv_16[:, :, 0], axis=0) / 180.0         # [16] 0-1
    col_sat = np.mean(hsv_16[:, :, 1], axis=0) / 255.0         # [16]

    # Per-column edge density (scanline edges)
    edges_16 = cv2.Canny(gray_16, 50, 150)
    col_edges = np.sum(edges_16 > 0, axis=0) / float(h)        # [16]

    # Per-column brightness gradient (for swing/timing offset)
    col_gradient = np.gradient(col_brightness)                  # [16]

    # Hue variance per column (glitch detection — sudden color changes)
    col_hue_var = np.var(hsv_16[:, :, 0].astype(float), axis=0) / (180.0**2)  # [16]

    # ── PATTERN EXTRACTION via adaptive thresholds ──
    # Use median as threshold so ~half the steps trigger (musical density)
    b_med = float(np.median(col_brightness))
    e_med = float(np.median(col_edges))
    bl_med = float(np.median(col_blue))

    kick_pattern = (col_brightness > max(b_med, SCAN_KICK_THRESH * 0.8)).astype(float)
    snare_pattern = (col_edges > max(e_med, SCAN_SNARE_THRESH * 0.5)).astype(float)
    hat_pattern = (col_blue > max(bl_med * 0.9, SCAN_HAT_THRESH * 0.7)).astype(float)

    # Weight patterns by how far above threshold (velocity)
    kick_vel = np.clip((col_brightness - b_med * 0.8) * 2, 0, 1) * kick_pattern
    snare_vel = np.clip((col_edges - e_med * 0.5) * 3, 0, 1) * snare_pattern
    hat_vel = np.clip((col_blue - bl_med * 0.7) * 2.5, 0, 1) * hat_pattern

    # ── VISUAL REPETITION DETECTION via autocorrelation ──
    # Check if column brightness has repeating pattern (stripes, grids)
    if len(col_brightness) > 4:
        centered = col_brightness - np.mean(col_brightness)
        acorr = np.correlate(centered, centered, mode='full')
        acorr = acorr[len(acorr)//2:]  # positive lags only
        if acorr[0] > 0:
            acorr_norm = acorr / acorr[0]
        else:
            acorr_norm = np.zeros_like(acorr)
        # Find strongest repetition period (skip lag 0)
        if len(acorr_norm) > 2:
            peaks = []
            for lag in range(2, len(acorr_norm)):
                if acorr_norm[lag] > 0.3:  # significant repetition
                    peaks.append((lag, float(acorr_norm[lag])))
            if peaks:
                best_lag, rep_strength = max(peaks, key=lambda x: x[1])
            else:
                best_lag, rep_strength = 0, 0.0
        else:
            best_lag, rep_strength = 0, 0.0
    else:
        best_lag, rep_strength = 0, 0.0

    return {
        'kick_pattern': kick_pattern,
        'snare_pattern': snare_pattern,
        'hat_pattern': hat_pattern,
        'kick_vel': kick_vel,
        'snare_vel': snare_vel,
        'hat_vel': hat_vel,
        'col_brightness': col_brightness,
        'col_hue': col_hue,
        'col_sat': col_sat,
        'col_hue_var': col_hue_var,
        'col_gradient': col_gradient,
        'col_red': col_red,
        'col_green': col_green,
        'col_blue': col_blue,
        'col_edges': col_edges,
        'rep_period': best_lag,
        'rep_strength': float(rep_strength),
    }


def hue_to_key(hue_01):
    """Map normalized hue (0-1) to musical key info."""
    idx = int(hue_01 * 12) % 12
    root, intervals, name = HUE_KEYS[idx]
    return root, intervals, name


def scale_freq(root, intervals, degree, octave=2):
    """Get frequency for a scale degree in given key and octave."""
    semitone = root + intervals[degree % len(intervals)] + 12 * octave
    return CHROMATIC.get(semitone, 65.41)


def analyze_frame(frame, prev_frame=None):
    """Frame → feature dict + scanline pattern. Glitch-art focused."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = {}

    r['brightness'] = float(np.mean(gray)) / 255.0
    r['contrast'] = min(float(np.std(gray)) / 128.0, 1.0)
    r['red'] = float(np.mean(frame[:, :, 2])) / 255.0
    r['green'] = float(np.mean(frame[:, :, 1])) / 255.0
    r['blue'] = float(np.mean(frame[:, :, 0])) / 255.0
    r['saturation'] = float(np.mean(hsv[:, :, 1])) / 255.0

    # HUE analysis (critical for glitch art)
    r['hue'] = float(np.mean(hsv[:, :, 0])) / 180.0  # 0-1
    r['hue_std'] = float(np.std(hsv[:, :, 0].astype(float))) / 180.0  # hue spread
    sat_mask = hsv[:, :, 1] > 20
    if np.any(sat_mask):
        r['hue_dominant'] = float(np.median(hsv[:, :, 0][sat_mask])) / 180.0
    else:
        r['hue_dominant'] = r['hue']

    edges = cv2.Canny(gray, 50, 150)
    r['edge_density'] = float(np.sum(edges > 0)) / float(edges.size)

    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    r['vert_edges'] = min(float(np.mean(np.abs(sobel_v))) / 128.0, 1.0)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    r['horiz_edges'] = min(float(np.mean(np.abs(sobel_h))) / 128.0, 1.0)

    warm = r['red'] + r['green'] * 0.3
    cool = r['blue'] + r['green'] * 0.3
    total = warm + cool
    r['warmth'] = warm / total if total > 0 else 0.5

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        r['motion'] = float(np.mean(diff)) / 255.0
    else:
        r['motion'] = 0.0

    # Spectral: image rows → frequency bands
    h, w = gray.shape
    n_bands = 48
    indices = np.linspace(0, h - 1, n_bands).astype(int)
    bands = gray[indices, :]
    amps = np.mean(bands, axis=1) / 255.0
    amps = amps[::-1]
    freqs = np.logspace(np.log10(60), np.log10(10000), n_bands)
    r['spectral_freqs'] = freqs
    r['spectral_amps'] = amps

    # Pixel hash for deterministic randomness
    pixel_sample = gray[::10, ::10].flatten()
    r['pixel_hash'] = int(np.sum(pixel_sample.astype(np.int64))) % (2**31)

    # SCANLINE pattern extraction
    r['scan'] = scanline_pattern(frame, gray, hsv)

    return r


def analyze_video(video_path):
    """Analyze all frames. Returns (analyses, fps, duration, total_frames, stats)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    analyses = []
    prev_small = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (160, 120))
        a = analyze_frame(small, prev_small)
        analyses.append(a)
        prev_small = small
        idx += 1
        if idx % 30 == 0:
            pct = idx / total_frames * 100
            print(f"\r  Analyzing [{'#' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)

    cap.release()
    print(f"\r  Analyzing [{'#' * 40}] 100%")
    print(f"  {len(analyses)} frames analyzed")

    # Compute deltas
    for i in range(len(analyses)):
        if i > 0:
            analyses[i]['brightness_delta'] = analyses[i]['brightness'] - analyses[i-1]['brightness']
            analyses[i]['edge_delta'] = analyses[i]['edge_density'] - analyses[i-1]['edge_density']
            analyses[i]['motion_delta'] = analyses[i]['motion'] - analyses[i-1]['motion']
            analyses[i]['hue_delta'] = abs(analyses[i]['hue'] - analyses[i-1]['hue'])
            analyses[i]['hue_dom_delta'] = abs(analyses[i]['hue_dominant'] - analyses[i-1]['hue_dominant'])
        else:
            analyses[i]['brightness_delta'] = 0.0
            analyses[i]['edge_delta'] = 0.0
            analyses[i]['motion_delta'] = 0.0
            analyses[i]['hue_delta'] = 0.0
            analyses[i]['hue_dom_delta'] = 0.0

    # Compute stats
    brights = [a['brightness'] for a in analyses]
    edges = [a['edge_density'] for a in analyses]
    motions = [a['motion'] for a in analyses]
    hues = [a['hue_dominant'] for a in analyses]
    hue_stds = [a['hue_std'] for a in analyses]
    reps = [a['scan']['rep_strength'] for a in analyses]

    stats = {
        'brightness_mean': float(np.mean(brights)),
        'brightness_std': float(np.std(brights)),
        'edge_mean': float(np.mean(edges)),
        'edge_std': float(np.std(edges)),
        'motion_mean': float(np.mean(motions)),
        'motion_std': float(np.std(motions)),
        'hue_mean': float(np.mean(hues)),
        'hue_std_mean': float(np.mean(hue_stds)),
        'rep_strength_mean': float(np.mean(reps)),
        'rep_strength_max': float(np.max(reps)) if reps else 0.0,
    }

    # Determine global key from dominant hue
    root, intervals, key_name = hue_to_key(stats['hue_mean'])
    stats['key_root'] = root
    stats['key_intervals'] = intervals
    stats['key_name'] = key_name

    return analyses, fps, duration, total_frames, stats


# ─── Main Render Engine ──────────────────────────────────────────────────────

def render(video_path, bpm=174, output_path=None, chaos=0.5, density=0.5,
           sonification_gain=0.35, no_sonification=False, stems=False,
           sr=SAMPLE_RATE):

    print(f"""
============================================================
  VIDEO2DNB v7 — SCANLINE GLITCH RENDERER
  Video: {video_path}
  BPM:   {bpm}
  SR:    {sr}
  Chaos: {chaos:.2f}
  Density: {density:.2f}
  Stems: {'YES' if stems else 'NO'}
  Sonification: {'OFF' if no_sonification else f'ON (gain={sonification_gain})'}
============================================================
""")

    # ─── Analyze ──────────────────────────────────────────────────────────
    print("[1/6] Analyzing video...")
    analyses, fps, duration, total_frames, stats = analyze_video(video_path)

    total_samples = int(duration * sr)
    tail = sr
    buf_len = total_samples + tail
    num_frames = len(analyses)

    beat_dur = 60.0 / bpm
    s16 = int(beat_dur / 4 * sr)  # samples per 16th note
    s16_sec = beat_dur / 4
    total_bars = int(duration / (beat_dur * 4))
    samples_per_frame = int(sr / fps) if fps > 0 else int(sr / 30)

    # Seed RNG from first frame pixel hash for determinism
    seed = analyses[0]['pixel_hash'] if analyses else 42
    rng = np.random.RandomState(seed)

    key_root = stats['key_root']
    key_intervals = stats['key_intervals']
    key_name = stats['key_name']

    # ─── Diagnostic report ────────────────────────────────────────────────
    print(f"""
  ──────────────────────────────────────────────────
  VIDEO STATISTICS
  ──────────────────────────────────────────────────
  Brightness:  mean={stats['brightness_mean']:.3f}  std={stats['brightness_std']:.3f}
  Edges:       mean={stats['edge_mean']:.3f}  std={stats['edge_std']:.3f}
  Motion:      mean={stats['motion_mean']:.3f}  std={stats['motion_std']:.3f}
  Hue:         mean={stats['hue_mean']:.3f}  spread={stats['hue_std_mean']:.3f}
  Repetition:  mean={stats['rep_strength_mean']:.3f}  max={stats['rep_strength_max']:.3f}
  ──────────────────────────────────────────────────
  MUSICAL DECISIONS
  ──────────────────────────────────────────────────
  Key:         {key_name} (from hue {stats['hue_mean']*360:.0f}°)
  Grid:        16th notes at {bpm} BPM ({s16_sec*1000:.1f}ms)
  Chaos:       {chaos:.2f} (0=strict scanline, 1=loose)
  Density:     {density:.2f}
  Bars:        {total_bars}
  Seed:        {seed}
  ──────────────────────────────────────────────────
  Patterns derived from SCANLINE of each frame.
  Hue drives harmony. Repetition drives phrase loops.
  ──────────────────────────────────────────────────
""")

    # ─── Helper: get frame analysis at sample position ────────────────────
    def fa_at(sample_pos):
        t = sample_pos / sr
        idx = min(int(t * fps), num_frames - 1)
        return analyses[max(0, idx)]

    # ─── Allocate buses ───────────────────────────────────────────────────
    print("[2/6] Rendering scanline drums...")
    kick_bus = np.zeros(buf_len)
    snare_bus = np.zeros(buf_len)
    hat_bus = np.zeros(buf_len)
    sub_bus = np.zeros(buf_len)
    bass_bus = np.zeros(buf_len)
    pad_bus_L = np.zeros(buf_len)
    pad_bus_R = np.zeros(buf_len)
    tone_bus = np.zeros(buf_len)
    texture_bus_L = np.zeros(buf_len)
    texture_bus_R = np.zeros(buf_len)
    impact_bus = np.zeros(buf_len)

    n_bands = 48
    tex_phases = np.zeros(n_bands)
    m_thresh = stats['motion_mean']

    # ─── DRUMS: scanline patterns on 16th-note grid ───────────────────────
    pos = 0
    beat_idx = 0
    kick_count = snare_count = hat_count = 0

    while pos < total_samples:
        step = beat_idx % 16
        fa = fa_at(pos)
        scan = fa['scan']

        # Swing: shift timing by column brightness gradient
        swing_samples = int(scan['col_gradient'][step] * s16 * 0.15 * chaos)
        actual_pos = max(0, pos + swing_samples)

        # ── KICK from scanline ──
        if scan['kick_pattern'][step] > 0:
            vel = float(scan['kick_vel'][step]) * density
            # Chaos adds random dropout
            if rng.random() < vel + 0.1:
                kick = synth_kick(sr, fa['brightness'], fa['motion'])
                place(kick_bus, kick * (0.5 + vel * 0.5), actual_pos)
                kick_count += 1

        # ── SNARE from scanline ──
        if scan['snare_pattern'][step] > 0:
            vel = float(scan['snare_vel'][step]) * density
            if rng.random() < vel + 0.15:
                snare = synth_snare(sr, fa['edge_density'], fa['contrast'])
                place(snare_bus, snare * (0.4 + vel * 0.6), actual_pos)
                snare_count += 1

        # ── HATS from scanline ──
        if scan['hat_pattern'][step] > 0:
            vel = float(scan['hat_vel'][step]) * density
            if rng.random() < vel + 0.2:
                hat = synth_hat(sr, fa['blue'], fa['saturation'])
                place(hat_bus, hat * vel, actual_pos)
                hat_count += 1

        # ── Extra: motion-triggered ghost hits (chaos layer) ──
        if fa['motion'] > m_thresh * 1.5 and rng.random() < chaos * 0.3:
            ghost = synth_snare(sr, fa['edge_density'] * 0.5, fa['contrast'])
            place(snare_bus, ghost * 0.15, actual_pos)

        beat_idx += 1
        pos += s16

    print(f"  Kicks: {kick_count}, Snares: {snare_count}, Hats: {hat_count}")

    # ─── SUB BASS: on grid, pitch from hue → key ─────────────────────────
    print("[3/6] Rendering bass + harmony...")
    print("  Sub bass...")
    sub_phase = 0.0
    pos = 0
    beat_idx = 0

    while pos < total_samples:
        step = beat_idx % 16
        fa = fa_at(pos)
        scan = fa['scan']

        # Bass follows kick pattern (play on kick hits)
        if scan['kick_pattern'][step] > 0 and float(scan['kick_vel'][step]) > 0.3:
            # Pitch from per-column hue mapped to scale
            col_hue = float(scan['col_hue'][step])
            degree = int(col_hue * len(key_intervals)) % len(key_intervals)
            freq = scale_freq(key_root, key_intervals, degree, octave=1)

            chunk_len = min(s16 * 2, total_samples - pos)  # sustain for 2 steps
            if chunk_len > 0:
                t = np.arange(chunk_len) / sr
                sub_phase_arr = sub_phase + np.cumsum(np.ones(chunk_len) * 2 * np.pi * freq / sr)
                sub = np.sin(sub_phase_arr)
                sub_phase = sub_phase_arr[-1] if chunk_len > 0 else sub_phase
                sub_gain = (0.3 + fa['red'] * 0.4) * (0.5 + fa['brightness'] * 0.5)
                sub_env = np.exp(-t * 5) * 0.7 + 0.3
                sub_bus[pos:pos+chunk_len] += sub * sub_gain * sub_env

        beat_idx += 1
        pos += s16

    # ─── REESE/MID BASS: scanline-driven ──────────────────────────────────
    print("  Reese bass...")
    pos = 0
    beat_idx = 0

    while pos < total_samples:
        step = beat_idx % 16
        fa = fa_at(pos)
        scan = fa['scan']

        # Play reese on strong kick positions with high red
        if scan['kick_pattern'][step] > 0 and float(scan['col_red'][step]) > 0.3:
            col_hue = float(scan['col_hue'][step])
            degree = int(col_hue * len(key_intervals)) % len(key_intervals)
            freq = scale_freq(key_root, key_intervals, degree, octave=2)

            chunk_len = min(s16, total_samples - pos)
            if chunk_len > 0:
                saw1 = synth_tone(sr, freq * 1.003, chunk_len, waveshape='saw')
                saw2 = synth_tone(sr, freq * 0.997, chunk_len, waveshape='saw')
                reese = (saw1 + saw2) * 0.3
                cutoff = 200 + fa['saturation'] * 2000 + fa['brightness'] * 1500
                reese = apply_lp(reese, cutoff, sr)
                t = np.arange(chunk_len) / sr
                reese *= np.exp(-t * 6) * (0.1 + fa['edge_density'] * 0.2)
                bass_bus[pos:pos+chunk_len] += reese

        beat_idx += 1
        pos += s16

    # ─── CHORD PAD: hue → key, saturation → richness ─────────────────────
    print("  Chord pad...")
    bar_samples = s16 * 16

    for bar_start in range(0, total_samples, bar_samples):
        bar_end = min(bar_start + bar_samples, total_samples)
        bar_len = bar_end - bar_start
        if bar_len <= 0:
            continue

        fa = fa_at(bar_start)
        # Per-bar hue → chord root degree
        degree = int(fa['hue_dominant'] * len(key_intervals)) % len(key_intervals)

        # Build chord: root + 3rd + 5th (or wider if high hue spread)
        chord_freqs = []
        chord_freqs.append(scale_freq(key_root, key_intervals, degree, octave=3))
        chord_freqs.append(scale_freq(key_root, key_intervals, degree + 2, octave=3))
        chord_freqs.append(scale_freq(key_root, key_intervals, degree + 4, octave=3))
        # High hue spread → add 7th for complexity
        if fa['hue_std'] > stats['hue_std_mean']:
            chord_freqs.append(scale_freq(key_root, key_intervals, degree + 6, octave=3))

        cutoff = 200 + fa['brightness'] * 800 + fa['saturation'] * 400
        pad = np.zeros(bar_len)
        for freq in chord_freqs:
            s1 = synth_tone(sr, freq * 1.002, bar_len, waveshape='saw')
            s2 = synth_tone(sr, freq * 0.998, bar_len, waveshape='saw')
            pad += (s1 + s2) * 0.12
        pad /= max(len(chord_freqs), 1)
        pad = apply_lp(pad, cutoff, sr)

        # Envelope
        attack_n = int(0.3 * sr)
        release_n = int(0.5 * sr)
        env = np.ones(bar_len)
        if bar_len > attack_n:
            env[:attack_n] = np.linspace(0, 1, attack_n)
        if bar_len > release_n:
            env[-release_n:] = np.linspace(1, 0, release_n)

        pad_gain = (0.03 + fa['green'] * 0.05)
        pad_bus_L[bar_start:bar_end] += pad * pad_gain * env
        pad_bus_R[bar_start:bar_end] += pad * pad_gain * env * 0.85

    # ─── TONAL EVENTS: hue shifts + glitch artifacts ──────────────────────
    print("  Tonal events (hue-driven)...")
    pos = 0
    beat_idx = 0

    while pos < total_samples:
        step = beat_idx % 16
        fa = fa_at(pos)
        scan = fa['scan']

        # Hue variance in this column → tonal event (glitch detection!)
        col_hue_v = float(scan['col_hue_var'][step])
        hue_glitch = col_hue_v > 0.02  # significant hue variance in column

        # Also trigger on frame-level hue shifts
        hue_shift = fa['hue_dom_delta'] > 0.04 + (1.0 - chaos) * 0.08

        if (hue_glitch or hue_shift) and rng.random() < density * 0.6:
            col_hue = float(scan['col_hue'][step])
            degree = int(col_hue * len(key_intervals)) % len(key_intervals)
            # Pick octave from brightness
            octave = 3 if fa['brightness'] < 0.4 else 4
            freq = scale_freq(key_root, key_intervals, degree, octave=octave)

            note_len = min(int(s16_sec * (1.5 + fa['brightness'] * 2) * sr), buf_len - pos)
            if note_len > 0:
                shapes = ['sine', 'tri', 'saw', 'square']
                shape_idx = fa['pixel_hash'] % len(shapes)
                tone = synth_tone(sr, freq, note_len, fa['brightness'], shapes[shape_idx])
                t_env = np.arange(note_len) / sr
                env = np.exp(-t_env * (4 + (1 - fa['saturation']) * 12))
                tone *= env * (0.025 + fa['saturation'] * 0.035)
                cutoff = 400 + fa['brightness'] * 3000 + fa['saturation'] * 1500
                tone = apply_lp(tone, cutoff, sr)
                place(tone_bus, tone, pos)

                # Delay echo on strong hue glitches
                if hue_glitch and col_hue_v > 0.05:
                    delay_s = int(s16_sec * 6 * sr)
                    place(tone_bus, tone * 0.3, pos + delay_s)

        beat_idx += 1
        pos += s16

    # ─── TEXTURE: scanline spectrogram (additive per frame) ───────────────
    if not no_sonification:
        print("[4/6] Rendering scanline texture...")
        for fi in range(num_frames):
            a = analyses[fi]
            pos = int(fi * samples_per_frame)
            if pos >= total_samples:
                break
            grain_len = min(samples_per_frame, total_samples - pos)
            if grain_len <= 0:
                continue

            tex_grain, tex_phases = synth_texture_grain(
                sr, a['spectral_freqs'], a['spectral_amps'], grain_len, tex_phases
            )
            peak = np.max(np.abs(tex_grain))
            if peak > 0:
                tex_grain /= peak
            tex_gain = (0.03 + a['green'] * 0.06 + a['brightness'] * 0.03) * sonification_gain
            width = 0.3 + a['contrast'] * 0.7
            texture_bus_L[pos:pos+grain_len] += tex_grain * tex_gain * (0.5 + width * 0.5)
            texture_bus_R[pos:pos+grain_len] += tex_grain * tex_gain * (0.5 - width * 0.2)
            d = int(0.004 * sr * (0.5 + a['contrast'] * 0.5))
            if pos + grain_len + d <= buf_len:
                texture_bus_R[pos+d:pos+grain_len+d] += tex_grain * tex_gain * 0.3

            if fi % 200 == 0:
                pct = fi / num_frames * 100
                print(f"\r  Texture [{'#' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)
        print(f"\r  Texture [{'#' * 40}] 100%")
    else:
        print("[4/6] Sonification skipped")

    # ─── IMPACTS: scene cuts ──────────────────────────────────────────────
    print("[5/6] Rendering impacts...")
    for fi in range(num_frames):
        a = analyses[fi]
        pos = int(fi * samples_per_frame)
        if pos >= total_samples:
            break
        if a['motion'] > m_thresh + stats['motion_std'] * 3:
            imp_len = min(int(sr * 0.3), buf_len - pos)
            if imp_len > 0:
                t_imp = np.arange(imp_len) / sr
                boom = np.sin(2 * np.pi * 40 * t_imp * np.exp(-t_imp * 3)) * np.exp(-t_imp * 5)
                crash = rng.randn(imp_len) * np.exp(-t_imp * 8) * 0.3
                crash = apply_hp(crash, 2000, sr)
                impact = (boom * 0.6 + crash * 0.4) * min(a['motion'] * 5, 1.0)
                place(impact_bus, impact, pos)

    # ─── Bus processing ───────────────────────────────────────────────────
    print("\n[6/6] Processing & mixing...")

    # Sub: keep it tight
    sub_bus = apply_lp(sub_bus[:total_samples], 120, sr)
    sub_bus = np.tanh(sub_bus * 1.3)

    # Bass: don't clash with sub
    bass_bus = apply_hp(bass_bus[:total_samples], 80, sr)
    bass_bus = np.tanh(bass_bus * 1.5)

    # Snare: slight room
    snare_bus = comb_reverb(snare_bus[:total_samples], sr, decay=0.12, delays_ms=(17,))

    # Pad: reverb
    pad_bus_L = comb_reverb(pad_bus_L[:total_samples], sr, decay=0.3, delays_ms=(60, 75))
    pad_bus_R = comb_reverb(pad_bus_R[:total_samples], sr, decay=0.3, delays_ms=(75, 90))

    # Texture: reverb for atmosphere
    texture_bus_L = comb_reverb(texture_bus_L[:total_samples], sr, decay=0.25, delays_ms=(43, 59))
    texture_bus_R = comb_reverb(texture_bus_R[:total_samples], sr, decay=0.25, delays_ms=(59, 73))

    # Tone: delay echo
    tone_bus_proc = tone_bus[:total_samples].copy()
    delay_samples = int(beat_dur * 0.75 * sr)  # dotted 8th
    if delay_samples < total_samples:
        delayed = np.zeros(total_samples)
        delayed[delay_samples:] = tone_bus_proc[:-delay_samples] * 0.35
        tone_bus_proc += delayed
    tone_bus_proc = comb_reverb(tone_bus_proc, sr, decay=0.2, delays_ms=(50, 65))

    # Impact: big reverb
    impact_bus = comb_reverb(impact_bus[:total_samples], sr, decay=0.4, delays_ms=(80, 110, 140))

    # ─── Stereo mix ───────────────────────────────────────────────────────
    print("  Mixing...")

    def trim(buf):
        b = buf[:total_samples] if len(buf) >= total_samples else np.pad(buf, (0, total_samples - len(buf)))
        return b

    mix_L = np.zeros(total_samples)
    mix_R = np.zeros(total_samples)

    # Center: kick, snare, sub
    mix_L += trim(kick_bus) * 1.0;    mix_R += trim(kick_bus) * 1.0
    mix_L += trim(snare_bus) * 0.85;  mix_R += trim(snare_bus) * 0.85
    mix_L += trim(sub_bus) * 0.8;     mix_R += trim(sub_bus) * 0.8

    # Slight pan: hats, bass
    mix_L += trim(hat_bus) * 0.55;    mix_R += trim(hat_bus) * 0.4
    mix_L += trim(bass_bus) * 0.5;    mix_R += trim(bass_bus) * 0.5

    # Stereo: pads, texture
    mix_L += trim(pad_bus_L);         mix_R += trim(pad_bus_R)
    mix_L += trim(texture_bus_L)
    mix_R += trim(texture_bus_R)

    # Tone (slight stereo)
    mix_L += trim(tone_bus_proc) * 0.4;   mix_R += trim(tone_bus_proc) * 0.35

    # Impact (wide)
    mix_L += trim(impact_bus) * 0.7;  mix_R += trim(impact_bus) * 0.6

    # ─── Master processing ────────────────────────────────────────────────
    print("  Master processing...")
    stereo = np.column_stack([mix_L, mix_R])

    # Normalize
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.95

    # Soft compression
    threshold = 0.55
    ratio = 3.5
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
    print("\n[5/5] Writing output...")

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_v7_scanline.wav"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    sf.write(output_path, stereo, sr, subtype='PCM_24')
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Main mix: {output_path} ({size_mb:.1f} MB)")

    # ─── Stems ────────────────────────────────────────────────────────────
    if stems:
        stem_dir = os.path.splitext(output_path)[0] + '_stems'
        os.makedirs(stem_dir, exist_ok=True)

        stem_groups = {
            'drums': (trim(kick_bus) + trim(snare_bus) + trim(hat_bus)),
            'bass': (trim(sub_bus) + trim(bass_bus)),
            'pads': np.column_stack([trim(pad_bus_L), trim(pad_bus_R)]),
            'texture': np.column_stack([trim(texture_bus_L), trim(texture_bus_R)]),
            'tones': trim(tone_bus_proc),
            'impacts': trim(impact_bus),
        }

        for name, data in stem_groups.items():
            stem_path = os.path.join(stem_dir, f'{name}.wav')
            if data.ndim == 1:
                data = np.column_stack([data, data])
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            pk = np.max(np.abs(data))
            if pk > 0:
                data = data / pk * 0.95
            sf.write(stem_path, data, sr, subtype='PCM_24')
            print(f"  Stem: {stem_path}")

        print(f"  Stems exported to: {stem_dir}/")

    print(f"""
============================================================
  Output:   {output_path}
  Duration: {duration:.1f}s
  Format:   {sr}Hz / 24-bit / Stereo
  Key:      {key_name} (from hue {stats['hue_mean']*360:.0f}°)
  Chaos:    {chaos:.2f} | Density: {density:.2f}
  SCANLINE PATTERNS + HUE HARMONY + GLITCH DETECTION
============================================================""")

    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='video2dnb v7 — Scanline Glitch Renderer. Image-derived patterns on a BPM grid.')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('--bpm', type=int, default=174, help='BPM (affects delay times)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output WAV path')
    parser.add_argument('--chaos', type=float, default=0.5,
                        help='Chaos level 0-1 (higher = more random triggers)')
    parser.add_argument('--density', type=float, default=0.5,
                        help='Event density scale 0-1 (higher = more events)')
    parser.add_argument('--no-sonification', action='store_true',
                        help='Disable spectral texture layer')
    parser.add_argument('--sonification-gain', type=float, default=0.35,
                        help='Sonification layer gain')
    parser.add_argument('--stems', action='store_true',
                        help='Export stems (drums, bass, pads, texture, tones, impacts)')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate')
    args = parser.parse_args()

    t0 = time.time()
    render(
        video_path=args.video,
        bpm=args.bpm,
        output_path=args.output,
        chaos=np.clip(args.chaos, 0, 1),
        density=np.clip(args.density, 0, 1),
        no_sonification=args.no_sonification,
        sonification_gain=args.sonification_gain,
        stems=args.stems,
        sr=args.sample_rate,
    )
    print(f"  Render time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
