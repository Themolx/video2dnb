#!/usr/bin/env python3
"""
video2dnb_v6.py — RAW CHAOS RENDERER

No hardcoded patterns. No 16th-note grid. No chord progressions.
Every sound event is triggered directly by frame-level pixel statistics.

Design:
  - Each video frame → audio grain (~1/fps seconds)
  - Brightness delta (frame-to-frame) → kick probability
  - Edge density spikes → snare probability
  - High-freq texture (top rows) → hat probability
  - Motion magnitude → transient density multiplier
  - Color channels → independent frequency layers
    - Red channel mean → low-frequency drone pitch
    - Green channel mean → mid-frequency texture density
    - Blue channel mean → high-frequency shimmer pitch
  - Pixel row brightness → additive oscillator bank (spectrogram style)
  - Hue → note selection from chromatic scale (no fixed key)
  - Saturation → resonance / harmonic richness
  - Contrast → stereo width
  - ALL triggers are probability-based, seeded by pixel hash for determinism
  - Rhythm emerges from the video, not from templates

Usage:
    python video2dnb_v6.py <video_path> [--bpm 174] [--output out.wav]
    python video2dnb_v6.py video.mp4 --chaos 0.8   # more random (0-1)
    python video2dnb_v6.py video.mp4 --density 0.5  # event density scale
    python video2dnb_v6.py video.mp4 --stems        # export stems
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

# Chromatic frequencies A1-A5 (no fixed key — hue picks from full chromatic)
CHROMATIC = {}
for i in range(36):  # A1 (55Hz) to A4 (880Hz) — 36 semitones
    CHROMATIC[i] = 55.0 * (2.0 ** (i / 12.0))


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

def analyze_frame(frame, prev_frame=None):
    """Raw frame → feature dict. Minimal processing, maximum data."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = {}

    r['brightness'] = float(np.mean(gray)) / 255.0
    r['contrast'] = min(float(np.std(gray)) / 128.0, 1.0)
    r['red'] = float(np.mean(frame[:, :, 2])) / 255.0
    r['green'] = float(np.mean(frame[:, :, 1])) / 255.0
    r['blue'] = float(np.mean(frame[:, :, 0])) / 255.0
    r['saturation'] = float(np.mean(hsv[:, :, 1])) / 255.0
    r['hue'] = float(np.mean(hsv[:, :, 0])) / 180.0  # 0-1

    edges = cv2.Canny(gray, 50, 150)
    r['edge_density'] = float(np.sum(edges > 0)) / float(edges.size)

    # Vertical edges → transient-like
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    r['vert_edges'] = min(float(np.mean(np.abs(sobel_v))) / 128.0, 1.0)

    # Horizontal edges → sustained tone-like
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
        # Regional motion (quadrants)
        h, w = diff.shape
        r['motion_tl'] = float(np.mean(diff[:h//2, :w//2])) / 255.0
        r['motion_tr'] = float(np.mean(diff[:h//2, w//2:])) / 255.0
        r['motion_bl'] = float(np.mean(diff[h//2:, :w//2])) / 255.0
        r['motion_br'] = float(np.mean(diff[h//2:, w//2:])) / 255.0
    else:
        r['motion'] = 0.0
        r['motion_tl'] = r['motion_tr'] = r['motion_bl'] = r['motion_br'] = 0.0

    # Spectral: image rows → frequency bands
    h, w = gray.shape
    n_bands = 48
    indices = np.linspace(0, h - 1, n_bands).astype(int)
    bands = gray[indices, :]
    amps = np.mean(bands, axis=1) / 255.0
    amps = amps[::-1]  # bottom rows = low freq
    freqs = np.logspace(np.log10(60), np.log10(10000), n_bands)
    r['spectral_freqs'] = freqs
    r['spectral_amps'] = amps

    # Per-column brightness variance (rhythmic texture)
    col_means = np.mean(gray, axis=0) / 255.0
    r['col_variance'] = float(np.var(col_means))

    # Pixel hash for deterministic randomness
    pixel_sample = gray[::10, ::10].flatten()
    r['pixel_hash'] = int(np.sum(pixel_sample.astype(np.int64))) % (2**31)

    return r


def analyze_video(video_path):
    """Analyze all frames. Returns (analyses, fps, duration, total_frames)."""
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

    # Compute deltas for spike detection
    for i in range(len(analyses)):
        if i > 0:
            analyses[i]['brightness_delta'] = analyses[i]['brightness'] - analyses[i-1]['brightness']
            analyses[i]['edge_delta'] = analyses[i]['edge_density'] - analyses[i-1]['edge_density']
            analyses[i]['motion_delta'] = analyses[i]['motion'] - analyses[i-1]['motion']
            analyses[i]['hue_delta'] = abs(analyses[i]['hue'] - analyses[i-1]['hue'])
        else:
            analyses[i]['brightness_delta'] = 0.0
            analyses[i]['edge_delta'] = 0.0
            analyses[i]['motion_delta'] = 0.0
            analyses[i]['hue_delta'] = 0.0

    # Compute running stats for adaptive thresholds
    brights = [a['brightness'] for a in analyses]
    edges = [a['edge_density'] for a in analyses]
    motions = [a['motion'] for a in analyses]

    stats = {
        'brightness_mean': float(np.mean(brights)),
        'brightness_std': float(np.std(brights)),
        'edge_mean': float(np.mean(edges)),
        'edge_std': float(np.std(edges)),
        'motion_mean': float(np.mean(motions)),
        'motion_std': float(np.std(motions)),
    }

    return analyses, fps, duration, total_frames, stats


# ─── Main Render Engine ──────────────────────────────────────────────────────

def render(video_path, bpm=174, output_path=None, chaos=0.5, density=0.5,
           sonification_gain=0.35, no_sonification=False, stems=False,
           sr=SAMPLE_RATE):

    print(f"""
============================================================
  VIDEO2DNB v6 — RAW CHAOS RENDERER
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
    print("[1/5] Analyzing video...")
    analyses, fps, duration, total_frames, stats = analyze_video(video_path)

    total_samples = int(duration * sr)
    tail = sr
    buf_len = total_samples + tail
    num_frames = len(analyses)

    beat_dur = 60.0 / bpm
    samples_per_frame = int(sr / fps) if fps > 0 else int(sr / 30)

    # Seed RNG from first frame pixel hash for determinism
    seed = analyses[0]['pixel_hash'] if analyses else 42
    rng = np.random.RandomState(seed)
    print(f"  RNG seed: {seed}")

    # ─── Diagnostic report ────────────────────────────────────────────────
    print(f"""
  ──────────────────────────────────────────────────
  VIDEO STATISTICS
  ──────────────────────────────────────────────────
  Brightness:  mean={stats['brightness_mean']:.3f}  std={stats['brightness_std']:.3f}
  Edges:       mean={stats['edge_mean']:.3f}  std={stats['edge_std']:.3f}
  Motion:      mean={stats['motion_mean']:.3f}  std={stats['motion_std']:.3f}
  ──────────────────────────────────────────────────
  CHAOS PARAMETERS
  ──────────────────────────────────────────────────
  Chaos:       {chaos:.2f} (higher = more random triggers)
  Density:     {density:.2f} (scales event probability)
  No patterns. No grid. Pure image-driven audio.
  ──────────────────────────────────────────────────
""")

    # ─── Allocate buses ───────────────────────────────────────────────────
    print("[2/5] Rendering layers...")
    kick_bus = np.zeros(buf_len)
    snare_bus = np.zeros(buf_len)
    hat_bus = np.zeros(buf_len)
    drone_bus = np.zeros(buf_len)
    texture_bus_L = np.zeros(buf_len)
    texture_bus_R = np.zeros(buf_len)
    tone_bus = np.zeros(buf_len)
    noise_bus = np.zeros(buf_len)
    impact_bus = np.zeros(buf_len)

    # Phase accumulators for continuous texture
    n_bands = 48
    tex_phases = np.zeros(n_bands)

    # Adaptive thresholds
    b_thresh = stats['brightness_mean']
    e_thresh = stats['edge_mean']
    m_thresh = stats['motion_mean']

    # ─── Per-frame rendering (NO GRID) ────────────────────────────────────
    print("  Processing frames...")
    prev_kick_pos = -sr  # cooldown tracking
    prev_snare_pos = -sr

    for fi in range(num_frames):
        a = analyses[fi]
        pos = int(fi * samples_per_frame)
        if pos >= total_samples:
            break

        grain_len = min(samples_per_frame, total_samples - pos)
        if grain_len <= 0:
            continue

        # Deterministic random for this frame
        frame_rand = rng.random()
        frame_rand2 = rng.random()
        frame_rand3 = rng.random()

        # ── KICK: triggered by brightness spikes or strong motion ──
        brightness_spike = a['brightness_delta'] > stats['brightness_std'] * (1.5 - chaos)
        motion_spike = a['motion'] > m_thresh + stats['motion_std'] * (1.0 - chaos * 0.5)
        kick_prob = 0.0
        if brightness_spike:
            kick_prob += 0.4 + chaos * 0.3
        if motion_spike:
            kick_prob += 0.3 + chaos * 0.2
        # Random chance scaled by density
        kick_prob += a['brightness'] * density * 0.05
        # Cooldown: don't retrigger too fast
        min_kick_gap = int(sr * (0.08 - chaos * 0.03))
        if kick_prob > 0 and frame_rand < kick_prob and (pos - prev_kick_pos) > min_kick_gap:
            kick = synth_kick(sr, a['brightness'], a['motion'])
            place(kick_bus, kick, pos)
            prev_kick_pos = pos

        # ── SNARE: triggered by edge density spikes ──
        edge_spike = a['edge_delta'] > stats['edge_std'] * (1.2 - chaos * 0.5)
        vert_spike = a['vert_edges'] > e_thresh + stats['edge_std'] * (0.8 - chaos * 0.3)
        snare_prob = 0.0
        if edge_spike:
            snare_prob += 0.35 + chaos * 0.25
        if vert_spike:
            snare_prob += 0.25 + chaos * 0.15
        snare_prob += a['edge_density'] * density * 0.04
        min_snare_gap = int(sr * (0.06 - chaos * 0.02))
        if snare_prob > 0 and frame_rand2 < snare_prob and (pos - prev_snare_pos) > min_snare_gap:
            snare = synth_snare(sr, a['edge_density'], a['contrast'])
            place(snare_bus, snare, pos)
            prev_snare_pos = pos

        # ── HATS: triggered by blue channel + high-freq texture ──
        hat_prob = a['blue'] * density * (0.15 + chaos * 0.2)
        hat_prob += a['col_variance'] * 2.0 * density
        if frame_rand3 < hat_prob:
            hat = synth_hat(sr, a['blue'], a['saturation'])
            place(hat_bus, hat, pos)

        # ── DRONE: continuous low tone from red channel ──
        # Pitch from chromatic scale indexed by hue
        note_idx = int(a['hue'] * 12) % 12  # chromatic note
        drone_freq = CHROMATIC.get(note_idx, 55.0)  # octave 1
        drone_grain = synth_drone(sr, drone_freq, grain_len, a['red'], a['saturation'])
        # Fade in/out for smooth grains
        fade = min(64, grain_len // 4)
        if fade > 1:
            drone_grain[:fade] *= np.linspace(0, 1, fade)
            drone_grain[-fade:] *= np.linspace(1, 0, fade)
        drone_bus[pos:pos+grain_len] += drone_grain * 0.3

        # ── TEXTURE: additive oscillator bank from row brightness ──
        if not no_sonification:
            tex_grain, tex_phases = synth_texture_grain(
                sr, a['spectral_freqs'], a['spectral_amps'], grain_len, tex_phases
            )
            peak = np.max(np.abs(tex_grain))
            if peak > 0:
                tex_grain /= peak
            tex_gain = (0.03 + a['green'] * 0.06 + a['brightness'] * 0.03) * sonification_gain
            # Stereo from contrast
            width = 0.3 + a['contrast'] * 0.7
            texture_bus_L[pos:pos+grain_len] += tex_grain * tex_gain * (0.5 + width * 0.5)
            texture_bus_R[pos:pos+grain_len] += tex_grain * tex_gain * (0.5 - width * 0.2)
            # Haas delay for width
            d = int(0.004 * sr * (0.5 + a['contrast'] * 0.5))
            if pos + grain_len + d <= buf_len:
                texture_bus_R[pos+d:pos+grain_len+d] += tex_grain * tex_gain * 0.3

        # ── TONAL EVENTS: note triggered by hue shifts ──
        if a['hue_delta'] > 0.05 + (1.0 - chaos) * 0.1:
            # Pick note from chromatic scale
            note_idx = int(a['hue'] * 36) % 36
            freq = CHROMATIC.get(note_idx, 220.0)
            note_len = min(int(sr * (0.05 + a['brightness'] * 0.2)), buf_len - pos)
            if note_len > 0:
                shapes = ['sine', 'tri', 'saw', 'square']
                shape_idx = a['pixel_hash'] % len(shapes)
                tone = synth_tone(sr, freq, note_len, a['brightness'], shapes[shape_idx])
                # Envelope
                t_env = np.arange(note_len) / sr
                env = np.exp(-t_env * (5 + (1 - a['saturation']) * 15))
                tone *= env * (0.03 + a['saturation'] * 0.04)
                # Filter
                cutoff = 500 + a['brightness'] * 4000
                tone = apply_lp(tone, cutoff, sr)
                place(tone_bus, tone, pos)

        # ── NOISE BURSTS: from motion spikes ──
        if a['motion'] > m_thresh * 2:
            burst_len = min(int(sr * 0.03 * (1 + chaos)), buf_len - pos)
            if burst_len > 0:
                burst = rng.randn(burst_len)
                t_b = np.arange(burst_len) / sr
                burst *= np.exp(-t_b * 30) * a['motion'] * 0.15
                # Bandpass from frame color
                center = 500 + a['warmth'] * 4000
                bw = 500 + a['contrast'] * 2000
                lo = max(20, center - bw/2)
                hi = min(sr/2 - 100, center + bw/2)
                if hi > lo + 50:
                    sos = signal.butter(2, [lo, hi], btype='band', fs=sr, output='sos')
                    burst = signal.sosfilt(sos, burst)
                place(noise_bus, burst, pos)

        # ── IMPACTS: from scene cuts (very large motion) ──
        if a['motion'] > m_thresh + stats['motion_std'] * 3:
            imp_len = min(int(sr * 0.3), buf_len - pos)
            if imp_len > 0:
                t_imp = np.arange(imp_len) / sr
                # Low boom
                boom = np.sin(2 * np.pi * 40 * t_imp * np.exp(-t_imp * 3)) * np.exp(-t_imp * 5)
                # Noise crash
                crash = rng.randn(imp_len) * np.exp(-t_imp * 8) * 0.3
                crash = apply_hp(crash, 2000, sr)
                impact = (boom * 0.6 + crash * 0.4) * min(a['motion'] * 5, 1.0)
                place(impact_bus, impact, pos)

        # Progress
        if fi % 100 == 0:
            pct = fi / num_frames * 100
            print(f"\r  Rendering [{'#' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)

    print(f"\r  Rendering [{'#' * 40}] 100%")

    # ─── Bus processing ───────────────────────────────────────────────────
    print("\n[3/5] Processing buses...")

    # Count events
    kick_events = np.sum(np.abs(kick_bus) > 0.01)
    snare_events = np.sum(np.abs(snare_bus) > 0.01)
    hat_events = np.sum(np.abs(hat_bus) > 0.01)
    print(f"  Kick samples active: {kick_events}, Snare: {snare_events}, Hat: {hat_events}")

    # Drone: lowpass to keep it sub
    drone_bus = apply_lp(drone_bus[:total_samples], 200, sr)
    drone_bus = np.tanh(drone_bus * 1.5)

    # Snare: slight room
    snare_bus = comb_reverb(snare_bus[:total_samples], sr, decay=0.12, delays_ms=(17,))

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

    # Noise: keep it raw
    noise_bus = noise_bus[:total_samples]

    # Impact: big reverb
    impact_bus = comb_reverb(impact_bus[:total_samples], sr, decay=0.4, delays_ms=(80, 110, 140))

    # ─── Stereo mix ───────────────────────────────────────────────────────
    print("\n[4/5] Mixing...")

    def trim(buf):
        b = buf[:total_samples] if len(buf) >= total_samples else np.pad(buf, (0, total_samples - len(buf)))
        return b

    mix_L = np.zeros(total_samples)
    mix_R = np.zeros(total_samples)

    # Center: kick, snare, drone
    mix_L += trim(kick_bus) * 1.0;    mix_R += trim(kick_bus) * 1.0
    mix_L += trim(snare_bus) * 0.85;  mix_R += trim(snare_bus) * 0.85
    mix_L += trim(drone_bus) * 0.7;   mix_R += trim(drone_bus) * 0.7

    # Slight pan: hats
    mix_L += trim(hat_bus) * 0.55;    mix_R += trim(hat_bus) * 0.4

    # Stereo texture
    mix_L += trim(texture_bus_L)
    mix_R += trim(texture_bus_R)

    # Tone + noise (slight stereo)
    mix_L += trim(tone_bus_proc) * 0.45;  mix_R += trim(tone_bus_proc) * 0.4
    mix_L += trim(noise_bus) * 0.5;       mix_R += trim(noise_bus) * 0.45

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
        output_path = f"{base}_v6_chaos.wav"

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
            'drone': trim(drone_bus),
            'texture': np.column_stack([trim(texture_bus_L), trim(texture_bus_R)]),
            'tones': trim(tone_bus_proc),
            'noise': trim(noise_bus),
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
  Chaos:    {chaos:.2f} | Density: {density:.2f}
  NO PATTERNS. NO GRID. PURE FRAME-DRIVEN.
============================================================""")

    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='video2dnb v6 — Raw Chaos Renderer. No patterns, no grid.')
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
                        help='Export stems (drums, drone, texture, tones, noise, impacts)')
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
