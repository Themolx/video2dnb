#!/usr/bin/env python3
"""
video2dnb_v4.py — Scanline Spectrogram DnB Renderer

The image IS the sound. Every frame is treated as a spectrogram:
  - Rows = frequency bands (bottom=bass, top=treble)
  - Pixel brightness = amplitude at that frequency
  - Columns = time slices within each frame's duration
  - RGB channels drive separate frequency ranges

Rhythm is extracted deterministically from image structure:
  - Vertical edges → transient hits (kicks, snares, clicks)
  - Horizontal edges → sustained tones
  - Column energy variance → rhythmic pulse strength
  - Diagonal patterns → rolling/shuffling rhythms

Based on v1 structure with synthesis primitives and DnB layering,
but everything is driven by raw pixel data — no templates, no randomness.

Usage:
    python video2dnb_v4.py video.mp4 --bpm 174 --output audio.wav
"""

import cv2
import numpy as np
import soundfile as sf
from scipy import signal
import argparse, os, sys, time

SR = 48000
DEFAULT_BPM = 174

# ─── Notes ───────────────────────────────────────────────────────────────────
NOTES = {
    'c1': 32.70, 'eb1': 38.89, 'f1': 43.65, 'ab1': 51.91, 'bb1': 58.27,
    'c2': 65.41, 'eb2': 77.78, 'f2': 87.31, 'g2': 98.00, 'ab2': 103.83, 'bb2': 116.54,
    'c3': 130.81, 'eb3': 155.56, 'f3': 174.61, 'g3': 196.00, 'ab3': 207.65, 'bb3': 233.08,
    'c4': 261.63, 'eb4': 311.13, 'f4': 349.23, 'g4': 392.00, 'ab4': 415.30, 'bb4': 466.16,
    'c5': 523.25, 'eb5': 622.25, 'g5': 783.99,
}

CHORDS = [
    ("Cm",  'c1','c2', ['c3','eb3','g3'],       ['c4','eb4','g4','c5']),
    ("Cm7", 'c1','c2', ['c3','eb3','g3','bb3'], ['c4','eb4','g4','bb4']),
    ("Ab",  'ab1','ab2',['ab3','c4','eb4'],      ['ab4','c5','eb5']),
    ("Fm",  'f1','f2', ['f3','ab3','c4'],        ['f4','ab4','c5']),
    ("Eb",  'eb1','eb2',['eb3','g3','bb3'],      ['eb4','g4','bb4']),
    ("Bbm", 'bb1','bb2',['bb3','eb4','g4'],      ['bb4','eb5','g5']),
]

PROGRESSIONS = [
    [0, 0, 2, 2], [0, 0, 2, 3], [0, 1, 2, 3],
    [1, 4, 2, 3], [2, 3, 0, 0], [3, 2, 0, 1],
]


# ─── Synthesis Primitives (from v1) ─────────────────────────────────────────

def make_kick(sr=SR, duration=0.15, gain=1.0, pitch_mul=1.0):
    n = int(sr * duration)
    t = np.arange(n) / sr
    pitch_env = (150 * pitch_mul) * np.exp(-t * 40) + 40
    phase = np.cumsum(2 * np.pi * pitch_env / sr)
    osc = np.sin(phase) * np.exp(-t * 12)
    click = np.exp(-t * 200) * 0.5
    return np.tanh((osc + click) * 1.5 * gain) * 0.9

def make_snare(sr=SR, duration=0.1, gain=1.0, tone_freq=200):
    n = int(sr * duration)
    t = np.arange(n) / sr
    noise = np.random.randn(n) * np.exp(-t * 20)
    sos = signal.butter(2, [2000, 8000], btype='band', fs=sr, output='sos')
    noise_f = signal.sosfilt(sos, noise)
    tone = np.sin(2 * np.pi * tone_freq * t) * np.exp(-t * 30)
    return np.tanh((noise_f * 0.7 + tone * 0.5) * gain * 2) * 0.7

def make_ghost(sr=SR, gain=0.15):
    return make_snare(sr=sr, duration=0.06, gain=gain)

def make_hihat(sr=SR, gain=0.3, is_open=False):
    dur = 0.15 if is_open else 0.04
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    env = np.exp(-t * (8 if is_open else 40))
    sos = signal.butter(2, 6000, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, noise) * env * gain

def make_click(sr=SR, gain=0.4, freq=4000):
    """Short transient click from image edges."""
    n = int(sr * 0.008)
    t = np.arange(n) / sr
    click = np.sin(2 * np.pi * freq * t) * np.exp(-t * 500) * gain
    return click

def make_saw(freq, n_samples, sr=SR, num_harmonics=15, gain=1.0):
    t = np.arange(n_samples) / sr
    wave = np.zeros(n_samples)
    for k in range(1, num_harmonics + 1):
        if k * freq > sr / 2: break
        wave += np.sin(2 * np.pi * k * freq * t) / k
    return wave * gain * (2.0 / np.pi)

def apply_lp(audio, cutoff_hz, sr=SR):
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype='low', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def apply_hp(audio, cutoff_hz, sr=SR):
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def comb_reverb(audio, sr=SR, decay=0.3, delays_ms=(23, 37, 49, 61)):
    out = audio.copy()
    for d_ms in delays_ms:
        d = int(d_ms * sr / 1000)
        if d < len(audio):
            delayed = np.zeros_like(audio)
            delayed[d:] = audio[:-d]
            out += delayed * decay * 0.25
    return out

def env_adsr(n, sr=SR, a=0.01, d=0.05, s=0.7, r=0.05):
    ai, di, ri = int(a*sr), int(d*sr), int(r*sr)
    env = np.zeros(n)
    pos = 0
    seg = min(ai, n); env[pos:pos+seg] = np.linspace(0, 1, seg) if seg > 0 else 0; pos += seg
    seg = min(di, n-pos); env[pos:pos+seg] = np.linspace(1, s, seg) if seg > 0 else 0; pos += seg
    sus = max(0, n-pos-ri); env[pos:pos+sus] = s; pos += sus
    seg = min(ri, n-pos); env[pos:pos+seg] = np.linspace(s, 0, seg) if seg > 0 else 0
    return env

def place(buf, sound, pos):
    end = min(pos + len(sound), len(buf))
    length = end - pos
    if length > 0 and pos >= 0:
        buf[pos:end] += sound[:length]


# ─── Scanline Image Analysis ────────────────────────────────────────────────

def scanline_analyze(frame, prev_frame=None):
    """
    Deep scanline analysis: extract everything from raw pixels.
    
    Returns dict with:
      - Standard visual params (brightness, contrast, etc.)
      - Scanline spectrogram data (rows → freq magnitudes per color channel)
      - Edge maps (horizontal + vertical separately)
      - Column rhythm pattern (16 steps from column energy)
      - Row melodic contour (32 bands)
      - Transient map (where edges are strongest → where to place hits)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = {}
    
    # ─── Basic visual params ──────────────────────────────────────────
    r['brightness'] = float(np.mean(gray)) / 255.0
    r['contrast'] = min(float(np.std(gray)) / 128.0, 1.0)
    r['red'] = float(np.mean(frame[:,:,2])) / 255.0
    r['green'] = float(np.mean(frame[:,:,1])) / 255.0
    r['blue'] = float(np.mean(frame[:,:,0])) / 255.0
    r['saturation'] = float(np.mean(hsv[:,:,1])) / 255.0
    
    edges = cv2.Canny(gray, 50, 150)
    r['edge_density'] = float(np.sum(edges > 0)) / float(edges.size)
    
    warm = r['red'] + r['green'] * 0.3
    cool = r['blue'] + r['green'] * 0.3
    total = warm + cool
    r['warmth'] = warm / total if total > 0 else 0.5
    
    sat_mask = hsv[:,:,1] > 20
    r['hue'] = float(np.median(hsv[:,:,0][sat_mask])) * 2.0 / 360.0 if np.any(sat_mask) else 0.0
    
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        r['motion'] = float(np.mean(cv2.absdiff(gray, prev_gray))) / 255.0
    else:
        r['motion'] = 0.0
    
    # ─── SCANLINE SPECTROGRAM ─────────────────────────────────────────
    # Treat image as 3-band spectrogram: R=low, G=mid, B=high
    # Each row = frequency band, brightness = amplitude
    h, w = frame.shape[:2]
    
    # 128 frequency bands from image rows (bottom=low, top=high)
    n_bands = 128
    row_indices = np.linspace(0, h-1, n_bands).astype(int)[::-1]  # flip: bottom=low freq
    
    # Per-channel scanline spectrograms
    r['scan_red'] = np.mean(frame[row_indices, :, 2], axis=1) / 255.0   # low freqs
    r['scan_green'] = np.mean(frame[row_indices, :, 1], axis=1) / 255.0 # mid freqs
    r['scan_blue'] = np.mean(frame[row_indices, :, 0], axis=1) / 255.0  # high freqs
    r['scan_gray'] = np.mean(gray[row_indices, :], axis=1) / 255.0      # combined
    
    # ─── COLUMN RHYTHM EXTRACTION ─────────────────────────────────────
    # 16 columns → 16th note grid, deterministic rhythm from pixel patterns
    col_indices = np.linspace(0, w-1, 16).astype(int)
    
    # Column energy = mean brightness
    col_energy = np.array([np.mean(gray[:, max(0,ci-2):ci+3]) / 255.0 for ci in col_indices])
    
    # Column edge density = vertical edges in each column strip
    # Sobel vertical (detects vertical edges = transients)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = np.abs(sobel_v)
    col_edges = np.array([np.mean(sobel_v[:, max(0,ci-2):ci+3]) / 255.0 for ci in col_indices])
    
    # Column contrast = how much variation in the column (drives ghost notes)
    col_contrast = np.array([np.std(gray[:, max(0,ci-2):ci+3]) / 128.0 for ci in col_indices])
    
    r['col_energy'] = col_energy
    r['col_edges'] = col_edges
    r['col_contrast'] = col_contrast
    
    # ─── DETERMINISTIC DRUM PATTERNS FROM IMAGE ──────────────────────
    # Kick: triggered by DARK columns (valleys in energy) + strong vertical edges
    # Snare: triggered by BRIGHT columns (peaks in energy) + high contrast
    # Hat: triggered by column edge density
    # Ghost: triggered by medium contrast columns
    
    # Normalize column energy to 0-1 relative to this frame
    ce_min, ce_max = np.min(col_energy), np.max(col_energy)
    ce_range = ce_max - ce_min if ce_max > ce_min else 0.001
    col_norm = (col_energy - ce_min) / ce_range
    
    # Kick: dark spots (low brightness) or vertical edge peaks
    kick_strength = (1.0 - col_norm) * 0.6 + col_edges * 0.4
    kick_thresh = np.percentile(kick_strength, 65)
    r['kick_pattern'] = (kick_strength > kick_thresh).astype(float) * kick_strength
    # Always have kick on beat 1
    r['kick_pattern'][0] = max(r['kick_pattern'][0], 0.7)
    
    # Snare: bright spots with high contrast
    snare_strength = col_norm * 0.5 + col_contrast * 0.5
    snare_thresh = np.percentile(snare_strength, 70)
    r['snare_pattern'] = (snare_strength > snare_thresh).astype(float) * snare_strength
    # Ensure snare on beat 2 (step 4) if there's any energy
    if r['snare_pattern'][4] < 0.3:
        r['snare_pattern'][4] = max(r['snare_pattern'][4], snare_strength[4] * 0.8)
    
    # Hat: edge density in columns
    hat_thresh = np.percentile(col_edges, 40)
    r['hat_pattern'] = (col_edges > hat_thresh).astype(float) * col_edges
    
    # Ghost: medium contrast, not where kick or snare is
    ghost_strength = col_contrast * (1.0 - r['kick_pattern']) * (1.0 - r['snare_pattern'])
    ghost_thresh = np.percentile(ghost_strength, 60)
    r['ghost_pattern'] = (ghost_strength > ghost_thresh).astype(float) * ghost_strength * 0.3
    
    # ─── HORIZONTAL EDGES → SUSTAINED TONES ──────────────────────────
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_h = np.abs(sobel_h)
    # Row-wise horizontal edge strength → which frequency bands have edges
    h_edge_rows = np.mean(sobel_h[row_indices, :], axis=1) / 255.0
    r['h_edges'] = h_edge_rows
    
    # ─── ROW MELODIC CONTOUR ─────────────────────────────────────────
    # Find the brightest rows → dominant pitch
    scan = r['scan_gray']
    # Weighted centroid = dominant pitch position
    positions = np.arange(n_bands)
    total_energy = np.sum(scan)
    if total_energy > 0:
        r['spectral_centroid'] = float(np.sum(positions * scan) / total_energy) / n_bands
    else:
        r['spectral_centroid'] = 0.5
    
    # Find peaks in the row profile → resonant frequencies
    peaks, props = signal.find_peaks(scan, height=0.15, distance=3)
    r['spectral_peaks'] = peaks
    r['spectral_peak_amps'] = scan[peaks] if len(peaks) > 0 else np.array([])
    
    # Composite energy
    r['energy'] = (0.25 * r['brightness'] + 0.25 * r['motion'] +
                   0.20 * r['edge_density'] + 0.15 * r['contrast'] +
                   0.15 * r['saturation'])
    
    return r


# ─── Video Analysis ──────────────────────────────────────────────────────────

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'"); sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
    
    analyses = []
    prev_small = None
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (160, 120))
        analysis = scanline_analyze(small, prev_small)
        analyses.append(analysis)
        prev_small = small
        idx += 1
        if idx % 30 == 0:
            pct = idx / total_frames * 100
            print(f"\r  Analyzing [{'#' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)
    
    cap.release()
    print(f"\r  Analyzing [{'#' * 40}] 100%")
    print(f"  {len(analyses)} frames analyzed")
    
    # Rank-normalize energy
    energies = [a['energy'] for a in analyses]
    e_min, e_max = min(energies), max(energies)
    e_range = e_max - e_min if e_max > e_min else 1.0
    for a in analyses:
        a['energy_norm'] = (a['energy'] - e_min) / e_range
    
    return analyses, fps, duration, total_frames


# ─── Scanline Spectrogram Synthesis ──────────────────────────────────────────

def render_scanline_spectrogram(analyses, fps, duration, sr=SR):
    """
    THE CORE: treat each frame as a spectrogram slice.
    
    - 128 frequency bands mapped to 30Hz-12kHz (log-spaced)
    - RGB channels drive different ranges:
        Red   → 30-200Hz   (sub/bass)
        Green → 200-2000Hz (mid)
        Blue  → 2000-12kHz (treble)
    - Phase-continuous oscillators for smooth sound
    - Horizontal edges boost sustained tones
    - Motion between frames adds FM modulation
    
    Returns (L, R) stereo tuple.
    """
    total_samples = int(sr * duration)
    n_frames = len(analyses)
    samples_per_frame = total_samples // n_frames if n_frames > 0 else 1
    
    n_bands = 128
    freqs = np.logspace(np.log10(30), np.log10(12000), n_bands)
    phases = np.zeros(n_bands)
    
    # Frequency range masks for RGB mixing
    red_mask = freqs < 200
    green_mask = (freqs >= 150) & (freqs < 2500)
    blue_mask = freqs >= 2000
    # Overlap zones for smooth blending
    red_weight = np.where(red_mask, 1.0, np.where(freqs < 300, (300-freqs)/100, 0.0))
    green_weight = np.where(green_mask, 1.0, 0.0)
    # Smooth edges
    green_weight = np.where((freqs >= 100) & (freqs < 150), (freqs-100)/50, green_weight)
    green_weight = np.where((freqs >= 2500) & (freqs < 3000), (3000-freqs)/500, green_weight)
    blue_weight = np.where(blue_mask, 1.0, np.where(freqs >= 1500, (freqs-1500)/500, 0.0))
    
    out_L = np.zeros(total_samples + sr)
    out_R = np.zeros(total_samples + sr)
    
    prev_scan = None
    
    for fi in range(n_frames):
        a = analyses[fi]
        chunk_start = fi * samples_per_frame
        chunk_end = min(chunk_start + samples_per_frame, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0: continue
        
        # Get scanline data per channel
        scan_r = a['scan_red']    # 128 bands
        scan_g = a['scan_green']
        scan_b = a['scan_blue']
        scan_gray = a['scan_gray']
        h_edges = a['h_edges']
        motion = a['motion']
        
        # Combine RGB channels weighted by frequency range
        combined_amps = np.zeros(n_bands)
        for i in range(n_bands):
            # Each channel contributes to its frequency range
            amp = (scan_r[i] * red_weight[i] + 
                   scan_g[i] * green_weight[i] + 
                   scan_b[i] * blue_weight[i])
            # Horizontal edges boost sustained tones
            amp *= (1.0 + h_edges[i] * 0.5)
            combined_amps[i] = amp
        
        # Normalize to prevent clipping
        peak_amp = np.max(combined_amps)
        if peak_amp > 0:
            combined_amps /= peak_amp
        
        # Smooth transition from previous frame
        if prev_scan is not None:
            alpha = 0.3  # interpolation speed
            combined_amps = prev_scan * (1 - alpha) + combined_amps * alpha
        prev_scan = combined_amps.copy()
        
        # Synthesize
        chunk_L = np.zeros(chunk_len)
        chunk_R = np.zeros(chunk_len)
        
        for i in range(n_bands):
            amp = combined_amps[i]
            if amp < 0.02: continue
            
            freq = freqs[i]
            if freq > sr / 2: continue
            
            # FM modulation from motion (subtle pitch drift when image moves)
            fm_depth = motion * 2.0
            fm_rate = 3.0 + motion * 10.0
            
            phase_inc = 2 * np.pi * freq / sr
            t_local = np.arange(chunk_len)
            fm = fm_depth * np.sin(2 * np.pi * fm_rate * t_local / sr)
            
            ph = phases[i] + np.cumsum(phase_inc * (1.0 + fm * 0.01))
            osc = np.sin(ph) * amp
            
            phases[i] = ph[-1] % (2 * np.pi)
            
            # Stereo placement based on frequency
            # Low = center, mid = wide, high = slightly narrower
            if freq < 200:
                pan_l, pan_r = 0.7, 0.7  # center
            elif freq < 2000:
                # Pan mid frequencies based on which side of image is brighter
                lr_bias = a['red'] - a['blue']
                pan_l = 0.5 + lr_bias * 0.3
                pan_r = 0.5 - lr_bias * 0.3
            else:
                pan_l, pan_r = 0.45, 0.55  # slightly right-biased highs
            
            chunk_L += osc * pan_l
            chunk_R += osc * pan_r
        
        # Apply brightness-driven amplitude envelope
        brightness_env = a['brightness'] * 0.6 + 0.4
        
        # Per-frame dynamics from contrast (high contrast = punchy, low = smooth)
        contrast_env = np.ones(chunk_len)
        if a['contrast'] > 0.3:
            # Add a slight punch at frame start
            punch_len = min(int(0.005 * sr), chunk_len)
            if punch_len > 0:
                contrast_env[:punch_len] *= (1.0 + a['contrast'] * 0.5)
        
        chunk_L *= brightness_env * contrast_env
        chunk_R *= brightness_env * contrast_env
        
        out_L[chunk_start:chunk_end] += chunk_L
        out_R[chunk_start:chunk_end] += chunk_R
    
    out_L = out_L[:total_samples]
    out_R = out_R[:total_samples]
    
    # Normalize
    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)))
    if peak > 0:
        out_L /= peak
        out_R /= peak
    
    return out_L, out_R


# ─── Main Render Engine ─────────────────────────────────────────────────────

def render_video_to_dnb(video_path, bpm=174, output_path=None,
                        scanline_gain=0.55, sr=SR):
    
    # ─── Analyze ─────────────────────────────────────────────────────
    print("[1/5] Analyzing video (scanline mode)...")
    analyses, fps, duration, total_frames = analyze_video(video_path)
    
    total_samples = int(duration * sr)
    tail = sr
    buf_len = total_samples + tail
    
    beat_dur = 60.0 / bpm
    s16 = int(beat_dur / 4 * sr)
    s16_sec = beat_dur / 4
    total_bars = int(duration / (beat_dur * 4))
    
    print(f"\n  BPM: {bpm}, 16th={s16_sec*1000:.1f}ms, bars={total_bars}")
    
    num_frames = len(analyses)
    
    def fa_at(sample_pos):
        t = sample_pos / sr
        idx = min(int(t * fps), num_frames - 1)
        return analyses[max(0, idx)]
    
    # Arrangement envelope
    intro_bars = max(1, min(4, total_bars // 4))
    outro_bars = max(1, min(4, total_bars // 4))
    
    def arr_gain(sample_pos):
        t = sample_pos / sr
        bar = t / (beat_dur * 4)
        if bar < intro_bars:
            return 0.2 + 0.8 * (bar / intro_bars)
        elif total_bars > outro_bars and bar > total_bars - outro_bars:
            remaining = max(0, total_bars - bar)
            return 0.2 + 0.8 * (remaining / outro_bars)
        return 1.0
    
    # ─── Allocate buses ──────────────────────────────────────────────
    print("\n[2/5] Rendering DnB layers (image-deterministic)...")
    kick_bus = np.zeros(buf_len)
    snare_bus = np.zeros(buf_len)
    hat_bus = np.zeros(buf_len)
    click_bus = np.zeros(buf_len)
    sub_bus = np.zeros(buf_len)
    reese_bus = np.zeros(buf_len)
    pad_bus_L = np.zeros(buf_len)
    pad_bus_R = np.zeros(buf_len)
    arp_bus = np.zeros(buf_len)
    mel_bus = np.zeros(buf_len)
    
    # ─── DRUMS: deterministic from image scanline patterns ───────────
    print("  Drums (scanline-deterministic)...")
    
    pos = 0
    beat_idx = 0
    
    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)
        ag = arr_gain(pos)
        
        # Get this frame's image-derived drum patterns
        kick_vel = fa['kick_pattern'][s16_in_bar]
        snare_vel = fa['snare_pattern'][s16_in_bar]
        hat_vel = fa['hat_pattern'][s16_in_bar]
        ghost_vel = fa['ghost_pattern'][s16_in_bar]
        
        # KICK — velocity from image column darkness + edge strength
        if kick_vel > 0.2:
            pitch_mul = 0.8 + fa['brightness'] * 0.4  # darker image = lower kick
            g = kick_vel * (0.7 + fa['energy_norm'] * 0.3) * ag
            k = make_kick(sr=sr, pitch_mul=pitch_mul, gain=g)
            place(kick_bus, k, pos)
        
        # SNARE — velocity from image column brightness + contrast
        if snare_vel > 0.2:
            tone_freq = 180 + fa['spectral_centroid'] * 80  # image content shifts snare tone
            g = snare_vel * (0.6 + fa['edge_density'] * 0.4) * ag
            s = make_snare(sr=sr, tone_freq=tone_freq, gain=g)
            place(snare_bus, s, pos)
        elif ghost_vel > 0.05:
            g = ghost_vel * ag
            place(snare_bus, make_ghost(sr=sr, gain=g), pos)
        
        # HAT — from column edge density
        if hat_vel > 0.1:
            is_open = (s16_in_bar in (6, 14) and fa['energy_norm'] > 0.5)
            h = make_hihat(sr=sr, gain=hat_vel * ag * 0.6, is_open=is_open)
            place(hat_bus, h, pos)
        
        # CLICK — from vertical edges (extra transients the image "asks for")
        edge_val = fa['col_edges'][s16_in_bar]
        if edge_val > 0.15:
            click_freq = 2000 + fa['spectral_centroid'] * 6000
            c = make_click(sr=sr, gain=edge_val * 0.5 * ag, freq=click_freq)
            place(click_bus, c, pos)
        
        beat_idx += 1
        pos += s16
    
    # ─── Sub bass ────────────────────────────────────────────────────
    print("  Sub bass (scanline-driven)...")
    sub_phase = 0.0
    
    for chunk_start in range(0, total_samples, s16):
        chunk_end = min(chunk_start + s16, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0: continue
        
        fa = fa_at(chunk_start)
        s16_in_bar = (chunk_start // s16) % 16
        
        # Chord from warmth
        warmth = fa['warmth']
        prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS)-1)), 0, len(PROGRESSIONS)-1))
        bar_in_prog = ((chunk_start // s16) // 16) % 4
        chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
        chord = CHORDS[chord_idx]
        
        # Sub follows kick pattern from image
        kick_vel = fa['kick_pattern'][s16_in_bar]
        if kick_vel > 0.15:
            freq = NOTES.get(chord[1], 32.7)
            # Red channel = sub bass character (more red = deeper sub)
            sub_red_boost = 0.8 + fa['red'] * 0.4
            freq *= sub_red_boost
            
            t = np.arange(chunk_len) / sr
            sub_phase_arr = sub_phase + np.cumsum(np.ones(chunk_len) * 2 * np.pi * freq / sr)
            sub = np.sin(sub_phase_arr)
            sub_phase = sub_phase_arr[-1] if chunk_len > 0 else sub_phase
            
            # Gain from low-frequency scanline energy
            low_scan_energy = np.mean(fa['scan_red'][:16])  # bottom 16 bands
            sub_gain = (0.2 + low_scan_energy * 0.6 + fa['red'] * 0.2) * arr_gain(chunk_start)
            sub_env = np.exp(-t * 4) * 0.7 + 0.3
            
            sub_bus[chunk_start:chunk_end] += sub * sub_gain * sub_env
    
    # ─── Reese bass ──────────────────────────────────────────────────
    print("  Reese bass (scanline-driven)...")
    
    for chunk_start in range(0, total_samples, s16):
        chunk_end = min(chunk_start + s16, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0: continue
        
        fa = fa_at(chunk_start)
        s16_in_bar = (chunk_start // s16) % 16
        
        warmth = fa['warmth']
        prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS)-1)), 0, len(PROGRESSIONS)-1))
        bar_in_prog = ((chunk_start // s16) // 16) % 4
        chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
        chord = CHORDS[chord_idx]
        
        # Reese follows bass pattern from image but with different threshold
        kick_vel = fa['kick_pattern'][s16_in_bar]
        if kick_vel > 0.3:
            freq = NOTES.get(chord[2], 65.41)
            
            # Detune amount from image edge density
            detune = 1.001 + fa['edge_density'] * 0.005
            saw1 = make_saw(freq * detune, chunk_len, sr=sr, gain=0.3)
            saw2 = make_saw(freq / detune, chunk_len, sr=sr, gain=0.3)
            reese = saw1 + saw2
            
            # Filter cutoff from scanline spectral centroid
            centroid = fa['spectral_centroid']
            cutoff = 150 + centroid * 3000 + fa['saturation'] * 1500
            reese = apply_lp(reese, cutoff, sr)
            
            t = np.arange(chunk_len) / sr
            reese_env = np.exp(-t * 6) * 0.7 + 0.3
            
            # Gain from mid-frequency scanline energy
            mid_scan_energy = np.mean(fa['scan_green'][16:64])
            reese_gain = (0.05 + mid_scan_energy * 0.25 + fa['edge_density'] * 0.15) * arr_gain(chunk_start)
            
            reese_bus[chunk_start:chunk_end] += reese * reese_gain * reese_env
    
    # ─── Chord pad ───────────────────────────────────────────────────
    print("  Chord pad (scanline-modulated)...")
    bar_samples = s16 * 16
    
    for bar_start in range(0, total_samples, bar_samples):
        bar_end = min(bar_start + bar_samples, total_samples)
        bar_len = bar_end - bar_start
        if bar_len <= 0: continue
        
        fa = fa_at(bar_start)
        warmth = fa['warmth']
        prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS)-1)), 0, len(PROGRESSIONS)-1))
        bar_in_prog = (bar_start // bar_samples) % 4
        chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
        chord = CHORDS[chord_idx]
        pad_notes = chord[3]
        
        # Cutoff driven by green channel scanline centroid
        green_centroid = float(np.sum(np.arange(128) * fa['scan_green']) / max(np.sum(fa['scan_green']), 0.01)) / 128.0
        cutoff = 150 + green_centroid * 1200 + fa['brightness'] * 600
        
        pad = np.zeros(bar_len)
        for note_name in pad_notes:
            freq = NOTES.get(note_name, 200)
            saw1 = make_saw(freq * 1.002, bar_len, sr=sr, gain=0.15, num_harmonics=8)
            saw2 = make_saw(freq * 0.998, bar_len, sr=sr, gain=0.15, num_harmonics=8)
            pad += saw1 + saw2
        
        pad /= max(len(pad_notes), 1)
        pad = apply_lp(pad, cutoff, sr)
        
        attack_n = int(0.3 * sr)
        release_n = int(0.5 * sr)
        env = np.ones(bar_len)
        if bar_len > attack_n: env[:attack_n] = np.linspace(0, 1, attack_n)
        if bar_len > release_n: env[-release_n:] = np.linspace(1, 0, release_n)
        
        pad_gain = (0.03 + fa['green'] * 0.06) * arr_gain(bar_start)
        
        pad_bus_L[bar_start:bar_end] += pad * pad_gain * env
        pad_bus_R[bar_start:bar_end] += pad * pad_gain * env * 0.8
    
    # ─── Arp lead (from spectral peaks) ──────────────────────────────
    print("  Arp lead (spectral peak-driven)...")
    arp_counter = 0
    pos = 0
    beat_idx = 0
    
    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)
        e = fa['energy_norm']
        
        # Arp density from energy
        play = False
        if e > 0.6: play = (s16_in_bar % 2 == 0)
        elif e > 0.35: play = (s16_in_bar % 4 == 0)
        else: play = (s16_in_bar % 8 == 0)
        
        if play:
            warmth = fa['warmth']
            prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS)-1)), 0, len(PROGRESSIONS)-1))
            bar_in_prog = (beat_idx // 16) % 4
            chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
            chord = CHORDS[chord_idx]
            arp_notes = chord[4]
            
            # Note selection from spectral peaks instead of simple counter
            peaks = fa['spectral_peaks']
            if len(peaks) > 0:
                # Map strongest spectral peak to note
                peak_pos = peaks[arp_counter % len(peaks)]
                note_idx = int(peak_pos / 128.0 * len(arp_notes)) % len(arp_notes)
            else:
                note_idx = arp_counter % len(arp_notes)
            
            note_name = arp_notes[note_idx]
            freq = NOTES.get(note_name, 440)
            
            note_len = int(s16_sec * 1.5 * sr)
            t = np.arange(note_len) / sr
            note = np.sin(2 * np.pi * freq * t)
            note *= env_adsr(note_len, sr, a=0.002, d=0.04, s=0.0, r=0.02)
            
            gain = e * 0.06 * arr_gain(pos)
            place(arp_bus, note * gain, pos)
            arp_counter += 1
        
        beat_idx += 1
        pos += s16
    
    # ─── Melody (from spectral centroid) ─────────────────────────────
    print("  Melody (spectral centroid-driven)...")
    pos = 0
    beat_idx = 0
    
    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)
        
        if s16_in_bar in (2, 10):
            warmth = fa['warmth']
            prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS)-1)), 0, len(PROGRESSIONS)-1))
            bar_in_prog = (beat_idx // 16) % 4
            chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
            chord = CHORDS[chord_idx]
            arp_notes = chord[4]
            
            # Note from spectral centroid
            centroid = fa['spectral_centroid']
            note_idx = int(centroid * len(arp_notes)) % len(arp_notes)
            note_name = arp_notes[note_idx]
            freq = NOTES.get(note_name, 440)
            
            note_len = int(s16_sec * 3 * sr)
            t = np.arange(note_len) / sr
            note = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
            note *= env_adsr(note_len, sr, a=0.01, d=0.15, s=0.2, r=0.2)
            note = apply_lp(note, 2000, sr)
            
            gain = 0.07 * arr_gain(pos)
            place(mel_bus, note * gain, pos)
            
            delay_s = int(s16_sec * 6 * sr)
            place(mel_bus, note * gain * 0.35, pos + delay_s)
            place(mel_bus, note * gain * 0.12, pos + delay_s * 2)
        
        beat_idx += 1
        pos += s16
    
    # ─── Scanline spectrogram ────────────────────────────────────────
    print("\n[3/5] Rendering scanline spectrogram...")
    print("  Scanning image → frequency magnitudes → audio...")
    scan_L, scan_R = render_scanline_spectrogram(analyses, fps, duration, sr=sr)
    
    # ─── Bus processing ──────────────────────────────────────────────
    print("\n[4/5] Processing & mixing...")
    
    sub_bus = apply_lp(sub_bus[:total_samples], 120, sr)
    sub_bus = np.tanh(sub_bus * 1.3)
    
    reese_bus = apply_hp(reese_bus[:total_samples], 80, sr)
    reese_bus = np.tanh(reese_bus * 1.5)
    
    snare_bus = comb_reverb(snare_bus[:total_samples], sr, decay=0.1, delays_ms=(15,))
    
    scan_L = comb_reverb(scan_L, sr, decay=0.15, delays_ms=(30, 45))
    scan_R = comb_reverb(scan_R, sr, decay=0.15, delays_ms=(45, 60))
    
    pad_bus_L = comb_reverb(pad_bus_L[:total_samples], sr, decay=0.3, delays_ms=(60, 75))
    pad_bus_R = comb_reverb(pad_bus_R[:total_samples], sr, decay=0.3, delays_ms=(75, 90))
    
    mel_bus = comb_reverb(mel_bus[:total_samples], sr, decay=0.3, delays_ms=(50, 70))
    arp_bus = comb_reverb(arp_bus[:total_samples], sr, decay=0.15, delays_ms=(25,))
    
    # ─── Stereo mix ──────────────────────────────────────────────────
    print("  Master processing...")
    
    def trim(buf):
        return buf[:total_samples] if len(buf) >= total_samples else np.pad(buf, (0, total_samples - len(buf)))
    
    mix_L = np.zeros(total_samples)
    mix_R = np.zeros(total_samples)
    
    # Center: kick, snare, sub
    mix_L += trim(kick_bus) * 1.0;     mix_R += trim(kick_bus) * 1.0
    mix_L += trim(snare_bus) * 0.9;    mix_R += trim(snare_bus) * 0.9
    mix_L += trim(sub_bus) * 0.85;     mix_R += trim(sub_bus) * 0.85
    
    # Slightly panned
    mix_L += trim(hat_bus) * 0.6;      mix_R += trim(hat_bus) * 0.45
    mix_L += trim(click_bus) * 0.4;    mix_R += trim(click_bus) * 0.35
    mix_L += trim(reese_bus) * 0.55;   mix_R += trim(reese_bus) * 0.55
    
    # Stereo elements
    mix_L += trim(pad_bus_L);          mix_R += trim(pad_bus_R)
    
    # Scanline spectrogram (the star of v4)
    mix_L += trim(scan_L) * scanline_gain
    mix_R += trim(scan_R) * scanline_gain
    
    # Arp + melody
    mix_L += trim(arp_bus) * 0.4;      mix_R += trim(arp_bus) * 0.35
    mix_L += trim(mel_bus) * 0.45;     mix_R += trim(mel_bus) * 0.5
    
    # ─── Master processing ───────────────────────────────────────────
    stereo = np.column_stack([mix_L, mix_R])
    
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.95
    
    # Compression
    threshold = 0.6
    ratio = 3.0
    for ch in range(2):
        above = np.abs(stereo[:, ch]) > threshold
        stereo[above, ch] = np.sign(stereo[above, ch]) * (
            threshold + (np.abs(stereo[above, ch]) - threshold) / ratio
        )
    
    stereo = np.clip(stereo, -0.98, 0.98)
    
    # Sanitize NaN/inf
    stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.98, neginf=-0.98)
    stereo = stereo.astype(np.float64)
    
    # ─── Write WAV ───────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_v4.wav"
    
    print(f"\n[5/5] Writing WAV: {output_path}")
    print(f"  Shape: {stereo.shape}, dtype: {stereo.dtype}, range: [{np.min(stereo):.3f}, {np.max(stereo):.3f}]")
    sf.write(output_path, stereo, sr, subtype='FLOAT')
    
    fsize = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"  Output:   {output_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Size:     {fsize / 1024 / 1024:.1f} MB")
    print(f"  Format:   {sr}Hz / 24-bit / Stereo")
    print(f"{'='*60}")
    
    return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="video2dnb v4 — Scanline spectrogram DnB renderer")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--scanline-gain", type=float, default=0.55,
                        help="Scanline spectrogram mix level (default: 0.55)")
    parser.add_argument("--sample-rate", type=int, default=SR)
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}"); sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  VIDEO2DNB v4 — SCANLINE SPECTROGRAM")
    print(f"  Video: {args.video}")
    print(f"  BPM:   {args.bpm}")
    print(f"  SR:    {args.sample_rate}")
    print(f"  Scanline gain: {args.scanline_gain}")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    render_video_to_dnb(
        args.video, args.bpm, args.output,
        scanline_gain=args.scanline_gain,
        sr=args.sample_rate,
    )
    print(f"  Render time: {time.time() - t0:.1f}s\n")

if __name__ == "__main__":
    main()
