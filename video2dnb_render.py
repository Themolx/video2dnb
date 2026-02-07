#!/usr/bin/env python3
"""
video2dnb_render.py - Video to DnB audio renderer with image sonification.

Reads every frame of a video and synthesizes DnB audio where you can FEEL
the image. Pixel data directly drives musical parameters while a spectral
sonification layer creates a continuous texture from the visual content.

Layers:
  - Kick, snare, ghost snares, hi-hats (synthesized, pattern-based)
  - Sub bass (sine), mid bass/reese (detuned saws), chord pad
  - Arp lead, sparse melody with delay
  - Spectral sonification texture (frame pixels → frequency magnitudes → iSTFT)

Frame → Audio Mapping:
  - Brightness → overall amplitude envelope, bass weight
  - Edge density → drum complexity, hat density, ghost snare activity
  - Motion → energy spikes, extra kicks, drum fills
  - Saturation → filter resonance on reese bass
  - Red channel → sub bass intensity
  - Green channel → pad/atmosphere gain
  - Blue channel → hi-hat brightness
  - Warm/cool ratio → chord progression selection
  - Dominant hue → arp/melody note selection
  - Full grayscale frame → spectral sonification frequency magnitudes

Usage:
    python video2dnb_render.py <video_path> [--bpm 174] [--output out.wav]
    python video2dnb_render.py video.mp4 --no-sonification  # DnB only
    python video2dnb_render.py video.mp4 --sonification-gain 0.5  # louder texture
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
DEFAULT_BPM = 174

# Note frequencies (Hz)
NOTES = {
    'c1': 32.70, 'eb1': 38.89, 'f1': 43.65, 'ab1': 51.91, 'bb1': 58.27,
    'c2': 65.41, 'eb2': 77.78, 'f2': 87.31, 'g2': 98.00, 'ab2': 103.83, 'bb2': 116.54,
    'c3': 130.81, 'eb3': 155.56, 'f3': 174.61, 'g3': 196.00, 'ab3': 207.65, 'bb3': 233.08,
    'c4': 261.63, 'eb4': 311.13, 'f4': 349.23, 'g4': 392.00, 'ab4': 415.30, 'bb4': 466.16,
    'c5': 523.25, 'eb5': 622.25, 'g5': 783.99,
}

# Chord palette: (name, sub_note, mid_note, pad_notes, arp_notes)
CHORDS = [
    ("Cm",   'c1',  'c2',  ['c3','eb3','g3'],         ['c4','eb4','g4','c5']),
    ("Cm7",  'c1',  'c2',  ['c3','eb3','g3','bb3'],   ['c4','eb4','g4','bb4']),
    ("Ab",   'ab1', 'ab2', ['ab3','c4','eb4'],         ['ab4','c5','eb5']),
    ("Fm",   'f1',  'f2',  ['f3','ab3','c4'],          ['f4','ab4','c5']),
    ("Eb",   'eb1', 'eb2', ['eb3','g3','bb3'],         ['eb4','g4','bb4']),
    ("Bbm",  'bb1', 'bb2', ['bb3','eb4','g4'],         ['bb4','eb5','g5']),
]

PROGRESSIONS = [
    [0, 0, 2, 2],    # Cm → Cm → Ab → Ab (dark)
    [0, 0, 2, 3],    # Cm → Cm → Ab → Fm (tension)
    [0, 1, 2, 3],    # Cm → Cm7 → Ab → Fm (journey)
    [1, 4, 2, 3],    # Cm7 → Eb → Ab → Fm (euphoric)
    [2, 3, 0, 0],    # Ab → Fm → Cm → Cm (resolving)
    [3, 2, 0, 1],    # Fm → Ab → Cm → Cm7 (building)
]

# 16-step drum patterns indexed by density (1-5)
KICK_PAT = {
    1: [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    2: [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
    3: [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
    4: [1,0,0,1, 0,0,1,0, 0,0,0,0, 0,0,0,0],
    5: [1,0,0,1, 0,0,1,0, 0,0,1,1, 0,0,0,0],
}
SNARE_PAT = {
    1: [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
    2: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
    3: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
    4: [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
    5: [0,0,0,0, 1,0,1,0, 0,0,0,1, 1,0,1,0],
}
GHOST_PAT = {
    1: [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    2: [0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    3: [0,0,1,0, 0,0,0,1, 0,0,0,0, 0,1,0,0],
    4: [0,1,0,0, 0,0,1,0, 1,0,0,0, 0,1,0,0],
    5: [0,1,0,1, 0,0,1,1, 0,1,0,0, 0,1,0,1],
}
HAT_PAT = {
    1: [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
    2: [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    3: [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    4: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    5: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
}
HAT_VEL = {
    1: [0.3]*16,
    2: [0.5,0.2,0.35,0.2]*4,
    3: [0.6,0.25,0.4,0.25]*4,
    4: [0.7,0.25,0.45,0.3, 0.65,0.25,0.45,0.3]*2,
    5: [0.8,0.3,0.5,0.35, 0.75,0.3,0.5,0.35]*2,
}
BASS_PAT = {
    1: [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    2: [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
    3: [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,0],
    4: [1,0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
    5: [1,0,1,0, 0,1,1,0, 1,0,1,0, 0,1,0,0],
}


# ─── Synthesis Primitives ─────────────────────────────────────────────────────

def make_kick(sr=SAMPLE_RATE, duration=0.15, gain=1.0):
    """Punchy DnB kick: sine sweep 150→40Hz + transient click."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    pitch_env = 150 * np.exp(-t * 40) + 40
    phase = np.cumsum(2 * np.pi * pitch_env / sr)
    osc = np.sin(phase)
    amp_env = np.exp(-t * 12)
    click = np.exp(-t * 200) * 0.5
    kick = (osc * amp_env + click) * gain
    return np.tanh(kick * 1.5) * 0.9


def make_snare(sr=SAMPLE_RATE, duration=0.1, gain=1.0):
    """Tight DnB snare: bandpassed noise + 200Hz tone body."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    noise_env = np.exp(-t * 20)
    sos = signal.butter(2, [2000, 8000], btype='band', fs=sr, output='sos')
    noise_filtered = signal.sosfilt(sos, noise) * noise_env
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 30)
    snare = (noise_filtered * 0.7 + tone * 0.5) * gain
    return np.tanh(snare * 2) * 0.7


def make_ghost(sr=SAMPLE_RATE, gain=0.15):
    """Quiet ghost snare hit."""
    return make_snare(sr=sr, duration=0.06, gain=gain)


def make_hihat(sr=SAMPLE_RATE, gain=0.3, is_open=False):
    """Hi-hat from highpassed noise."""
    dur = 0.15 if is_open else 0.04
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    decay = 8 if is_open else 40
    env = np.exp(-t * decay)
    sos = signal.butter(2, 6000, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, noise) * env * gain


def make_saw(freq, n_samples, sr=SAMPLE_RATE, num_harmonics=15, gain=1.0):
    """Bandlimited sawtooth wave via additive synthesis."""
    t = np.arange(n_samples) / sr
    wave = np.zeros(n_samples)
    for k in range(1, num_harmonics + 1):
        if k * freq > sr / 2:
            break
        wave += np.sin(2 * np.pi * k * freq * t) / k
    return wave * gain * (2.0 / np.pi)


def apply_lp(audio, cutoff_hz, sr=SAMPLE_RATE):
    """Lowpass filter (safe clamping)."""
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype='low', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)


def apply_hp(audio, cutoff_hz, sr=SAMPLE_RATE):
    """Highpass filter (safe clamping)."""
    cutoff_hz = np.clip(cutoff_hz, 20, sr / 2 - 100)
    sos = signal.butter(2, cutoff_hz, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)


def comb_reverb(audio, sr=SAMPLE_RATE, decay=0.3, delays_ms=(23, 37, 49, 61)):
    """Simple comb-filter reverb."""
    out = audio.copy()
    for d_ms in delays_ms:
        d = int(d_ms * sr / 1000)
        if d < len(audio):
            delayed = np.zeros_like(audio)
            delayed[d:] = audio[:-d]
            out += delayed * decay * 0.25
    return out


def env_adsr(n, sr=SAMPLE_RATE, a=0.01, d=0.05, s=0.7, r=0.05):
    """ADSR envelope in seconds."""
    ai, di, ri = int(a*sr), int(d*sr), int(r*sr)
    env = np.zeros(n)
    pos = 0
    seg = min(ai, n)
    if seg > 0:
        env[pos:pos+seg] = np.linspace(0, 1, seg)
        pos += seg
    seg = min(di, n - pos)
    if seg > 0:
        env[pos:pos+seg] = np.linspace(1, s, seg)
        pos += seg
    sus_len = max(0, n - pos - ri)
    if sus_len > 0:
        env[pos:pos+sus_len] = s
        pos += sus_len
    seg = min(ri, n - pos)
    if seg > 0:
        env[pos:pos+seg] = np.linspace(s, 0, seg)
    return env


def place(buf, sound, pos):
    """Mix sound into buffer at sample position."""
    end = min(pos + len(sound), len(buf))
    length = end - pos
    if length > 0 and pos >= 0:
        buf[pos:end] += sound[:length]


# ─── Video Analysis ───────────────────────────────────────────────────────────

def analyze_frame(frame, prev_frame=None):
    """Analyze a video frame → dict of visual parameters + gray image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = {}

    r['brightness'] = float(np.mean(gray)) / 255.0
    r['contrast'] = min(float(np.std(gray)) / 128.0, 1.0)
    r['red'] = float(np.mean(frame[:, :, 2])) / 255.0
    r['green'] = float(np.mean(frame[:, :, 1])) / 255.0
    r['blue'] = float(np.mean(frame[:, :, 0])) / 255.0
    r['saturation'] = float(np.mean(hsv[:, :, 1])) / 255.0

    edges = cv2.Canny(gray, 50, 150)
    r['edge_density'] = float(np.sum(edges > 0)) / float(edges.size)

    warm = r['red'] + r['green'] * 0.3
    cool = r['blue'] + r['green'] * 0.3
    total = warm + cool
    r['warmth'] = warm / total if total > 0 else 0.5

    sat_mask = hsv[:, :, 1] > 20
    if np.any(sat_mask):
        hue = float(np.median(hsv[:, :, 0][sat_mask])) * 2.0
        r['hue'] = hue / 360.0  # normalized 0-1
    else:
        r['hue'] = 0.0

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        r['motion'] = float(np.mean(diff)) / 255.0
    else:
        r['motion'] = 0.0

    # Spectral content: image rows → frequency band amplitudes
    freqs, amps = _spectral_from_rows(gray)
    r['spectral_freqs'] = freqs
    r['spectral_amps'] = amps

    # Composite energy
    r['energy'] = (0.25 * r['brightness'] + 0.25 * r['motion'] +
                   0.20 * r['edge_density'] + 0.15 * r['contrast'] +
                   0.15 * r['saturation'])

    return r, gray


def _spectral_from_rows(gray, num_bands=64, freq_min=80, freq_max=8000):
    """Image rows → log-spaced frequency band amplitudes."""
    h, w = gray.shape
    indices = np.linspace(0, h - 1, num_bands).astype(int)
    bands = gray[indices, :]
    amps = np.mean(bands, axis=1) / 255.0
    amps = amps[::-1]  # bottom=low freq
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), num_bands)
    return freqs, amps


def analyze_video(video_path):
    """Analyze all video frames. Returns (analyses, fps, duration, total_frames)."""
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
        analysis, _ = analyze_frame(small, prev_small)
        analyses.append(analysis)
        prev_small = small
        idx += 1
        if idx % 30 == 0:
            pct = idx / total_frames * 100
            print(f"\r  Analyzing [{'#' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)

    cap.release()
    print(f"\r  Analyzing [{'#' * 40}] 100%")
    print(f"  {len(analyses)} frames analyzed")

    # Rank-normalize energy for better spread
    energies = [a['energy'] for a in analyses]
    e_min, e_max = min(energies), max(energies)
    e_range = e_max - e_min if e_max > e_min else 1.0
    for a in analyses:
        a['energy_norm'] = (a['energy'] - e_min) / e_range

    return analyses, fps, duration, total_frames


# ─── Spectral Sonification (iSTFT-based) ─────────────────────────────────────

def render_sonification(analyses, fps, duration, sr=SAMPLE_RATE,
                        freq_lo=60.0, freq_hi=8000.0, bass_boost=2.0):
    """
    Spectral sonification: frame pixels → frequency magnitudes → iSTFT audio.

    Each frame's row-averaged brightness becomes a frequency magnitude slice.
    This creates a continuous texture that directly represents the visual content.
    """
    total_samples = int(sr * duration)
    n_frames = len(analyses)

    # FFT parameters
    n_fft = 2048
    hop = max(64, total_samples // n_frames)
    n_fft_bins = n_fft // 2 + 1
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Precompute frequency bin mapping (video freq bands → FFT bins)
    sample_freqs = np.logspace(np.log10(freq_lo), np.log10(freq_hi), 64)
    fft_bin_map = np.round(sample_freqs / (sr / n_fft)).astype(int)
    fft_bin_map = np.clip(fft_bin_map, 0, n_fft_bins - 1)

    # Build magnitude spectrogram from video frames
    magnitudes = np.zeros((n_fft_bins, n_frames))
    for i, a in enumerate(analyses):
        amps = a['spectral_amps']  # 64 bands
        for j, (fft_bin, amp) in enumerate(zip(fft_bin_map, amps)):
            magnitudes[fft_bin, i] = max(magnitudes[fft_bin, i], amp)

    # Bass boost
    if bass_boost > 1.0:
        for b in range(n_fft_bins):
            f = fft_freqs[b]
            if f < 300:
                magnitudes[b, :] *= bass_boost
            elif f < 600:
                magnitudes[b, :] *= 1.0 + (bass_boost - 1.0) * (600 - f) / 300

    # Random phase
    phases = np.random.uniform(-np.pi, np.pi, magnitudes.shape)
    stft = magnitudes * np.exp(1j * phases)

    # iSTFT via overlap-add
    audio = np.zeros(total_samples + n_fft)
    window = np.hanning(n_fft)

    for i in range(n_frames):
        # Build full symmetric FFT frame
        full_fft = np.zeros(n_fft, dtype=complex)
        full_fft[:n_fft_bins] = stft[:, i]
        if n_fft_bins > 2:
            full_fft[n_fft_bins:] = np.conj(stft[-2:0:-1, i])

        frame_audio = np.real(np.fft.ifft(full_fft)) * window
        pos = i * hop
        end = min(pos + n_fft, len(audio))
        audio[pos:end] += frame_audio[:end - pos]

    audio = audio[:total_samples]

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return audio


# ─── Additive Texture (img2sound-style per-frame oscillators) ─────────────────

def render_additive_texture(analyses, fps, duration, sr=SAMPLE_RATE):
    """
    img2sound-style additive synthesis: each image row drives a sine oscillator.
    Phase-continuous across frames for smooth transitions.
    Returns stereo (L, R) tuple.
    """
    total_samples = int(sr * duration)
    n_frames = len(analyses)
    samples_per_frame = total_samples // n_frames if n_frames > 0 else 1

    n_bands = 64
    phases = np.zeros(n_bands)

    out_L = np.zeros(total_samples + sr)
    out_R = np.zeros(total_samples + sr)

    for fi in range(n_frames):
        a = analyses[fi]
        freqs = a['spectral_freqs']
        amps = a['spectral_amps']
        green = a['green']
        brightness = a['brightness']

        chunk_start = fi * samples_per_frame
        chunk_end = min(chunk_start + samples_per_frame, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0:
            continue

        texture = np.zeros(chunk_len)
        for i in range(n_bands):
            if amps[i] < 0.04:
                continue
            freq = freqs[i]
            if freq > sr / 2:
                continue
            phase_inc = 2 * np.pi * freq / sr
            ph = phases[i] + np.cumsum(np.ones(chunk_len) * phase_inc)
            texture += np.sin(ph) * amps[i]
            phases[i] = ph[-1] % (2 * np.pi)

        peak = np.max(np.abs(texture))
        if peak > 0:
            texture /= peak

        gain = 0.05 + green * 0.1 + brightness * 0.05

        out_L[chunk_start:chunk_end] += texture * gain * 0.7
        out_R[chunk_start:chunk_end] += texture * gain * 0.5
        # Haas delay for width
        d = int(0.005 * sr)
        if chunk_end + d <= len(out_R):
            out_R[chunk_start + d:chunk_end + d] += texture * gain * 0.3

    return out_L[:total_samples], out_R[:total_samples]


# ─── Main Rendering Engine ────────────────────────────────────────────────────

def render_video_to_dnb(video_path, bpm=174, output_path=None,
                        no_sonification=False, sonification_gain=0.35,
                        sr=SAMPLE_RATE):
    """Main render: video → stereo DnB WAV."""

    # ─── Analyze ──────────────────────────────────────────────────────────
    print("[1/5] Analyzing video...")
    analyses, fps, duration, total_frames = analyze_video(video_path)

    total_samples = int(duration * sr)
    tail = sr  # 1s tail for reverb etc
    buf_len = total_samples + tail

    beat_dur = 60.0 / bpm
    s16 = int(beat_dur / 4 * sr)  # samples per 16th note
    s16_sec = beat_dur / 4
    total_bars = int(duration / (beat_dur * 4))

    print(f"\n  BPM: {bpm}, 16th={s16_sec*1000:.1f}ms, bars={total_bars}")

    num_frames = len(analyses)

    def fa_at(sample_pos):
        """Get frame analysis at sample position."""
        t = sample_pos / sr
        idx = min(int(t * fps), num_frames - 1)
        return analyses[max(0, idx)]

    # ─── Arrangement envelope ─────────────────────────────────────────────
    # Intro ramp (first 4 bars), outro fade (last 4 bars)
    intro_bars = max(1, min(4, total_bars // 4))
    outro_bars = max(1, min(4, total_bars // 4))

    def arr_gain(sample_pos):
        """Arrangement gain: intro ramp and outro fade."""
        t = sample_pos / sr
        bar = t / (beat_dur * 4)
        if bar < intro_bars:
            return 0.2 + 0.8 * (bar / intro_bars)
        elif total_bars > outro_bars and bar > total_bars - outro_bars:
            remaining = max(0, total_bars - bar)
            return 0.2 + 0.8 * (remaining / outro_bars)
        return 1.0

    # ─── Allocate stereo buses ────────────────────────────────────────────
    print("\n[2/5] Rendering DnB layers...")
    kick_bus = np.zeros(buf_len)
    snare_bus = np.zeros(buf_len)
    hat_bus = np.zeros(buf_len)
    sub_bus = np.zeros(buf_len)
    reese_bus = np.zeros(buf_len)
    pad_bus_L = np.zeros(buf_len)
    pad_bus_R = np.zeros(buf_len)
    arp_bus = np.zeros(buf_len)
    mel_bus = np.zeros(buf_len)

    # Pre-generate one-shot sounds
    kick_snd = make_kick(sr=sr)
    snare_snd = make_snare(sr=sr)
    ghost_snd = make_ghost(sr=sr)

    # ─── Drums ────────────────────────────────────────────────────────────
    print("  Drums...")
    pos = 0
    beat_idx = 0

    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)
        e = fa['energy_norm']
        ag = arr_gain(pos)

        # Density levels
        dd = int(np.clip(round(e * 4) + 1, 1, 5))  # drum density
        hd = int(np.clip(round((0.4*fa['edge_density'] + 0.3*fa['contrast'] + 0.3*fa['saturation']) * 4) + 1, 1, 5))

        # KICK
        if KICK_PAT.get(dd, KICK_PAT[3])[s16_in_bar]:
            g = (0.7 + e * 0.3) * ag
            place(kick_bus, kick_snd * g, pos)

        # SNARE
        if SNARE_PAT.get(dd, SNARE_PAT[3])[s16_in_bar]:
            g = (0.6 + e * 0.4) * ag
            place(snare_bus, snare_snd * g, pos)
        elif GHOST_PAT.get(max(1, dd-1), GHOST_PAT[1])[s16_in_bar]:
            g = (0.1 + fa['edge_density'] * 0.15) * ag
            place(snare_bus, ghost_snd * g, pos)

        # HATS
        if HAT_PAT.get(hd, HAT_PAT[3])[s16_in_bar]:
            is_open = (s16_in_bar in (6, 14) and e > 0.5)
            vel = HAT_VEL.get(hd, HAT_VEL[3])[s16_in_bar % len(HAT_VEL.get(hd, HAT_VEL[3]))]
            hat = make_hihat(sr=sr, gain=vel * ag, is_open=is_open)
            place(hat_bus, hat, pos)

        beat_idx += 1
        pos += s16

    # ─── Sub bass ─────────────────────────────────────────────────────────
    print("  Sub bass...")
    sub_phase = 0.0

    for chunk_start in range(0, total_samples, s16):
        chunk_end = min(chunk_start + s16, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0:
            continue

        fa = fa_at(chunk_start)
        s16_in_bar = (chunk_start // s16) % 16

        # Chord selection from warmth
        warmth = fa['warmth']
        prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS) - 1)), 0, len(PROGRESSIONS) - 1))
        bar_in_prog = ((chunk_start // s16) // 16) % 4
        chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
        chord = CHORDS[chord_idx]

        # Bass density from energy
        bd = int(np.clip(round(fa['energy_norm'] * 4) + 1, 1, 5))

        if BASS_PAT.get(bd, BASS_PAT[3])[s16_in_bar]:
            freq = NOTES.get(chord[1], 32.7)
            # Slight wobble from red channel
            wobble = 1.0 + np.sin(chunk_start / sr * np.pi) * fa['red'] * 0.02
            freq *= wobble

            t = np.arange(chunk_len) / sr
            sub_phase_arr = sub_phase + np.cumsum(np.ones(chunk_len) * 2 * np.pi * freq / sr)
            sub = np.sin(sub_phase_arr)
            sub_phase = sub_phase_arr[-1] if chunk_len > 0 else sub_phase

            sub_gain = (0.3 + fa['red'] * 0.5) * (0.5 + fa['brightness'] * 0.5)
            sub_env = np.exp(-t * 4) * 0.7 + 0.3
            ag = arr_gain(chunk_start)

            sub_bus[chunk_start:chunk_end] += sub * sub_gain * sub_env * ag

    # ─── Reese bass ───────────────────────────────────────────────────────
    print("  Reese bass...")

    for chunk_start in range(0, total_samples, s16):
        chunk_end = min(chunk_start + s16, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0:
            continue

        fa = fa_at(chunk_start)
        s16_in_bar = (chunk_start // s16) % 16

        warmth = fa['warmth']
        prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS) - 1)), 0, len(PROGRESSIONS) - 1))
        bar_in_prog = ((chunk_start // s16) // 16) % 4
        chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
        chord = CHORDS[chord_idx]

        bd = int(np.clip(round(fa['energy_norm'] * 4) + 1, 1, 5))

        if BASS_PAT.get(bd, BASS_PAT[3])[s16_in_bar]:
            freq = NOTES.get(chord[2], 65.41)

            saw1 = make_saw(freq * 1.003, chunk_len, sr=sr, gain=0.3)
            saw2 = make_saw(freq * 0.997, chunk_len, sr=sr, gain=0.3)
            reese = saw1 + saw2

            cutoff = 200 + fa['saturation'] * 2000 + fa['brightness'] * 1500
            reese = apply_lp(reese, cutoff, sr)

            t = np.arange(chunk_len) / sr
            reese_env = np.exp(-t * 6) * 0.7 + 0.3
            reese_gain = (0.1 + fa['edge_density'] * 0.3) * arr_gain(chunk_start)

            reese_bus[chunk_start:chunk_end] += reese * reese_gain * reese_env

    # ─── Chord pad ────────────────────────────────────────────────────────
    print("  Chord pad...")
    bar_samples = s16 * 16

    for bar_start in range(0, total_samples, bar_samples):
        bar_end = min(bar_start + bar_samples, total_samples)
        bar_len = bar_end - bar_start
        if bar_len <= 0:
            continue

        fa = fa_at(bar_start)
        warmth = fa['warmth']
        prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS) - 1)), 0, len(PROGRESSIONS) - 1))
        bar_in_prog = (bar_start // bar_samples) % 4
        chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
        chord = CHORDS[chord_idx]
        pad_notes = chord[3]

        cutoff = 200 + fa['brightness'] * 800 + fa['saturation'] * 400

        pad = np.zeros(bar_len)
        for note_name in pad_notes:
            freq = NOTES.get(note_name, 200)
            saw1 = make_saw(freq * 1.002, bar_len, sr=sr, gain=0.15, num_harmonics=8)
            saw2 = make_saw(freq * 0.998, bar_len, sr=sr, gain=0.15, num_harmonics=8)
            pad += saw1 + saw2

        pad /= max(len(pad_notes), 1)
        pad = apply_lp(pad, cutoff, sr)

        # Slow attack/release
        attack_n = int(0.3 * sr)
        release_n = int(0.5 * sr)
        env = np.ones(bar_len)
        if bar_len > attack_n:
            env[:attack_n] = np.linspace(0, 1, attack_n)
        if bar_len > release_n:
            env[-release_n:] = np.linspace(1, 0, release_n)

        pad_gain = (0.04 + fa['green'] * 0.06) * arr_gain(bar_start)

        pad_bus_L[bar_start:bar_end] += pad * pad_gain * env
        pad_bus_R[bar_start:bar_end] += pad * pad_gain * env * 0.8

    # ─── Arp lead ─────────────────────────────────────────────────────────
    print("  Arp lead...")
    arp_counter = 0
    pos = 0
    beat_idx = 0

    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)
        e = fa['energy_norm']

        # Arp density based on energy
        play = False
        if e > 0.6:
            play = (s16_in_bar % 2 == 0)
        elif e > 0.35:
            play = (s16_in_bar % 4 == 0)
        else:
            play = (s16_in_bar % 8 == 0)

        if play:
            warmth = fa['warmth']
            prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS) - 1)), 0, len(PROGRESSIONS) - 1))
            bar_in_prog = (beat_idx // 16) % 4
            chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
            chord = CHORDS[chord_idx]
            arp_notes = chord[4]

            note_name = arp_notes[arp_counter % len(arp_notes)]
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

    # ─── Melody ───────────────────────────────────────────────────────────
    print("  Melody...")
    pos = 0
    beat_idx = 0

    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)

        if s16_in_bar in (2, 10):
            warmth = fa['warmth']
            prog_idx = int(np.clip(round(warmth * (len(PROGRESSIONS) - 1)), 0, len(PROGRESSIONS) - 1))
            bar_in_prog = (beat_idx // 16) % 4
            chord_idx = PROGRESSIONS[prog_idx][bar_in_prog]
            chord = CHORDS[chord_idx]
            arp_notes = chord[4]

            note_idx = int(fa['hue'] * len(arp_notes)) % len(arp_notes)
            note_name = arp_notes[note_idx]
            freq = NOTES.get(note_name, 440)

            note_len = int(s16_sec * 3 * sr)
            t = np.arange(note_len) / sr
            # Triangle wave
            note = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
            note *= env_adsr(note_len, sr, a=0.01, d=0.15, s=0.2, r=0.2)
            note = apply_lp(note, 2000, sr)

            gain = 0.07 * arr_gain(pos)
            place(mel_bus, note * gain, pos)

            # Dotted-8th delay echoes
            delay_s = int(s16_sec * 6 * sr)
            place(mel_bus, note * gain * 0.35, pos + delay_s)
            place(mel_bus, note * gain * 0.12, pos + delay_s * 2)

        beat_idx += 1
        pos += s16

    # ─── Sonification layers ─────────────────────────────────────────────
    if not no_sonification:
        print("\n[3/5] Rendering sonification layers...")

        print("  iSTFT spectral sonification...")
        son_istft = render_sonification(analyses, fps, duration, sr=sr)

        print("  Additive texture (img2sound-style)...")
        tex_L, tex_R = render_additive_texture(analyses, fps, duration, sr=sr)
    else:
        print("\n[3/5] Sonification skipped")
        son_istft = np.zeros(total_samples)
        tex_L = np.zeros(total_samples)
        tex_R = np.zeros(total_samples)

    # ─── Bus processing ──────────────────────────────────────────────────
    print("\n[4/5] Processing & mixing...")

    # Sub: keep it tight
    sub_bus = apply_lp(sub_bus[:total_samples], 120, sr)
    sub_bus = np.tanh(sub_bus * 1.3)

    # Reese: don't clash with sub
    reese_bus = apply_hp(reese_bus[:total_samples], 80, sr)
    reese_bus = np.tanh(reese_bus * 1.5)

    # Snare: room
    snare_bus = comb_reverb(snare_bus[:total_samples], sr, decay=0.1, delays_ms=(15,))

    # Texture: reverb
    tex_L = comb_reverb(tex_L, sr, decay=0.2, delays_ms=(40, 55))
    tex_R = comb_reverb(tex_R, sr, decay=0.2, delays_ms=(55, 70))

    # Pad: reverb
    pad_bus_L = comb_reverb(pad_bus_L[:total_samples], sr, decay=0.3, delays_ms=(60, 75))
    pad_bus_R = comb_reverb(pad_bus_R[:total_samples], sr, decay=0.3, delays_ms=(75, 90))

    # Melody: reverb
    mel_bus = comb_reverb(mel_bus[:total_samples], sr, decay=0.3, delays_ms=(50, 70))

    # Arp: slight room
    arp_bus = comb_reverb(arp_bus[:total_samples], sr, decay=0.15, delays_ms=(25,))

    # ─── Stereo mix ──────────────────────────────────────────────────────
    def trim(buf):
        return buf[:total_samples] if len(buf) >= total_samples else np.pad(buf, (0, total_samples - len(buf)))

    mix_L = np.zeros(total_samples)
    mix_R = np.zeros(total_samples)

    # Center: kick, snare, sub
    mix_L += trim(kick_bus) * 1.0;    mix_R += trim(kick_bus) * 1.0
    mix_L += trim(snare_bus) * 0.9;   mix_R += trim(snare_bus) * 0.9
    mix_L += trim(sub_bus) * 0.85;    mix_R += trim(sub_bus) * 0.85

    # Slightly panned
    mix_L += trim(hat_bus) * 0.6;     mix_R += trim(hat_bus) * 0.45
    mix_L += trim(reese_bus) * 0.55;  mix_R += trim(reese_bus) * 0.55

    # Stereo
    mix_L += trim(pad_bus_L);         mix_R += trim(pad_bus_R)
    mix_L += trim(tex_L) * sonification_gain
    mix_R += trim(tex_R) * sonification_gain

    # Sonification (mono center)
    mix_L += trim(son_istft) * sonification_gain * 0.5
    mix_R += trim(son_istft) * sonification_gain * 0.5

    # Arp + melody (slight stereo)
    mix_L += trim(arp_bus) * 0.4;     mix_R += trim(arp_bus) * 0.35
    mix_L += trim(mel_bus) * 0.45;    mix_R += trim(mel_bus) * 0.5

    # ─── Master processing ────────────────────────────────────────────────
    print("  Master processing...")
    stereo = np.column_stack([mix_L, mix_R])

    # Normalize
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.95

    # Simple compression
    threshold = 0.6
    ratio = 3.0
    for ch in range(2):
        above = np.abs(stereo[:, ch]) > threshold
        stereo[above, ch] = np.sign(stereo[above, ch]) * (
            threshold + (np.abs(stereo[above, ch]) - threshold) / ratio
        )

    # Final limiter
    stereo = np.clip(stereo, -0.98, 0.98)

    # ─── Write WAV ────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"{base}_dnb.wav")

    print(f"\n[5/5] Writing WAV: {output_path}")
    sf.write(output_path, stereo, sr, subtype='PCM_24')

    fsize = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"  Output:   {output_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Size:     {fsize / 1024 / 1024:.1f} MB")
    print(f"  Format:   {sr}Hz / 24-bit / Stereo")
    print(f"{'='*60}")

    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render video to DnB audio with image sonification")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM, help="BPM (default: 174)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output WAV path")
    parser.add_argument("--no-sonification", action="store_true",
                        help="Skip sonification layers (DnB only)")
    parser.add_argument("--sonification-gain", type=float, default=0.35,
                        help="Sonification mix level 0-1 (default: 0.35)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help="Sample rate (default: 48000)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  VIDEO2DNB RENDERER")
    print(f"  Video: {args.video}")
    print(f"  BPM:   {args.bpm}")
    print(f"  SR:    {args.sample_rate}")
    print(f"  Sonification: {'OFF' if args.no_sonification else f'ON (gain={args.sonification_gain})'}")
    print(f"{'='*60}\n")

    t0 = time.time()
    render_video_to_dnb(
        args.video, args.bpm, args.output,
        no_sonification=args.no_sonification,
        sonification_gain=args.sonification_gain,
        sr=args.sample_rate,
    )
    print(f"  Render time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
