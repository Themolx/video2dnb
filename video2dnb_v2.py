#!/usr/bin/env python3
"""
video2dnb_v2.py — Video-driven DnB renderer, v2.

Key difference from v1: the IMAGE drives the rhythm, not a template.
  - No hardcoded drum patterns — hits are triggered by visual features
  - Scene cuts / big frame changes → impacts, filter sweeps, crashes
  - Image columns scanned left→right within each beat to derive rhythm
  - Per-frame continuous modulation of all synth parameters
  - Motion spikes trigger drum fills and risers
  - Spectral texture is primary, not an overlay

Usage:
    python video2dnb_v2.py video.mp4 [--bpm 174] [--output out.wav]
"""

import cv2
import numpy as np
import soundfile as sf
from scipy import signal
import argparse, os, sys, time

SR = 48000

# ─── Notes in C minor ────────────────────────────────────────────────────────
NOTE_HZ = {
    'C1':32.70,'Eb1':38.89,'F1':43.65,'Ab1':51.91,'Bb1':58.27,
    'C2':65.41,'Eb2':77.78,'F2':87.31,'G2':98.0,'Ab2':103.83,'Bb2':116.54,
    'C3':130.81,'Eb3':155.56,'F3':174.61,'G3':196.0,'Ab3':207.65,'Bb3':233.08,
    'C4':261.63,'Eb4':311.13,'F4':349.23,'G4':392.0,'Ab4':415.30,'Bb4':466.16,
    'C5':523.25,'Eb5':622.25,'G5':783.99,
}

# Chord voicings: (sub, mid, pad_freqs)
CHORDS = [
    (NOTE_HZ['C1'],  NOTE_HZ['C2'],  [NOTE_HZ['C3'],NOTE_HZ['Eb3'],NOTE_HZ['G3']]),
    (NOTE_HZ['Ab1'], NOTE_HZ['Ab2'], [NOTE_HZ['Ab3'],NOTE_HZ['C4'],NOTE_HZ['Eb4']]),
    (NOTE_HZ['F1'],  NOTE_HZ['F2'],  [NOTE_HZ['F3'],NOTE_HZ['Ab3'],NOTE_HZ['C4']]),
    (NOTE_HZ['Eb1'], NOTE_HZ['Eb2'], [NOTE_HZ['Eb3'],NOTE_HZ['G3'],NOTE_HZ['Bb3']]),
]

# ─── Synthesis ────────────────────────────────────────────────────────────────

def synth_kick(gain=1.0, pitch_mul=1.0):
    n = int(SR * 0.15)
    t = np.arange(n) / SR
    pitch = (150 * pitch_mul) * np.exp(-t * 40) + 40
    phase = np.cumsum(2 * np.pi * pitch / SR)
    osc = np.sin(phase) * np.exp(-t * 12)
    click = np.exp(-t * 200) * 0.5
    return np.tanh((osc + click) * gain * 1.5) * 0.9

def synth_snare(gain=1.0, tone_hz=200, noise_bw=(2000,8000)):
    n = int(SR * 0.1)
    t = np.arange(n) / SR
    noise = np.random.randn(n) * np.exp(-t * 20)
    sos = signal.butter(2, noise_bw, btype='band', fs=SR, output='sos')
    nf = signal.sosfilt(sos, noise)
    tone = np.sin(2*np.pi*tone_hz*t) * np.exp(-t*30)
    return np.tanh((nf*0.7 + tone*0.5) * gain * 2) * 0.7

def synth_hat(gain=0.3, is_open=False):
    dur = 0.15 if is_open else 0.035
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = np.random.randn(n) * np.exp(-t * (8 if is_open else 50))
    sos = signal.butter(2, 6000, btype='high', fs=SR, output='sos')
    return signal.sosfilt(sos, noise) * gain

def synth_impact(gain=1.0):
    """Low boom + noise burst for scene cuts."""
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    boom = np.sin(2*np.pi*(80*np.exp(-t*8))*t) * np.exp(-t*5) * 0.8
    noise = np.random.randn(n) * np.exp(-t*15) * 0.4
    sos = signal.butter(2, 400, btype='low', fs=SR, output='sos')
    noise = signal.sosfilt(sos, noise)
    return np.tanh((boom + noise) * gain * 2) * 0.9

def synth_riser(duration_s, gain=0.3):
    """Filtered noise sweep rising in pitch."""
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    noise = np.random.randn(n)
    # Sweep filter from 200Hz to 8000Hz
    cutoffs = 200 + (t/duration_s)**2 * 7800
    # Apply time-varying filter via chunked processing
    chunk = SR // 10
    out = np.zeros(n)
    for i in range(0, n, chunk):
        end = min(i+chunk, n)
        cf = np.clip(cutoffs[i], 20, SR/2-100)
        sos = signal.butter(2, cf, btype='low', fs=SR, output='sos')
        out[i:end] = signal.sosfilt(sos, noise[i:end])
    env = np.linspace(0, 1, n)**2
    return out * env * gain

def synth_saw(freq, n_samples, gain=1.0, harmonics=12):
    t = np.arange(n_samples) / SR
    w = np.zeros(n_samples)
    for k in range(1, harmonics+1):
        if k*freq > SR/2: break
        w += np.sin(2*np.pi*k*freq*t) / k
    return w * gain * (2/np.pi)

def apply_lp(audio, cutoff):
    cutoff = np.clip(cutoff, 20, SR/2-100)
    sos = signal.butter(2, cutoff, btype='low', fs=SR, output='sos')
    return signal.sosfilt(sos, audio)

def apply_hp(audio, cutoff):
    cutoff = np.clip(cutoff, 20, SR/2-100)
    sos = signal.butter(2, cutoff, btype='high', fs=SR, output='sos')
    return signal.sosfilt(sos, audio)

def place(buf, snd, pos):
    end = min(pos + len(snd), len(buf))
    l = end - pos
    if l > 0 and pos >= 0:
        buf[pos:end] += snd[:l]

def comb_verb(audio, decay=0.25, delays_ms=(23,37,49)):
    out = audio.copy()
    for d_ms in delays_ms:
        d = int(d_ms * SR / 1000)
        if d < len(audio):
            delayed = np.zeros_like(audio)
            delayed[d:] = audio[:-d]
            out += delayed * decay / len(delays_ms)
    return out


# ─── Video Analysis ──────────────────────────────────────────────────────────

def analyze_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: cannot open '{path}'"); sys.exit(1)
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = total / fps
    print(f"  {total} frames, {fps:.1f} fps, {dur:.1f}s")
    
    analyses = []
    prev = None
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        a = {}
        a['brightness'] = np.mean(gray) / 255.0
        a['contrast'] = min(np.std(gray) / 128.0, 1.0)
        a['red'] = np.mean(small[:,:,2]) / 255.0
        a['green'] = np.mean(small[:,:,1]) / 255.0
        a['blue'] = np.mean(small[:,:,0]) / 255.0
        a['saturation'] = np.mean(hsv[:,:,1]) / 255.0
        
        edges = cv2.Canny(gray, 50, 150)
        a['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Row brightness profile (for spectral synthesis)
        a['row_profile'] = np.mean(gray, axis=1) / 255.0  # 120 values
        
        # Column brightness profile (for rhythm derivation)
        a['col_profile'] = np.mean(gray, axis=0) / 255.0  # 160 values
        
        # Quadrant energies (for spatial drum mapping)
        h, w = gray.shape
        a['quad_tl'] = np.mean(gray[:h//2, :w//2]) / 255.0  # kick zone
        a['quad_tr'] = np.mean(gray[:h//2, w//2:]) / 255.0  # hat zone
        a['quad_bl'] = np.mean(gray[h//2:, :w//2]) / 255.0  # snare zone
        a['quad_br'] = np.mean(gray[h//2:, w//2:]) / 255.0  # bass zone
        
        # Hue
        sat_mask = hsv[:,:,1] > 20
        if np.any(sat_mask):
            a['hue'] = float(np.median(hsv[:,:,0][sat_mask])) / 90.0  # 0-2 range
        else:
            a['hue'] = 0.0
        
        # Motion
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            a['motion'] = np.mean(diff) / 255.0
            a['motion_max'] = np.max(diff) / 255.0
            # Regional motion
            a['motion_top'] = np.mean(diff[:h//2, :]) / 255.0
            a['motion_bot'] = np.mean(diff[h//2:, :]) / 255.0
        else:
            a['motion'] = a['motion_max'] = 0.0
            a['motion_top'] = a['motion_bot'] = 0.0
        
        # Scene cut detection: very high motion = scene change
        a['is_scene_cut'] = a['motion'] > 0.15
        
        prev = gray
        analyses.append(a)
        idx += 1
        if idx % 30 == 0:
            print(f"\r  Analyzing [{('#'*int(idx/total*40)):40s}] {idx/total*100:.0f}%", end="", flush=True)
    
    cap.release()
    print(f"\r  Analyzing [{'#'*40}] 100%")
    
    # Normalize motion to 0-1 range for this video
    motions = [a['motion'] for a in analyses]
    m_max = max(motions) if max(motions) > 0 else 1.0
    for a in analyses:
        a['motion_norm'] = a['motion'] / m_max
    
    return analyses, fps, dur


# ─── Image-Driven Rhythm Generator ──────────────────────────────────────────

def derive_rhythm_from_frame(fa, n_steps=16, prev_fa=None):
    """
    Derive a 16-step rhythm pattern directly from image content.
    No templates — the image IS the rhythm.
    
    Returns dict of boolean arrays for each instrument.
    """
    col = fa['col_profile']  # 160 values
    # Resample column profile to n_steps
    indices = np.linspace(0, len(col)-1, n_steps).astype(int)
    col_steps = col[indices]
    
    motion = fa['motion_norm']
    edge = fa['edge_density']
    brightness = fa['brightness']
    
    # KICK: driven by bottom-left quadrant brightness + column dips
    # Kicks happen at brightness valleys (dark spots = heavy)
    kick_threshold = 0.5 - motion * 0.2  # more motion = easier kicks
    kick = col_steps < kick_threshold
    # Always ensure at least beat 1
    kick[0] = True
    # High motion: add more kicks
    if motion > 0.5:
        kick[int(n_steps*0.6)] = True
    
    # SNARE: driven by edge density + column peaks
    snare_threshold = 0.55 - edge * 0.15
    snare = col_steps > snare_threshold
    # Ensure at least one snare around beat 2-3
    if not any(snare[3:6]):
        snare[4] = True
    # Avoid kick+snare on same step (unless high energy)
    if motion < 0.3:
        snare = snare & ~kick
    
    # HATS: driven by edge density — more edges = more hats
    if edge > 0.15:
        hat = np.ones(n_steps, dtype=bool)  # every 16th
    elif edge > 0.08:
        hat = np.zeros(n_steps, dtype=bool)
        hat[::2] = True  # every 8th
    else:
        hat = np.zeros(n_steps, dtype=bool)
        hat[::4] = True  # every quarter
    
    # Hat velocity from column brightness
    hat_vel = col_steps * 0.6 + 0.2
    
    # GHOST SNARES: from motion
    ghost = np.zeros(n_steps, dtype=bool)
    if motion > 0.3:
        # Place ghosts where column has medium brightness
        mid = (col_steps > 0.3) & (col_steps < 0.6) & ~kick & ~snare
        ghost = mid
    
    return {
        'kick': kick,
        'snare': snare,
        'hat': hat,
        'hat_vel': hat_vel,
        'ghost': ghost,
    }


# ─── Main Renderer ───────────────────────────────────────────────────────────

def render(video_path, bpm=174, output_path=None, son_gain=0.5):
    
    print("[1/4] Analyzing video...")
    analyses, fps, duration = analyze_video(video_path)
    
    total_samples = int(duration * SR)
    buf_len = total_samples + SR  # 1s tail
    n_frames = len(analyses)
    
    beat_dur = 60.0 / bpm
    s16 = int(beat_dur / 4 * SR)
    s16_sec = beat_dur / 4
    
    print(f"  BPM: {bpm}, 16th: {s16_sec*1000:.1f}ms")
    
    def fa_at(sample_pos):
        t = sample_pos / SR
        idx = min(int(t * fps), n_frames - 1)
        return analyses[max(0, idx)]
    
    def fa_interp(sample_pos, key):
        """Interpolated frame parameter."""
        t = sample_pos / SR
        fi = t * fps
        i0 = max(0, min(int(fi), n_frames - 1))
        i1 = min(i0 + 1, n_frames - 1)
        frac = fi - int(fi)
        return analyses[i0][key] * (1-frac) + analyses[i1][key] * frac
    
    # ─── Detect scene cuts and motion spikes ─────────────────────────────
    scene_cuts = []  # sample positions of scene cuts
    motion_spikes = []  # (sample_pos, intensity)
    
    for i, a in enumerate(analyses):
        sample_pos = int(i / fps * SR)
        if a['is_scene_cut']:
            scene_cuts.append(sample_pos)
        if a['motion_norm'] > 0.7:
            motion_spikes.append((sample_pos, a['motion_norm']))
    
    print(f"  Scene cuts: {len(scene_cuts)}, Motion spikes: {len(motion_spikes)}")
    
    # ─── Allocate buses ──────────────────────────────────────────────────
    print("\n[2/4] Synthesizing...")
    kick_bus = np.zeros(buf_len)
    snare_bus = np.zeros(buf_len)
    hat_bus = np.zeros(buf_len)
    sub_bus = np.zeros(buf_len)
    reese_bus = np.zeros(buf_len)
    pad_L = np.zeros(buf_len)
    pad_R = np.zeros(buf_len)
    impact_bus = np.zeros(buf_len)
    spectral_L = np.zeros(buf_len)
    spectral_R = np.zeros(buf_len)
    
    # ─── Drums (image-driven) ────────────────────────────────────────────
    print("  Drums (image-driven rhythm)...")
    pos = 0
    beat_idx = 0
    prev_fa = None
    
    while pos < total_samples:
        s16_in_bar = beat_idx % 16
        fa = fa_at(pos)
        
        # Every bar (16 steps), derive a new rhythm from the current frame
        if s16_in_bar == 0:
            rhythm = derive_rhythm_from_frame(fa, n_steps=16, prev_fa=prev_fa)
            prev_fa = fa
        
        motion = fa['motion_norm']
        
        # KICK
        if rhythm['kick'][s16_in_bar]:
            # Pitch varies with brightness (darker = deeper)
            pitch_mul = 0.8 + fa['brightness'] * 0.4
            gain = 0.7 + motion * 0.4
            place(kick_bus, synth_kick(gain=gain, pitch_mul=pitch_mul), pos)
        
        # SNARE
        if rhythm['snare'][s16_in_bar]:
            # Tone frequency from quadrant
            tone_hz = 150 + fa['quad_bl'] * 150
            gain = 0.5 + fa['edge_density'] * 0.6
            place(snare_bus, synth_snare(gain=gain, tone_hz=tone_hz), pos)
        
        # GHOST
        if rhythm['ghost'][s16_in_bar]:
            place(snare_bus, synth_snare(gain=0.12 + motion*0.08, tone_hz=180), pos)
        
        # HAT
        if rhythm['hat'][s16_in_bar]:
            vel = rhythm['hat_vel'][s16_in_bar]
            is_open = (fa['blue'] > 0.5 and s16_in_bar in (6, 14))
            place(hat_bus, synth_hat(gain=vel*0.5, is_open=is_open), pos)
        
        beat_idx += 1
        pos += s16
    
    # ─── Impacts on scene cuts ───────────────────────────────────────────
    print("  Scene-cut impacts...")
    for sc_pos in scene_cuts:
        if sc_pos < total_samples:
            place(impact_bus, synth_impact(gain=1.2), sc_pos)
            # Also add a short riser before the cut
            riser_dur = min(0.5, sc_pos / SR)
            if riser_dur > 0.1:
                riser_start = max(0, sc_pos - int(riser_dur * SR))
                riser = synth_riser(riser_dur, gain=0.25)
                place(impact_bus, riser, riser_start)
    
    # ─── Sub bass (continuous, per-frame) ────────────────────────────────
    print("  Sub bass...")
    sub_phase = 0.0
    
    for chunk_start in range(0, total_samples, s16):
        chunk_end = min(chunk_start + s16, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0: continue
        
        fa = fa_at(chunk_start)
        
        # Chord from hue (continuous selection)
        chord_idx = int(fa['hue'] * len(CHORDS)) % len(CHORDS)
        chord = CHORDS[chord_idx]
        
        freq = chord[0]  # sub note
        # Pitch modulation from motion
        freq *= (1.0 + np.sin(chunk_start/SR * 2*np.pi * 0.5) * fa['motion_norm'] * 0.03)
        
        t = np.arange(chunk_len) / SR
        sub_phase_arr = sub_phase + np.cumsum(np.ones(chunk_len) * 2*np.pi*freq/SR)
        sub = np.sin(sub_phase_arr)
        sub_phase = sub_phase_arr[-1] if chunk_len > 0 else sub_phase
        
        # Gain driven by red channel + overall brightness
        sub_gain = (0.25 + fa['red'] * 0.5) * (0.4 + fa['brightness'] * 0.6)
        
        # Envelope: follows kick pattern loosely
        s16_in_bar = (chunk_start // s16) % 16
        env = np.exp(-t * (3 + fa['motion_norm'] * 5)) * 0.7 + 0.3
        
        sub_bus[chunk_start:chunk_end] += sub * sub_gain * env
    
    # ─── Reese bass (filter modulated by saturation per-frame) ───────────
    print("  Reese bass...")
    
    for chunk_start in range(0, total_samples, s16):
        chunk_end = min(chunk_start + s16, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0: continue
        
        fa = fa_at(chunk_start)
        chord_idx = int(fa['hue'] * len(CHORDS)) % len(CHORDS)
        freq = CHORDS[chord_idx][1]
        
        saw1 = synth_saw(freq*1.004, chunk_len, gain=0.3)
        saw2 = synth_saw(freq*0.996, chunk_len, gain=0.3)
        reese = saw1 + saw2
        
        # Filter sweeps with saturation + motion
        cutoff = 150 + fa['saturation']*2500 + fa['motion_norm']*1500 + fa['brightness']*800
        reese = apply_lp(reese, cutoff)
        
        gain = (0.08 + fa['edge_density']*0.25 + fa['motion_norm']*0.15)
        t = np.arange(chunk_len) / SR
        env = np.exp(-t*5) * 0.7 + 0.3
        
        reese_bus[chunk_start:chunk_end] += reese * gain * env
    
    # ─── Pad (brightness-driven filter, chord from hue) ──────────────────
    print("  Pad...")
    bar_samples = s16 * 16
    
    for bar_start in range(0, total_samples, bar_samples):
        bar_end = min(bar_start + bar_samples, total_samples)
        bar_len = bar_end - bar_start
        if bar_len <= 0: continue
        
        fa = fa_at(bar_start)
        chord_idx = int(fa['hue'] * len(CHORDS)) % len(CHORDS)
        pad_freqs = CHORDS[chord_idx][2]
        
        pad = np.zeros(bar_len)
        for freq in pad_freqs:
            pad += synth_saw(freq*1.002, bar_len, gain=0.12, harmonics=8)
            pad += synth_saw(freq*0.998, bar_len, gain=0.12, harmonics=8)
        pad /= max(len(pad_freqs), 1)
        
        # Filter opens and closes with brightness over the bar
        # Sample brightness at multiple points across the bar
        n_filter_points = 8
        for fp in range(n_filter_points):
            fp_start = bar_start + fp * bar_len // n_filter_points
            fp_end = bar_start + (fp+1) * bar_len // n_filter_points
            fp_len = fp_end - fp_start
            local_fa = fa_at(fp_start)
            cutoff = 150 + local_fa['brightness'] * 1200 + local_fa['saturation'] * 600
            seg = pad[fp*bar_len//n_filter_points : (fp+1)*bar_len//n_filter_points]
            if len(seg) > 10:
                pad[fp*bar_len//n_filter_points:(fp+1)*bar_len//n_filter_points] = apply_lp(seg, cutoff)
        
        # Envelope
        attack = int(0.2 * SR)
        release = int(0.3 * SR)
        env = np.ones(bar_len)
        if bar_len > attack: env[:attack] = np.linspace(0, 1, attack)
        if bar_len > release: env[-release:] = np.linspace(1, 0, release)
        
        gain = 0.03 + fa['green'] * 0.07
        
        pad_L[bar_start:bar_end] += pad * gain * env
        pad_R[bar_start:bar_end] += pad * gain * env * 0.85
    
    # ─── Spectral texture (primary layer — feel the image) ───────────────
    print("  Spectral texture (image → sound)...")
    
    n_bands = 64
    freqs = np.logspace(np.log10(60), np.log10(10000), n_bands)
    phases = np.zeros(n_bands)
    samples_per_frame = max(1, total_samples // n_frames)
    
    for fi in range(n_frames):
        fa = analyses[fi]
        row_prof = fa['row_profile']  # 120 values
        # Resample to n_bands
        indices = np.linspace(0, len(row_prof)-1, n_bands).astype(int)
        amps = row_prof[indices][::-1]  # flip: bottom=low freq
        
        chunk_start = fi * samples_per_frame
        chunk_end = min(chunk_start + samples_per_frame, total_samples)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0: continue
        
        # Motion amplifies the texture
        motion_boost = 1.0 + fa['motion_norm'] * 2.0
        
        texture = np.zeros(chunk_len)
        for i in range(n_bands):
            if amps[i] < 0.03: continue
            freq = freqs[i]
            if freq > SR/2: continue
            ph_inc = 2*np.pi*freq/SR
            ph = phases[i] + np.cumsum(np.ones(chunk_len) * ph_inc)
            texture += np.sin(ph) * amps[i] * motion_boost
            phases[i] = ph[-1] % (2*np.pi)
        
        peak = np.max(np.abs(texture))
        if peak > 0: texture /= peak
        
        # Gain: green drives base, motion drives boost
        gain = (0.06 + fa['green']*0.12 + fa['brightness']*0.08) * motion_boost * 0.5
        
        # Stereo from color balance
        lr_balance = fa['red'] - fa['blue']  # positive=more left, negative=more right
        l_gain = 0.5 + lr_balance * 0.3
        r_gain = 0.5 - lr_balance * 0.3
        
        spectral_L[chunk_start:chunk_end] += texture * gain * l_gain
        spectral_R[chunk_start:chunk_end] += texture * gain * r_gain
        # Width via haas
        d = int(0.006 * SR)
        if chunk_end + d < buf_len:
            spectral_R[chunk_start+d:chunk_end+d] += texture * gain * 0.2
    
    # ─── Mix ─────────────────────────────────────────────────────────────
    print("\n[3/4] Mixing...")
    
    # Process buses
    sub_bus = apply_lp(sub_bus[:total_samples], 120)
    sub_bus = np.tanh(sub_bus * 1.3)
    reese_bus = apply_hp(reese_bus[:total_samples], 80)
    reese_bus = np.tanh(reese_bus * 1.5)
    snare_bus = comb_verb(snare_bus[:total_samples], decay=0.08, delays_ms=(12,))
    spectral_L = comb_verb(spectral_L[:total_samples], decay=0.15, delays_ms=(35,50))
    spectral_R = comb_verb(spectral_R[:total_samples], decay=0.15, delays_ms=(45,65))
    pad_L = comb_verb(pad_L[:total_samples], decay=0.25, delays_ms=(55,70))
    pad_R = comb_verb(pad_R[:total_samples], decay=0.25, delays_ms=(70,85))
    
    def t(buf): return buf[:total_samples] if len(buf)>=total_samples else np.pad(buf,(0,total_samples-len(buf)))
    
    mix_L = np.zeros(total_samples)
    mix_R = np.zeros(total_samples)
    
    # Drums
    mix_L += t(kick_bus)*1.0;     mix_R += t(kick_bus)*1.0
    mix_L += t(snare_bus)*0.85;   mix_R += t(snare_bus)*0.85
    mix_L += t(hat_bus)*0.55;     mix_R += t(hat_bus)*0.45
    
    # Bass
    mix_L += t(sub_bus)*0.8;      mix_R += t(sub_bus)*0.8
    mix_L += t(reese_bus)*0.5;    mix_R += t(reese_bus)*0.5
    
    # Spectral (PRIMARY — louder)
    mix_L += t(spectral_L)*son_gain
    mix_R += t(spectral_R)*son_gain
    
    # Pad
    mix_L += t(pad_L);            mix_R += t(pad_R)
    
    # Impacts
    mix_L += t(impact_bus)*0.9;   mix_R += t(impact_bus)*0.9
    
    # ─── Master ──────────────────────────────────────────────────────────
    print("  Master processing...")
    stereo = np.column_stack([mix_L, mix_R])
    peak = np.max(np.abs(stereo))
    if peak > 0: stereo = stereo / peak * 0.95
    
    # Compression
    thresh = 0.55
    ratio = 3.5
    for ch in range(2):
        above = np.abs(stereo[:,ch]) > thresh
        stereo[above,ch] = np.sign(stereo[above,ch]) * (thresh + (np.abs(stereo[above,ch])-thresh)/ratio)
    
    stereo = np.clip(stereo, -0.98, 0.98)
    
    # ─── Write ───────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(os.path.abspath(video_path)), f"{base}_v2.wav")
    
    print(f"\n[4/4] Writing: {output_path}")
    sf.write(output_path, stereo, SR, subtype='PCM_24')
    
    size = os.path.getsize(output_path)
    print(f"\n✅ Done! {duration:.1f}s, {size/1024/1024:.1f} MB")
    return output_path


def main():
    p = argparse.ArgumentParser(description="Video→DnB v2: image IS the rhythm")
    p.add_argument("video", help="Input video")
    p.add_argument("--bpm", type=int, default=174)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--spectral-gain", type=float, default=0.5, help="Spectral texture gain (default 0.5)")
    args = p.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: not found: {args.video}"); sys.exit(1)
    
    start = time.time()
    render(args.video, args.bpm, args.output, args.spectral_gain)
    print(f"  Render time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
