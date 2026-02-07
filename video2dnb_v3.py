#!/usr/bin/env python3
"""
video2dnb_v3.py — Visual Guide-Driven DnB Synthesis

Uses pre-analyzed visual patterns to create more rhythmic but still image-driven audio.
Combines the best of v1 (structured rhythm) and v2 (visual responsiveness).

Key improvements over v2:
- Visual guide provides rhythmic templates extracted from actual video
- More structured drum patterns while staying image-responsive
- Better groove and flow from analyzed rhythmic patterns
- Preserves visual responsiveness through dynamic modulation

Usage:
    python video2dnb_v3.py video.mp4 --bpm 174 --output audio.wav
"""

import cv2, numpy as np, argparse, json, os, sys
from scipy import signal as sig
import soundfile as sf

SR = 48000

# ─── Notes ───────────────────────────────────────────────────────────────────
NOTE_HZ = {
    'C1':32.70,'Eb1':38.89,'F1':43.65,'Ab1':51.91,'Bb1':58.27,
    'C2':65.41,'Eb2':77.78,'F2':87.31,'G2':98.0,'Ab2':103.83,'Bb2':116.54,
    'C3':130.81,'Eb3':155.56,'F3':174.61,'G3':196.0,'Ab3':207.65,'Bb3':233.08,
    'C4':261.63,'Eb4':311.13,'F4':349.23,'G4':392.0,'Ab4':415.30,'Bb4':466.16,
    'C5':523.25,'Eb5':622.25,'G5':783.99,
}

CHORDS = [
    (NOTE_HZ['C1'],  NOTE_HZ['C2'],  [NOTE_HZ['C3'],NOTE_HZ['Eb3'],NOTE_HZ['G3']]),
    (NOTE_HZ['Ab1'], NOTE_HZ['Ab2'], [NOTE_HZ['Ab3'],NOTE_HZ['C4'],NOTE_HZ['Eb4']]),
    (NOTE_HZ['F1'],  NOTE_HZ['F2'],  [NOTE_HZ['F3'],NOTE_HZ['Ab3'],NOTE_HZ['C4']]),
    (NOTE_HZ['Eb1'], NOTE_HZ['Eb2'], [NOTE_HZ['Eb3'],NOTE_HZ['G3'],NOTE_HZ['Bb3']]),
]


# ─── Load Visual Guide ───────────────────────────────────────────────────────

def load_visual_guide(guide_path="visual_guide.json"):
    """Load pre-analyzed visual guide."""
    if not os.path.exists(guide_path):
        print(f"Warning: Visual guide not found at {guide_path}")
        print("Run analyze_samples.py first to create the guide")
        return None
    
    with open(guide_path, 'r') as f:
        return json.load(f)


# ─── Video Analysis ───────────────────────────────────────────────────────────

def analyze_video(video_path):
    """Analyze video frames and map to visual guide patterns."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"[1/4] Analyzing video...")
    print(f"  {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
    
    frames_data = []
    prev_gray = None
    motion_history = []
    scene_cuts = []
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 1000 == 0 and frame_idx > 0:
            print(f"  {frame_idx} frames...")
        
        # Resize and analyze
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Basic visual features
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0
        sat = np.mean(hsv[:,:,1]) / 255.0
        
        # Color channels
        b = np.mean(small[:,:,0]) / 255.0
        g = np.mean(small[:,:,1]) / 255.0
        r = np.mean(small[:,:,2]) / 255.0
        
        # Hue
        sat_mask = hsv[:,:,1] > 20
        hue = float(np.median(hsv[:,:,0][sat_mask])) / 90.0 if np.any(sat_mask) else 0.0
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Motion detection
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff) / 255.0
            motion_history.append(motion)
            
            # Scene cut detection
            if motion > 0.12:
                scene_cuts.append(frame_idx)
        else:
            motion = 0.0
        
        # Column and row profiles
        col_profile = np.mean(gray, axis=0) / 255.0
        col_indices = np.linspace(0, len(col_profile)-1, 16).astype(int)
        col_16 = col_profile[col_indices]
        
        row_profile = np.mean(gray, axis=1) / 255.0
        row_indices = np.linspace(0, len(row_profile)-1, 32).astype(int)
        row_32 = row_profile[row_indices][::-1]
        
        # Quadrant energies
        h, w = gray.shape
        qtl = np.mean(gray[:h//2,:w//2]) / 255.0
        qtr = np.mean(gray[:h//2,w//2:]) / 255.0
        qbl = np.mean(gray[h//2:,:w//2]) / 255.0
        qbr = np.mean(gray[h//2:,w//2:]) / 255.0
        
        frames_data.append({
            'frame_idx': frame_idx,
            'brightness': brightness,
            'contrast': contrast,
            'saturation': sat,
            'hue': hue,
            'red': r, 'green': g, 'blue': b,
            'edge_density': edge_density,
            'motion': motion,
            'col_profile': col_16.tolist(),
            'row_profile': row_32.tolist(),
            'quadrants': [qtl, qtr, qbl, qbr],
        })
        
        prev_gray = gray.copy()
    
    cap.release()
    
    # Calculate motion statistics
    motion_array = np.array(motion_history)
    motion_mean = np.mean(motion_array) if len(motion_array) > 0 else 0
    motion_std = np.std(motion_array) if len(motion_array) > 0 else 0
    motion_threshold = motion_mean + 2 * motion_std
    motion_spikes = np.sum(motion_array > motion_threshold) if len(motion_array) > 0 else 0
    
    print(f"  Scene cuts: {len(scene_cuts)}, Motion spikes: {motion_spikes}")
    
    return frames_data, fps, scene_cuts, motion_spikes


# ─── Pattern Matching ─────────────────────────────────────────────────────────

def match_to_guide_patterns(frame_data, visual_guide):
    """Match current frame to guide patterns for rhythmic templates."""
    if visual_guide is None:
        # Fallback to simple patterns if no guide
        return {
            'kick_pattern': [0, 4, 8, 12],
            'snare_pattern': [4, 12],
            'hat_pattern': [2, 6, 10, 14],
        }
    
    # Find closest guide pattern based on column profile similarity
    current_profile = np.array(frame_data['col_profile'])
    guide_patterns = visual_guide['rhythmic_patterns']
    
    # Calculate similarity to each guide pattern
    similarities = []
    for pattern in guide_patterns:
        guide_profile = np.array(pattern['profile'])
        similarity = 1.0 - np.mean(np.abs(current_profile - guide_profile))
        similarities.append(similarity)
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_pattern = guide_patterns[best_idx]
    
    # Extract rhythmic positions
    kick_pattern = best_pattern['kicks']
    snare_pattern = best_pattern['snares']
    hat_pattern = best_pattern['hats']
    
    # Add some variation based on current frame
    motion_boost = frame_data['motion']
    edge_boost = frame_data['edge_density']
    
    # Add extra kicks on high motion
    if motion_boost > 0.1 and 8 not in kick_pattern:
        kick_pattern.append(8)
    
    # Add extra snares on high edge density
    if edge_boost > 0.15 and 14 not in snare_pattern:
        snare_pattern.append(14)
    
    return {
        'kick_pattern': sorted(kick_pattern),
        'snare_pattern': sorted(snare_pattern),
        'hat_pattern': sorted(hat_pattern),
        'similarity': similarities[best_idx],
    }


# ─── Audio Synthesis ─────────────────────────────────────────────────────────

def synthesize_kick(pitch_mul=1.0):
    """Generate kick drum."""
    n = int(SR * 0.15)
    t = np.arange(n) / SR
    pitch = (60 * pitch_mul) * np.exp(-t * 30) + 40
    phase = np.cumsum(2 * np.pi * pitch / SR)
    osc = np.sin(phase) * np.exp(-t * 10)
    click = np.exp(-t * 150) * 0.6
    return np.tanh((osc + click) * 1.5) * 0.85

def synthesize_snare():
    """Generate snare drum."""
    n = int(SR * 0.1)
    t = np.arange(n) / SR
    noise = np.random.randn(n) * np.exp(-t * 20)
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 25)
    return np.tanh((noise * 0.7 + tone * 0.3) * 2) * 0.65

def synthesize_hat(gain=0.3, is_open=False):
    """Generate hi-hat."""
    dur = 0.08 if is_open else 0.03
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = np.random.randn(n) * np.exp(-t * (10 if is_open else 60))
    # Simple highpass
    out = np.diff(noise, prepend=0) * 3.0
    return out * gain

def synthesize_sub(freq, duration, modulation=0.0):
    """Generate sub bass."""
    t = np.arange(int(SR * duration)) / SR
    freq_mod = freq * (1.0 + modulation * np.sin(2 * np.pi * 2 * t))
    phase = np.cumsum(2 * np.pi * freq_mod / SR)
    return np.sin(phase) * 0.4

def synthesize_reese(freq, duration, cutoff=0.3):
    """Generate reese bass."""
    t = np.arange(int(SR * duration)) / SR
    r1 = np.zeros_like(t)
    r2 = np.zeros_like(t)
    for k in range(1, 8):
        if k * freq * 1.004 > SR / 2: break
        r1 += np.sin(2 * np.pi * k * freq * 1.004 * t) / k
        r2 += np.sin(2 * np.pi * k * freq * 0.996 * t) / k
    reese = (r1 + r2) * 0.15
    
    # Simple lowpass
    cutoff_norm = cutoff * (SR / 2)
    alpha = cutoff_norm / SR
    for i in range(1, len(reese)):
        reese[i] = reese[i-1] + alpha * (reese[i] - reese[i-1])
    
    return reese * 0.15

def synthesize_pad(chord_freqs, duration, brightness=0.5):
    """Generate chord pad."""
    t = np.arange(int(SR * duration)) / SR
    pad = np.zeros_like(t)
    
    for i, freq in enumerate(chord_freqs):
        phase = np.cumsum(2 * np.pi * freq / SR)
        osc = np.sin(phase) * 0.1
        # Filter sweep based on brightness
        filter_env = brightness * np.exp(-t * 0.5) + (1 - brightness) * 0.3
        osc = osc * filter_env
        pad += osc
    
    return pad * 0.3

def synthesize_spectral(row_profile, duration, motion=0.0):
    """Generate spectral texture from row profile."""
    t = np.arange(int(SR * duration)) / SR
    freqs = np.logspace(np.log10(60), np.log10(8000), len(row_profile))
    
    texture = np.zeros_like(t)
    for i, (freq, amp) in enumerate(zip(freqs, row_profile)):
        if amp < 0.05: continue
        if freq > SR / 2: continue
        phase = np.cumsum(2 * np.pi * freq / SR)
        osc = np.sin(phase) * amp
        texture += osc
    
    # Normalize and apply motion boost
    peak = np.max(np.abs(texture))
    if peak > 0: texture /= peak
    
    motion_boost = 1.0 + motion * 2.0
    gain = 0.1 * motion_boost
    
    return texture * gain


# ─── Main Render Engine ─────────────────────────────────────────────────────

def render_video2dnb_v3(video_path, bpm, output_path, visual_guide=None):
    """Render video to DnB using visual guide patterns."""
    
    # Analyze video
    frames_data, fps, scene_cuts, motion_spikes = analyze_video(video_path)
    
    # Load visual guide
    if visual_guide is None:
        visual_guide = load_visual_guide()
    
    beat_dur = 60.0 / bpm
    s16_dur = beat_dur / 4
    total_samples = int(len(frames_data) / fps * SR)
    
    print(f"[2/4] Synthesizing...")
    print(f"  BPM: {bpm}, 16th: {s16_dur*1000:.1f}ms")
    
    # Initialize audio buffers
    audio = np.zeros((total_samples, 2))
    
    # Render each frame
    for frame_idx, frame_data in enumerate(frames_data):
        frame_time = frame_idx / fps
        sample_start = int(frame_time * SR)
        sample_end = min(sample_start + int(SR / fps), total_samples)
        
        if sample_start >= total_samples:
            break
        
        # Match to guide patterns
        patterns = match_to_guide_patterns(frame_data, visual_guide)
        
        # Get current 16th note position
        s16_pos = int((frame_time % beat_dur) / s16_dur) % 16
        
        # Check if we should play drums at this position
        play_kick = s16_pos in patterns['kick_pattern']
        play_snare = s16_pos in patterns['snare_pattern']
        play_hat = s16_pos in patterns['hat_pattern']
        
        # Synthesize drums
        if play_kick:
            pitch_mul = 0.85 + frame_data['brightness'] * 0.3
            kick = synthesize_kick(pitch_mul)
            kick_len = min(len(kick), sample_end - sample_start)
            audio[sample_start:sample_start+kick_len, 0] += kick[:kick_len]
            audio[sample_start:sample_start+kick_len, 1] += kick[:kick_len]
        
        if play_snare:
            snare = synthesize_snare()
            snare_len = min(len(snare), sample_end - sample_start)
            audio[sample_start:sample_start+snare_len, 0] += snare[:snare_len] * 0.85
            audio[sample_start:sample_start+snare_len, 1] += snare[:snare_len] * 0.85
        
        if play_hat:
            is_open = frame_data['blue'] > 0.5 and s16_pos in (6, 14)
            hat_gain = 0.2 + frame_data['edge_density'] * 0.3
            hat = synthesize_hat(gain=hat_gain, is_open=is_open)
            hat_len = min(len(hat), sample_end - sample_start)
            audio[sample_start:sample_start+hat_len, 0] += hat[:hat_len] * 0.55
            audio[sample_start:sample_start+hat_len, 1] += hat[:hat_len] * 0.45
        
        # Continuous bass elements
        chord_idx = int(frame_data['hue'] * len(CHORDS)) % len(CHORDS)
        sub_freq = CHORDS[chord_idx][0]
        mid_freq = CHORDS[chord_idx][1]
        
        # Sub bass
        sub_mod = frame_data['motion'] * 0.02
        sub = synthesize_sub(sub_freq, 1/fps, sub_mod)
        sub_len = min(len(sub), sample_end - sample_start)
        audio[sample_start:sample_start+sub_len, 0] += sub[:sub_len] * 0.7
        audio[sample_start:sample_start+sub_len, 1] += sub[:sub_len] * 0.7
        
        # Reese bass
        reese_cutoff = 0.2 + frame_data['saturation'] * 0.3 + frame_data['motion'] * 0.1
        reese = synthesize_reese(mid_freq, 1/fps, reese_cutoff)
        reese_len = min(len(reese), sample_end - sample_start)
        audio[sample_start:sample_start+reese_len, 0] += reese[:reese_len]
        audio[sample_start:sample_start+reese_len, 1] += reese[:reese_len]
        
        # Pad
        pad = synthesize_pad(CHORDS[chord_idx][2], 1/fps, frame_data['brightness'])
        pad_len = min(len(pad), sample_end - sample_start)
        audio[sample_start:sample_start+pad_len, 0] += pad[:pad_len]
        audio[sample_start:sample_start+pad_len, 1] += pad[:pad_len]
        
        # Spectral texture
        spectral = synthesize_spectral(frame_data['row_profile'], 1/fps, frame_data['motion'])
        spec_len = min(len(spectral), sample_end - sample_start)
        lr_balance = frame_data['red'] - frame_data['blue']
        audio[sample_start:sample_start+spec_len, 0] += spectral[:spec_len] * (0.5 + lr_balance * 0.3)
        audio[sample_start:sample_start+spec_len, 1] += spectral[:spec_len] * (0.5 - lr_balance * 0.3)
        
        # Scene cut impacts
        if frame_idx in scene_cuts:
            impact_len = min(int(SR * 0.2), sample_end - sample_start)
            t_impact = np.arange(impact_len) / SR
            impact = np.sin(2 * np.pi * (80 * np.exp(-t_impact * 8)) * t_impact) * np.exp(-t_impact * 5) * 0.6
            audio[sample_start:sample_start+impact_len, 0] += impact
            audio[sample_start:sample_start+impact_len, 1] += impact
        
        if frame_idx % 1000 == 0 and frame_idx > 0:
            print(f"  {frame_idx} frames...")
    
    print(f"[3/4] Mixing...")
    
    # Master processing
    # Soft clip
    audio = np.tanh(audio * 0.8) * 0.95
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    print(f"[4/4] Writing: {output_path}")
    
    # Save audio
    duration = len(audio) / SR
    sf.write(output_path, audio, SR, subtype='PCM_24')
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✅ Done! {duration:.1f}s, {size_mb:.1f} MB")
    
    return output_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="video2dnb v3 — visual guide-driven DnB synthesis")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--bpm", type=int, default=174, help="BPM for DnB synthesis")
    parser.add_argument("--output", required=True, help="Output WAV file")
    parser.add_argument("--guide", default="visual_guide.json", help="Visual guide JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)
    
    # Load visual guide
    visual_guide = load_visual_guide(args.guide)
    
    print("=" * 60)
    print("  video2dnb v3 — VISUAL GUIDE-DRIVEN")
    print(f"  Video: {os.path.basename(args.video)}")
    print(f"  Guide: {args.guide}")
    print(f"  BPM: {args.bpm}")
    print("=" * 60)
    
    # Render
    render_video2dnb_v3(args.video, args.bpm, args.output, visual_guide)

if __name__ == "__main__":
    main()
