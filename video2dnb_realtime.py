#!/usr/bin/env python3
"""
video2dnb_realtime.py — Realtime video-to-DnB audio engine.

Reads from webcam or video file and synthesizes DnB audio in real-time.
Every frame directly drives the audio synthesis — you feel the image live.

Usage:
    python video2dnb_realtime.py                      # webcam
    python video2dnb_realtime.py video.mp4             # video file
    python video2dnb_realtime.py --bpm 174 --device 0  # webcam device 0
    python video2dnb_realtime.py video.mp4 --record out.wav  # record to WAV

Keys (when video window is focused):
    q / ESC  — quit
    1-9      — switch camera (device index)
    f        — load video file (prints prompt in terminal)
    +/-      — adjust BPM
    s        — toggle spectral texture
    d        — toggle drums
    v        — toggle video display
    r        — toggle recording to WAV
"""

import cv2
import numpy as np
from scipy import signal as sig
import argparse, os, sys, time, threading

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

SR = 48000
BLOCK_SIZE = 512  # ~10.7ms latency at 48kHz

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


# ─── Shared State ────────────────────────────────────────────────────────────

class VisualParams:
    """Thread-safe container for visual parameters from video frames."""
    def __init__(self):
        self.lock = threading.Lock()
        self.brightness = 0.5
        self.contrast = 0.3
        self.red = 0.5
        self.green = 0.5
        self.blue = 0.5
        self.saturation = 0.3
        self.edge_density = 0.1
        self.motion = 0.0
        self.motion_norm = 0.0
        self.hue = 0.0
        self.quad_tl = 0.5  # kick zone
        self.quad_tr = 0.5  # hat zone
        self.quad_bl = 0.5  # snare zone
        self.quad_br = 0.5  # bass zone
        self.row_profile = np.zeros(64)
        self.col_profile = np.zeros(16)
        self.is_scene_cut = False
        self.frame_count = 0
        self.max_motion = 0.01
    
    def update(self, frame, prev_gray):
        """Analyze frame and update all params. Returns new gray frame."""
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        b = np.mean(gray) / 255.0
        c = min(np.std(gray) / 128.0, 1.0)
        r = np.mean(small[:,:,2]) / 255.0
        g = np.mean(small[:,:,1]) / 255.0
        bl = np.mean(small[:,:,0]) / 255.0
        sat = np.mean(hsv[:,:,1]) / 255.0
        
        edges = cv2.Canny(gray, 50, 150)
        ed = np.sum(edges > 0) / edges.size
        
        h, w = gray.shape
        qtl = np.mean(gray[:h//2,:w//2]) / 255.0
        qtr = np.mean(gray[:h//2,w//2:]) / 255.0
        qbl = np.mean(gray[h//2:,:w//2]) / 255.0
        qbr = np.mean(gray[h//2:,w//2:]) / 255.0
        
        # Row profile for spectral
        row_prof = np.mean(gray, axis=1) / 255.0
        indices = np.linspace(0, len(row_prof)-1, 64).astype(int)
        row_64 = row_prof[indices][::-1]
        
        # Column profile for rhythm
        col_prof = np.mean(gray, axis=0) / 255.0
        col_indices = np.linspace(0, len(col_prof)-1, 16).astype(int)
        col_16 = col_prof[col_indices]
        
        # Hue
        sat_mask = hsv[:,:,1] > 20
        hue = float(np.median(hsv[:,:,0][sat_mask])) / 90.0 if np.any(sat_mask) else 0.0
        
        # Motion
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mot = np.mean(diff) / 255.0
        else:
            mot = 0.0
        
        with self.lock:
            self.brightness = b
            self.contrast = c
            self.red = r
            self.green = g
            self.blue = bl
            self.saturation = sat
            self.edge_density = ed
            self.motion = mot
            self.max_motion = max(self.max_motion, mot, 0.01)
            self.motion_norm = mot / self.max_motion
            self.hue = hue
            self.quad_tl = qtl
            self.quad_tr = qtr
            self.quad_bl = qbl
            self.quad_br = qbr
            self.row_profile = row_64
            self.col_profile = col_16
            self.is_scene_cut = mot > 0.12
            self.frame_count += 1
        
        return gray
    
    def snapshot(self):
        """Get a copy of all params (thread-safe)."""
        with self.lock:
            return {
                'brightness': self.brightness,
                'contrast': self.contrast,
                'red': self.red,
                'green': self.green,
                'blue': self.blue,
                'saturation': self.saturation,
                'edge_density': self.edge_density,
                'motion': self.motion,
                'motion_norm': self.motion_norm,
                'hue': self.hue,
                'quad_tl': self.quad_tl,
                'quad_tr': self.quad_tr,
                'quad_bl': self.quad_bl,
                'quad_br': self.quad_br,
                'row_profile': self.row_profile.copy(),
                'col_profile': self.col_profile.copy(),
                'is_scene_cut': self.is_scene_cut,
            }


# ─── Realtime Audio Engine ───────────────────────────────────────────────────

class DnBEngine:
    """Realtime DnB audio synthesis engine driven by visual parameters."""
    
    def __init__(self, bpm=174):
        self.bpm = bpm
        self.sample_pos = 0
        self.sub_phase = 0.0
        self.spectral_phases = np.zeros(64)
        self.spectral_freqs = np.logspace(np.log10(60), np.log10(10000), 64)
        
        # Smoothed params (exponential moving average)
        self.smooth = {
            'brightness': 0.5, 'contrast': 0.3, 'red': 0.5, 'green': 0.5,
            'blue': 0.5, 'saturation': 0.3, 'edge_density': 0.1,
            'motion_norm': 0.0, 'hue': 0.0,
            'quad_tl': 0.5, 'quad_tr': 0.5, 'quad_bl': 0.5, 'quad_br': 0.5,
        }
        self.smooth_alpha = 0.15  # smoothing factor (0=no change, 1=instant)
        
        self.row_profile = np.zeros(64)
        self.col_profile = np.zeros(16)
        
        # Toggles
        self.drums_on = True
        self.spectral_on = True
        
        # Pre-generate one-shot sounds at different settings
        self._kick_cache = {}
        self._snare_cache = {}
        
        # Recording
        self.recording = False
        self.recorded_frames = []
    
    def update_params(self, params):
        """Update with new visual params (called from video thread)."""
        for key in self.smooth:
            if key in params:
                self.smooth[key] += self.smooth_alpha * (params[key] - self.smooth[key])
        
        if 'row_profile' in params:
            self.row_profile += 0.2 * (params['row_profile'] - self.row_profile)
        if 'col_profile' in params:
            self.col_profile += 0.2 * (params['col_profile'] - self.col_profile)
    
    def _get_kick(self, pitch_mul):
        """Get/cache kick sound."""
        key = round(pitch_mul, 2)
        if key not in self._kick_cache:
            n = int(SR * 0.12)
            t = np.arange(n) / SR
            pitch = (150*key) * np.exp(-t*40) + 40
            phase = np.cumsum(2*np.pi*pitch/SR)
            osc = np.sin(phase) * np.exp(-t*12)
            click = np.exp(-t*200) * 0.5
            self._kick_cache[key] = np.tanh((osc+click)*1.5) * 0.85
        return self._kick_cache[key]
    
    def _get_snare(self):
        n = int(SR * 0.08)
        t = np.arange(n) / SR
        noise = np.random.randn(n) * np.exp(-t*25)
        tone = np.sin(2*np.pi*200*t) * np.exp(-t*30)
        return np.tanh((noise*0.6 + tone*0.4)*2) * 0.65
    
    def _get_hat(self, gain=0.3, is_open=False):
        dur = 0.12 if is_open else 0.03
        n = int(SR * dur)
        t = np.arange(n) / SR
        noise = np.random.randn(n) * np.exp(-t*(8 if is_open else 55))
        # Simple highpass approximation
        out = np.diff(noise, prepend=0) * 3.0
        return out * gain
    
    def render_block(self, n_samples):
        """Render n_samples of audio. Returns stereo array (n, 2)."""
        out_L = np.zeros(n_samples)
        out_R = np.zeros(n_samples)
        
        p = self.smooth
        beat_dur = 60.0 / self.bpm
        s16_samples = int(beat_dur / 4 * SR)
        
        # ─── Drums ───────────────────────────────────────────────────────
        if self.drums_on:
            for i in range(n_samples):
                global_pos = self.sample_pos + i
                
                # Check if we're at a 16th note boundary
                if global_pos % s16_samples == 0:
                    s16_idx = (global_pos // s16_samples) % 16
                    col_val = self.col_profile[s16_idx] if s16_idx < len(self.col_profile) else 0.5
                    motion = p['motion_norm']
                    edge = p['edge_density']
                    
                    # KICK: dark spots in column profile + beat 1
                    kick_thresh = 0.5 - motion * 0.15
                    if col_val < kick_thresh or s16_idx == 0:
                        pitch_mul = 0.85 + p['brightness'] * 0.3
                        kick = self._get_kick(pitch_mul) * (0.7 + motion*0.3)
                        end = min(i + len(kick), n_samples)
                        out_L[i:end] += kick[:end-i]
                        out_R[i:end] += kick[:end-i]
                    
                    # SNARE: bright spots
                    snare_thresh = 0.55 - edge * 0.1
                    if col_val > snare_thresh or s16_idx == 4:
                        snare = self._get_snare() * (0.5 + edge*0.5)
                        end = min(i + len(snare), n_samples)
                        out_L[i:end] += snare[:end-i] * 0.85
                        out_R[i:end] += snare[:end-i] * 0.85
                    
                    # HAT: edge density drives density
                    play_hat = False
                    if edge > 0.12:
                        play_hat = True  # every 16th
                    elif edge > 0.06:
                        play_hat = (s16_idx % 2 == 0)  # 8ths
                    else:
                        play_hat = (s16_idx % 4 == 0)  # quarters
                    
                    if play_hat:
                        is_open = p['blue'] > 0.5 and s16_idx in (6, 14)
                        vel = col_val * 0.5 + 0.15
                        hat = self._get_hat(gain=vel, is_open=is_open)
                        end = min(i + len(hat), n_samples)
                        out_L[i:end] += hat[:end-i] * 0.55
                        out_R[i:end] += hat[:end-i] * 0.45
        
        # ─── Sub bass (continuous) ───────────────────────────────────────
        chord_idx = int(p['hue'] * len(CHORDS)) % len(CHORDS)
        sub_freq = CHORDS[chord_idx][0]
        sub_freq *= (1.0 + np.sin(self.sample_pos/SR * np.pi) * p['motion_norm'] * 0.02)
        
        t = np.arange(n_samples)
        sub_phase_arr = self.sub_phase + np.cumsum(np.ones(n_samples) * 2*np.pi*sub_freq/SR)
        sub = np.sin(sub_phase_arr) * (0.2 + p['red']*0.4) * (0.4 + p['brightness']*0.6)
        self.sub_phase = sub_phase_arr[-1] % (2*np.pi)
        
        out_L += sub * 0.7
        out_R += sub * 0.7
        
        # ─── Reese bass ─────────────────────────────────────────────────
        mid_freq = CHORDS[chord_idx][1]
        t_arr = np.arange(n_samples) / SR
        r1 = np.zeros(n_samples)
        r2 = np.zeros(n_samples)
        for k in range(1, 8):
            if k*mid_freq*1.004 > SR/2: break
            r1 += np.sin(2*np.pi*k*mid_freq*1.004*t_arr) / k
            r2 += np.sin(2*np.pi*k*mid_freq*0.996*t_arr) / k
        reese = (r1 + r2) * 0.15
        
        # Simple lowpass via exponential smoothing
        cutoff_norm = (150 + p['saturation']*2000 + p['motion_norm']*1000) / (SR/2)
        cutoff_norm = np.clip(cutoff_norm, 0.001, 0.99)
        alpha_f = cutoff_norm * 0.5
        for i in range(1, n_samples):
            reese[i] = reese[i-1] + alpha_f * (reese[i] - reese[i-1])
        
        reese_gain = 0.06 + p['edge_density']*0.2 + p['motion_norm']*0.1
        out_L += reese * reese_gain
        out_R += reese * reese_gain
        
        # ─── Spectral texture ───────────────────────────────────────────
        if self.spectral_on:
            motion_boost = 1.0 + p['motion_norm'] * 2.0
            texture = np.zeros(n_samples)
            
            for i in range(64):
                amp = self.row_profile[i]
                if amp < 0.03: continue
                freq = self.spectral_freqs[i]
                if freq > SR/2: continue
                ph_inc = 2*np.pi*freq/SR
                ph = self.spectral_phases[i] + np.cumsum(np.ones(n_samples)*ph_inc)
                texture += np.sin(ph) * amp * motion_boost
                self.spectral_phases[i] = ph[-1] % (2*np.pi)
            
            peak = np.max(np.abs(texture))
            if peak > 0: texture /= peak
            
            gain = (0.05 + p['green']*0.1 + p['brightness']*0.06) * motion_boost * 0.4
            lr = p['red'] - p['blue']
            out_L += texture * gain * (0.5 + lr*0.3)
            out_R += texture * gain * (0.5 - lr*0.3)
        
        # ─── Scene cut impact ───────────────────────────────────────────
        if p['motion_norm'] > 0.85:
            n_imp = min(int(SR*0.3), n_samples)
            t_imp = np.arange(n_imp) / SR
            impact = np.sin(2*np.pi*(80*np.exp(-t_imp*8))*t_imp) * np.exp(-t_imp*5) * 0.6
            out_L[:n_imp] += impact
            out_R[:n_imp] += impact
        
        # ─── Mix and clip ───────────────────────────────────────────────
        stereo = np.column_stack([out_L, out_R])
        
        # Soft clip
        stereo = np.tanh(stereo * 0.8) * 0.95
        
        self.sample_pos += n_samples
        
        # Record if active
        if self.recording:
            self.recorded_frames.append(stereo.copy())
        
        return stereo.astype(np.float32)


# ─── Video Thread ────────────────────────────────────────────────────────────

def open_source(source):
    """Open a video source (file path or camera index). Returns (cap, is_file, fps)."""
    if isinstance(source, str) and os.path.exists(source):
        cap = cv2.VideoCapture(source)
        is_file = True
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"  📹 Opened video file: {os.path.basename(source)} ({fps:.0f}fps)")
    else:
        idx = int(source) if str(source).isdigit() else 0
        cap = cv2.VideoCapture(idx)
        is_file = False
        fps = 30
        if cap.isOpened():
            print(f"  📷 Opened camera {idx}")
        else:
            print(f"  ❌ Cannot open camera {idx}")
    return cap, is_file, fps


def video_thread_func(source, vis_params, engine, show_video, stop_event, switch_request):
    """Capture and analyze video frames in a loop. Supports live source switching."""
    cap, is_file, fps = open_source(source)
    
    if not cap.isOpened():
        print(f"Error: cannot open video source: {source}")
        stop_event.set()
        return
    
    frame_delay = 1.0 / fps
    prev_gray = None
    video_display_ok = True  # track if cv2.imshow works
    
    while not stop_event.is_set():
        t_start = time.time()
        
        # Check for source switch request
        if switch_request[0] is not None:
            new_source = switch_request[0]
            switch_request[0] = None
            cap.release()
            prev_gray = None
            cap, is_file, fps = open_source(new_source)
            frame_delay = 1.0 / fps
            if not cap.isOpened():
                print(f"  ❌ Failed to open: {new_source}, trying camera 0...")
                cap, is_file, fps = open_source('0')
            continue
        
        ret, frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                time.sleep(0.1)
                continue
        
        prev_gray = vis_params.update(frame, prev_gray)
        engine.update_params(vis_params.snapshot())
        
        if show_video[0] and video_display_ok:
            try:
                display = frame.copy()
                h, dw = display.shape[:2]
                
                with vis_params.lock:
                    src_name = "Camera" if not is_file else "File"
                    info = [
                        f"SRC: {src_name} | BPM: {engine.bpm}",
                        f"Bright: {vis_params.brightness:.2f}  Motion: {vis_params.motion_norm:.2f}",
                        f"Edges: {vis_params.edge_density:.2f}  Sat: {vis_params.saturation:.2f}",
                        f"R:{vis_params.red:.2f} G:{vis_params.green:.2f} B:{vis_params.blue:.2f}",
                        f"Drums: {'ON' if engine.drums_on else 'OFF'}  Spectral: {'ON' if engine.spectral_on else 'OFF'}",
                    ]
                    if engine.recording:
                        info.append("● REC")
                
                # Dark overlay for text readability
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (320, 18 + len(info)*20), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                
                for i, txt in enumerate(info):
                    cv2.putText(display, txt, (8, 18+i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                
                # Column profile rhythm bars at bottom
                col = vis_params.col_profile
                bar_w = max(1, dw // len(col))
                for i, v in enumerate(col):
                    x = i * bar_w
                    bh = int(v * 50)
                    color = (0, int(v*255), int((1-v)*255))
                    cv2.rectangle(display, (x, h-bh), (x+bar_w-1, h), color, -1)
                
                # Row profile spectral bars on right side
                row = vis_params.row_profile
                bar_h = max(1, h // len(row))
                for i, v in enumerate(row):
                    y = i * bar_h
                    bw = int(v * 40)
                    cv2.rectangle(display, (dw-bw, y), (dw, y+bar_h-1), (int(v*200),100,255), -1)
                
                cv2.imshow("video2dnb realtime", display)
            except Exception as e:
                print(f"  ⚠️  Video display error: {e}")
                print(f"  Disabling video display. Audio continues.")
                video_display_ok = False
        
        # Key handling — use cv2.waitKey even without display for timing
        try:
            key = cv2.waitKey(1) & 0xFF
        except:
            key = 255
            time.sleep(frame_delay)
        
        if key == ord('q') or key == 27:
            stop_event.set()
            break
        elif key in range(ord('1'), ord('9')+1):
            cam_idx = key - ord('0')
            print(f"  Switching to camera {cam_idx}...")
            switch_request[0] = str(cam_idx)
        elif key == ord('0'):
            print(f"  Switching to camera 0...")
            switch_request[0] = '0'
        elif key == ord('f'):
            print("  Enter video file path (in terminal):")
            try:
                path = input("  > ").strip()
                if os.path.exists(path):
                    switch_request[0] = path
                else:
                    print(f"  ❌ File not found: {path}")
            except:
                pass
        elif key == ord('+') or key == ord('='):
            engine.bpm = min(300, engine.bpm + 5)
            print(f"  BPM: {engine.bpm}")
        elif key == ord('-'):
            engine.bpm = max(60, engine.bpm - 5)
            print(f"  BPM: {engine.bpm}")
        elif key == ord('s'):
            engine.spectral_on = not engine.spectral_on
            print(f"  Spectral: {'ON' if engine.spectral_on else 'OFF'}")
        elif key == ord('d'):
            engine.drums_on = not engine.drums_on
            print(f"  Drums: {'ON' if engine.drums_on else 'OFF'}")
        elif key == ord('v'):
            show_video[0] = not show_video[0]
            if not show_video[0]:
                try: cv2.destroyAllWindows()
                except: pass
            else:
                video_display_ok = True  # re-enable
        elif key == ord('r'):
            if engine.recording:
                engine.recording = False
                print("  ⏹ Recording stopped")
            else:
                engine.recorded_frames = []
                engine.recording = True
                print("  ● Recording started...")
        
        # Maintain frame rate
        elapsed = time.time() - t_start
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    cap.release()
    try: cv2.destroyAllWindows()
    except: pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="video2dnb realtime — live video to DnB")
    p.add_argument("source", nargs='?', default='0', help="Video file or webcam index (default: 0)")
    p.add_argument("--bpm", type=int, default=174)
    p.add_argument("--device", type=int, default=None, help="Audio output device index")
    p.add_argument("--record", type=str, default=None, help="Record output to WAV file")
    p.add_argument("--no-video", action="store_true", help="Don't show video window")
    p.add_argument("--list-devices", action="store_true", help="List audio devices")
    args = p.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        return
    
    print("=" * 60)
    print("  video2dnb REALTIME")
    print(f"  Source: {args.source}")
    print(f"  BPM: {args.bpm}")
    print(f"  Audio: {SR}Hz, block={BLOCK_SIZE}")
    print("=" * 60)
    print("\nControls:")
    print("  q/ESC    quit")
    print("  0-9      switch camera")
    print("  f        load video file")
    print("  +/-      BPM up/down")
    print("  d        toggle drums")
    print("  s        toggle spectral")
    print("  v        toggle video display")
    print("  r        toggle recording")
    print()
    
    vis_params = VisualParams()
    engine = DnBEngine(bpm=args.bpm)
    
    if args.record:
        engine.recording = True
        engine.recorded_frames = []
        print(f"Recording to: {args.record}")
    
    stop_event = threading.Event()
    show_video = [not args.no_video]
    switch_request = [None]  # mutable container for source switch requests
    
    # Audio callback (runs on its own thread via sounddevice)
    def audio_callback(outdata, frames, time_info, status):
        if status and 'underflow' not in str(status).lower():
            print(f"Audio status: {status}")
        try:
            block = engine.render_block(frames)
            outdata[:] = block
        except Exception as e:
            print(f"Audio error: {e}")
            outdata[:] = 0
    
    # Start audio stream
    try:
        stream = sd.OutputStream(
            samplerate=SR,
            blocksize=BLOCK_SIZE,
            channels=2,
            dtype='float32',
            callback=audio_callback,
            device=args.device,
        )
    except Exception as e:
        print(f"Error opening audio: {e}")
        print("\nAvailable devices:")
        print(sd.query_devices())
        sys.exit(1)
    
    print("Starting audio engine...")
    stream.start()
    print("Starting video capture...")
    print("🎵 LIVE! Press q to quit.\n")
    
    # Run video on MAIN THREAD (required by macOS for cv2.imshow)
    try:
        video_thread_func(args.source, vis_params, engine, show_video, stop_event, switch_request)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    
    stream.stop()
    stream.close()
    
    # Save recording if active
    if args.record and engine.recorded_frames:
        print(f"\nSaving recording to {args.record}...")
        import soundfile as sf
        audio = np.concatenate(engine.recorded_frames, axis=0)
        sf.write(args.record, audio, SR, subtype='PCM_24')
        dur = len(audio) / SR
        size = os.path.getsize(args.record) / 1024 / 1024
        print(f"Saved: {dur:.1f}s, {size:.1f} MB")
    
    print("Done.")

if __name__ == "__main__":
    main()
