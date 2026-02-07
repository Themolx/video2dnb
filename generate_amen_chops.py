#!/usr/bin/env python3
"""
Generate amen break chops — lo-fi, gritty, 1969-character drum hits.
Outputs 16 slices that approximate the classic amen break pattern.
"""
import numpy as np
import soundfile as sf
from scipy import signal
import os

SR = 44100
OUT = os.path.join(os.path.dirname(__file__), 'samples', 'amen')
os.makedirs(OUT, exist_ok=True)

# ─── Lo-fi processing chain (vinyl/tape character) ──────────────────────
def lofi(audio, sr=SR):
    """Add grit: bitcrush, saturation, bandpass, noise floor."""
    # Soft saturation
    audio = np.tanh(audio * 1.8) * 0.7
    # Slight bitcrush (reduce to ~12 bit feel)
    levels = 2**12
    audio = np.round(audio * levels) / levels
    # Bandpass 80-8000 Hz (lo-fi vinyl range)
    sos = signal.butter(2, [80, 8000], btype='band', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    # Add tiny noise floor
    audio += np.random.randn(len(audio)) * 0.008
    # Normalize
    pk = np.max(np.abs(audio))
    if pk > 0: audio = audio / pk * 0.92
    return audio

# ─── Synthesis primitives ───────────────────────────────────────────────
def amen_kick(sr=SR):
    """Punchy, slightly ringy kick like the original."""
    n = int(sr * 0.12); t = np.arange(n) / sr
    pitch = 160 * np.exp(-t * 35) + 50
    phase = np.cumsum(2 * np.pi * pitch / sr)
    body = np.sin(phase) * np.exp(-t * 18)
    click = np.exp(-t * 150) * 0.6
    # Add some room thump
    room = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 12) * 0.3
    return np.tanh((body + click + room) * 2.0) * 0.85

def amen_snare(sr=SR, tight=False):
    """Crispy, ringy snare — the iconic amen snare."""
    dur = 0.08 if tight else 0.14
    n = int(sr * dur); t = np.arange(n) / sr
    # Snare wires (noise)
    noise = np.random.randn(n) * np.exp(-t * (25 if tight else 15))
    sos_hi = signal.butter(2, [1500, min(9000, sr/2-100)], btype='band', fs=sr, output='sos')
    wires = signal.sosfilt(sos_hi, noise) * 0.7
    # Body tone — slightly higher than modern snares, more ring
    tone = np.sin(2 * np.pi * 220 * t) * np.exp(-t * 20) * 0.6
    tone2 = np.sin(2 * np.pi * 330 * t) * np.exp(-t * 25) * 0.25
    # Transient crack
    crack = np.exp(-t * 200) * 0.5
    return np.tanh((wires + tone + tone2 + crack) * 2.2) * 0.8

def amen_hat(sr=SR, open_hat=False):
    """Ride-like hi-hat / ride cymbal — the amen ride is iconic."""
    dur = 0.25 if open_hat else 0.06
    n = int(sr * dur); t = np.arange(n) / sr
    noise = np.random.randn(n)
    env = np.exp(-t * (4 if open_hat else 30))
    # Multiple resonant bands to simulate cymbal
    out = np.zeros(n)
    for freq in [3200, 4800, 6400, 8200, 10500]:
        if freq > sr/2 - 200: continue
        bw = freq * 0.15
        lo = max(freq - bw, 20); hi = min(freq + bw, sr/2 - 100)
        sos = signal.butter(2, [lo, hi], btype='band', fs=sr, output='sos')
        out += signal.sosfilt(sos, noise) * 0.3
    # Add metallic ping
    ping = np.sin(2 * np.pi * 5500 * t) * np.exp(-t * 15) * 0.15
    return (out * env + ping * env) * 0.6

def amen_ride(sr=SR):
    """Open ride cymbal — sustained, shimmery."""
    n = int(sr * 0.35); t = np.arange(n) / sr
    noise = np.random.randn(n)
    env = np.exp(-t * 3)
    out = np.zeros(n)
    for freq in [2800, 4200, 5800, 7500, 9800]:
        if freq > sr/2 - 200: continue
        bw = freq * 0.12
        lo = max(freq - bw, 20); hi = min(freq + bw, sr/2 - 100)
        sos = signal.butter(2, [lo, hi], btype='band', fs=sr, output='sos')
        out += signal.sosfilt(sos, noise) * 0.25
    bell = np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 8) * 0.2
    return (out * env + bell) * 0.5

# ─── Build the 16 amen chops ───────────────────────────────────────────
# Classic amen break pattern (The Winstons - "Amen, Brother"):
# Written as 16th notes over 4 beats at ~136 BPM original
# The break is typically chopped into 8 slices (8th notes) or 16 (16ths)
#
# Beat 1: Kick+Ride | . | ride | .
# Beat 2: Snare+Ride | . | kick+hat | .
# Beat 3: Kick+Ride | . | snare+ride | .
# Beat 4: Kick | . | hat+snare | ride

def make_chop(components, dur_sec=0.18):
    """Mix components into a single chop at given duration."""
    n = int(SR * dur_sec)
    buf = np.zeros(n)
    for sound, gain in components:
        s = sound * gain
        end = min(len(s), n)
        buf[:end] += s[:end]
    return lofi(buf)

print("Generating amen break chops...")

kick = amen_kick()
snare = amen_snare()
snare_t = amen_snare(tight=True)
hat = amen_hat()
hat_o = amen_hat(open_hat=True)
ride = amen_ride()

# 16 chops — the full amen break at 16th note resolution
chops = [
    # Beat 1
    ("kick_ride",      [(kick, 1.0), (ride, 0.7)]),         # 0: KICK + ride
    ("ride_ghost",     [(ride, 0.4), (hat, 0.2)]),           # 1: ghost ride
    ("snare_ride",     [(snare, 1.0), (ride, 0.6)]),         # 2: SNARE + ride
    ("hat_tick",       [(hat, 0.5)]),                         # 3: hat tick
    # Beat 2
    ("kick_hat",       [(kick, 0.9), (hat_o, 0.5)]),         # 4: kick + open hat
    ("ride_soft",      [(ride, 0.35)]),                       # 5: soft ride
    ("snare_ghost",    [(snare_t, 0.4), (ride, 0.3)]),       # 6: ghost snare
    ("hat_ride",       [(hat, 0.6), (ride, 0.4)]),            # 7: hat + ride
    # Beat 3
    ("kick_ride2",     [(kick, 1.0), (ride, 0.65)]),         # 8: KICK + ride
    ("hat_soft",       [(hat, 0.35)]),                        # 9: soft hat
    ("snare_ride2",    [(snare, 0.95), (ride, 0.55)]),       # 10: SNARE + ride
    ("ride_ring",      [(ride, 0.5), (hat, 0.15)]),          # 11: ride ring
    # Beat 4
    ("kick_short",     [(kick, 0.85)]),                       # 12: kick alone
    ("ride_tick",      [(ride, 0.3), (hat, 0.2)]),           # 13: ride tick
    ("snare_hat",      [(snare, 0.9), (hat_o, 0.4)]),       # 14: snare + hat
    ("ride_end",       [(ride, 0.6)]),                        # 15: ride tail
]

# Also create individual hits
singles = [
    ("kick",      [(kick, 1.0)]),
    ("snare",     [(snare, 1.0)]),
    ("snare_ghost",[(snare_t, 0.5)]),
    ("hat",       [(hat, 0.8)]),
    ("hat_open",  [(hat_o, 0.8)]),
    ("ride",      [(ride, 0.9)]),
]

for i, (name, components) in enumerate(chops):
    chop = make_chop(components, dur_sec=0.2)
    path = os.path.join(OUT, f"{i:02d}_{name}.wav")
    sf.write(path, chop, SR, subtype='PCM_16')
    print(f"  [{i:2d}] {name}")

print("\nGenerating individual hits...")
for name, components in singles:
    chop = make_chop(components, dur_sec=0.25)
    path = os.path.join(OUT, f"hit_{name}.wav")
    sf.write(path, chop, SR, subtype='PCM_16')
    print(f"  hit_{name}")

print(f"\nDone! {len(chops)} chops + {len(singles)} hits → {OUT}/")
