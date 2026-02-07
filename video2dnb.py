#!/usr/bin/env python3
"""
video2dnb - Convert video visuals into DnB patterns for Strudel.

Analyzes video frames for visual properties (brightness, color, edges,
contrast, saturation) and maps them to drum'n'bass musical parameters,
outputting a complete Strudel live-coding script.

Key design: rank-based normalization stretches the video's own visual
range across the full musical parameter space, so even subtle visual
changes produce audible musical dynamics.

Usage:
    python video2dnb.py <video_path> [--sections N] [--bpm BPM] [--output FILE]
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ─── DnB Musical Constants ───────────────────────────────────────────────────

# C minor key center - chord palette used in typical DnB
# Each entry: (name, pad_notes, sub_bass_note, mid_bass_note, arp_notes)
CHORD_PALETTE = [
    ("Cm",   ["c3","eb3","g3"],         "c1",  "c2",  ["c4","eb4","g4","c5"]),
    ("Cm7",  ["c3","eb3","g3","bb3"],   "c1",  "c2",  ["c4","eb4","g4","bb4"]),
    ("Abmaj",["ab2","c3","eb3"],        "ab0", "ab2", ["ab3","c4","eb4","ab4"]),
    ("Fm",   ["f2","ab2","c3","eb3"],  "f0",  "f2",  ["f3","ab3","c4","f4"]),
    ("Ebmaj",["eb3","g3","bb3"],        "eb1", "eb2", ["eb4","g4","bb4","eb5"]),
    ("Bbm",  ["bb2","d3","f3"],         "bb0", "bb2", ["bb3","d4","f4","bb4"]),
]

# DnB chord progressions (indices into CHORD_PALETTE) - 4 bars each
# We pick a progression based on the section's visual character
PROGRESSIONS = [
    [0, 0, 2, 2],        # Cm → Cm → Ab → Ab (classic dark DnB)
    [0, 0, 2, 3],        # Cm → Cm → Ab → Fm (tension builder)
    [0, 1, 2, 3],        # Cm → Cm7 → Ab → Fm (full journey)
    [1, 4, 2, 3],        # Cm7 → Eb → Ab → Fm (euphoric)
    [2, 3, 0, 0],        # Ab → Fm → Cm → Cm (resolving)
    [3, 2, 0, 1],        # Fm → Ab → Cm → Cm7 (building)
]

# Kick patterns ordered by energy level (0=silence, 5=busiest)
KICK_PATTERNS = [
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",                  # 0: silence
    "bd ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",                 # 1: minimal
    "bd ~ ~ ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~ ~ ~",                # 2: half-time
    "bd ~ ~ ~ ~ ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~",                # 3: two-step
    "bd ~ ~ bd ~ ~ bd ~ ~ ~ ~ ~ ~ ~ ~ ~",               # 4: driving
    "bd ~ ~ bd ~ ~ bd ~ ~ bd bd ~ ~ ~ ~ ~",             # 5: busy
]

# Snare patterns ordered by energy
SNARE_PATTERNS = [
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",                  # 0: silence
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~",                 # 1: end hit
    "~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~",                # 2: standard 2&4
    "~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~",                # 3: standard
    "~ ~ ~ ~ sd ~ ~ sd ~ ~ ~ ~ sd ~ ~ ~",               # 4: syncopated
    "~ ~ ~ ~ sd ~ sd ~ ~ ~ ~ sd sd ~ sd ~",             # 5: jungle
]

# Ghost snare patterns
GHOST_PATTERNS = [
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",                  # 0: none
    "~ ~ sd:1 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",               # 1: single ghost
    "~ ~ sd:1 ~ ~ ~ ~ sd:1 ~ ~ ~ ~ ~ sd:1 ~ ~",         # 2: scattered
    "~ sd:1 ~ ~ ~ ~ sd:1 ~ sd:1 ~ ~ ~ ~ sd:1 ~ ~",      # 3: rolling
    "~ sd:1 ~ ~ ~ ~ sd:1 ~ sd:1 ~ ~ ~ ~ sd:1 ~ sd:1",   # 4: busy
    "~ sd:1 ~ sd:1 ~ ~ sd:1 sd:1 ~ sd:1 ~ ~ ~ sd:1 ~ sd:1", # 5: jungle
]

# Hi-hat patterns ordered by energy
HAT_PATTERNS = [
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",                  # 0: silence
    "~ ~ hh ~ ~ ~ hh ~ ~ ~ hh ~ ~ ~ ~ hh",              # 1: sparse
    "hh ~ hh ~ hh ~ hh ~",                               # 2: 8ths
    "hh*8",                                               # 3: 8ths alt
    "hh*16",                                              # 4: 16ths
    "[hh hh hh hh hh hh hh hh hh hh hh hh hh hh hh hh]",# 5: 16ths explicit
]

# Hi-hat gain patterns (velocity) for each density level
HAT_VELOCITIES = [
    ".15",
    ".15 .1 .12 .1",
    ".35 .15 .25 .15 .35 .15 .28 .18",
    ".4 .18 .3 .18 .4 .18 .32 .2",
    ".5 .22 .38 .22 .48 .22 .38 .25 .5 .22 .38 .22 .48 .22 .42 .28",
    ".55 .25 .42 .28 .55 .28 .42 .3 .55 .25 .42 .28 .55 .28 .48 .35",
]

# Open hat patterns
OPEN_HAT_PATTERNS = [
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~",
    "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ oh ~",
    "~ ~ ~ ~ ~ ~ oh ~ ~ ~ ~ ~ ~ ~ oh ~",
    "~ ~ ~ ~ ~ ~ oh ~ ~ ~ ~ ~ ~ oh ~ ~",
    "~ ~ ~ oh ~ ~ oh ~ ~ ~ ~ oh ~ ~ oh ~",
]

# Bass rhythms ordered by energy
BASS_RHYTHMS = [
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # 0: single hit
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],  # 1: two hits
    [1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],  # 2: three hits
    [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],  # 3: syncopated
    [1,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0],  # 4: busy
    [1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0],  # 5: jungle
]


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    dominant_hue: float = 0.0
    hue_spread: float = 0.0       # how varied are the hues
    edge_density: float = 0.0
    red_energy: float = 0.0
    green_energy: float = 0.0
    blue_energy: float = 0.0
    warm_cool_ratio: float = 0.5  # >0.5 = warm, <0.5 = cool
    motion: float = 0.0


@dataclass
class SectionAnalysis:
    """Aggregated analysis for a section of the video."""
    # Raw averaged values
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    edge_density: float = 0.0
    red_energy: float = 0.0
    green_energy: float = 0.0
    blue_energy: float = 0.0
    motion: float = 0.0
    warm_cool_ratio: float = 0.5
    hue_spread: float = 0.0
    brightness_trend: float = 0.0
    motion_variance: float = 0.0  # spiky motion = drops/impacts

    # Normalized 0-1 values (rank-stretched within the video)
    n_brightness: float = 0.0
    n_contrast: float = 0.0
    n_saturation: float = 0.0
    n_edge_density: float = 0.0
    n_motion: float = 0.0
    n_warmth: float = 0.5
    n_hue_spread: float = 0.0
    n_motion_variance: float = 0.0

    # Musical parameters (set after normalization)
    energy: float = 0.0
    drum_density: int = 0
    bass_density: int = 0
    hat_density: int = 0
    ghost_density: int = 0
    progression_idx: int = 0
    filter_cutoff: float = 500.0
    gain_mult: float = 0.5
    room_amount: float = 0.3
    has_ride: bool = False
    has_crash: bool = False
    has_arp: bool = False
    has_melody: bool = False
    is_breakdown: bool = False


def analyze_frame(frame: np.ndarray, prev_frame: np.ndarray = None) -> FrameAnalysis:
    """Analyze a single video frame for visual properties."""
    result = FrameAnalysis()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result.brightness = float(np.mean(hsv[:, :, 2])) / 255.0
    result.contrast = float(np.std(gray)) / 128.0
    result.contrast = min(result.contrast, 1.0)
    result.saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    # Hue analysis
    sat_mask = hsv[:, :, 1] > 20
    if np.any(sat_mask):
        hue_values = hsv[:, :, 0][sat_mask].astype(float) * 2.0
        result.dominant_hue = float(np.median(hue_values))
        result.hue_spread = float(np.std(hue_values)) / 180.0
    else:
        result.dominant_hue = 0.0
        result.hue_spread = 0.0

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    result.edge_density = float(np.sum(edges > 0)) / float(edges.size)

    # Color channels
    result.blue_energy = float(np.mean(frame[:, :, 0])) / 255.0
    result.green_energy = float(np.mean(frame[:, :, 1])) / 255.0
    result.red_energy = float(np.mean(frame[:, :, 2])) / 255.0

    # Warm/cool ratio (red+yellow vs blue+cyan)
    warm = result.red_energy + result.green_energy * 0.5
    cool = result.blue_energy + result.green_energy * 0.5
    total = warm + cool
    result.warm_cool_ratio = warm / total if total > 0 else 0.5

    # Motion
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        result.motion = float(np.mean(diff)) / 255.0
    else:
        result.motion = 0.0

    return result


def analyze_video(video_path: str, num_sections: int) -> List[SectionAnalysis]:
    """Extract frames from video and analyze them in sections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
    print(f"Splitting into {num_sections} sections ({duration/num_sections:.1f}s each)")

    frames_per_section = total_frames // num_sections
    sample_interval = max(1, frames_per_section // 30)  # ~30 samples per section

    all_frame_analyses: List[List[FrameAnalysis]] = [[] for _ in range(num_sections)]
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        section_for_frame = min(frame_idx * num_sections // total_frames, num_sections - 1)

        if frame_idx % sample_interval == 0:
            small = cv2.resize(frame, (160, 120))
            analysis = analyze_frame(small, prev_frame)
            all_frame_analyses[section_for_frame].append(analysis)
            prev_frame = small

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100
            print(f"\rAnalyzing [{'█' * int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)

    cap.release()
    print(f"\rAnalyzing [{'█' * 40}] 100%")

    # Aggregate each section
    sections = []
    for frames in all_frame_analyses:
        sections.append(aggregate_section(frames))
    while len(sections) < num_sections:
        sections.append(SectionAnalysis())

    # CRITICAL: rank-normalize across all sections
    normalize_sections(sections)

    # Derive musical parameters from normalized values
    derive_musical_params(sections)

    print(f"Analyzed {len(sections)} sections")
    return sections


def aggregate_section(frames: List[FrameAnalysis]) -> SectionAnalysis:
    """Aggregate frame analyses into raw section values."""
    s = SectionAnalysis()
    n = len(frames)
    if n == 0:
        return s

    s.brightness = float(np.mean([f.brightness for f in frames]))
    s.contrast = float(np.mean([f.contrast for f in frames]))
    s.saturation = float(np.mean([f.saturation for f in frames]))
    s.edge_density = float(np.mean([f.edge_density for f in frames]))
    s.red_energy = float(np.mean([f.red_energy for f in frames]))
    s.green_energy = float(np.mean([f.green_energy for f in frames]))
    s.blue_energy = float(np.mean([f.blue_energy for f in frames]))
    s.motion = float(np.mean([f.motion for f in frames]))
    s.warm_cool_ratio = float(np.mean([f.warm_cool_ratio for f in frames]))
    s.hue_spread = float(np.mean([f.hue_spread for f in frames]))

    # Motion variance (spiky = transient events)
    motions = [f.motion for f in frames]
    s.motion_variance = float(np.std(motions)) if n > 1 else 0.0

    # Brightness trend
    if n > 1:
        brightnesses = [f.brightness for f in frames]
        half = n // 2
        s.brightness_trend = float(np.mean(brightnesses[half:]) - np.mean(brightnesses[:half]))

    return s


def rank_normalize(values: List[float]) -> List[float]:
    """Normalize values by rank, stretching to full 0-1 range.
    The lowest value becomes 0, the highest becomes 1.
    Ties get averaged ranks."""
    n = len(values)
    if n <= 1:
        return [0.5] * n

    # Get sort indices
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n

    # Handle ties
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j+1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    # Normalize ranks to 0-1
    max_rank = n - 1
    if max_rank == 0:
        return [0.5] * n
    return [r / max_rank for r in ranks]


def normalize_sections(sections: List[SectionAnalysis]):
    """Rank-normalize all parameters across sections to use the full 0-1 range."""
    n = len(sections)
    if n == 0:
        return

    # Extract raw values
    attrs = [
        ('brightness', 'n_brightness'),
        ('contrast', 'n_contrast'),
        ('saturation', 'n_saturation'),
        ('edge_density', 'n_edge_density'),
        ('motion', 'n_motion'),
        ('warm_cool_ratio', 'n_warmth'),
        ('hue_spread', 'n_hue_spread'),
        ('motion_variance', 'n_motion_variance'),
    ]

    for raw_attr, norm_attr in attrs:
        raw_values = [getattr(s, raw_attr) for s in sections]
        normalized = rank_normalize(raw_values)
        for s, nv in zip(sections, normalized):
            setattr(s, norm_attr, nv)

    print(f"  Rank-normalized {len(attrs)} parameters across {n} sections")


def derive_musical_params(sections: List[SectionAnalysis]):
    """Map normalized visual parameters to musical parameters."""
    n = len(sections)

    for i, s in enumerate(sections):
        # Overall energy combines multiple visual features
        s.energy = (
            0.25 * s.n_brightness +
            0.25 * s.n_motion +
            0.20 * s.n_edge_density +
            0.15 * s.n_contrast +
            0.15 * s.n_motion_variance
        )

        # Drum density: motion + edges + contrast → rhythmic complexity
        drum_score = 0.35 * s.n_motion + 0.30 * s.n_edge_density + 0.20 * s.n_contrast + 0.15 * s.n_motion_variance
        s.drum_density = int(np.clip(round(drum_score * 4) + 1, 1, 5))  # min 1, always active

        # Bass density: brightness + warmth → bass weight
        bass_score = 0.40 * s.n_brightness + 0.35 * s.n_warmth + 0.25 * s.n_motion
        s.bass_density = int(np.clip(round(bass_score * 4) + 1, 1, 5))  # min 1, always active

        # Hat density: edge density + contrast → texture detail
        hat_score = 0.40 * s.n_edge_density + 0.30 * s.n_contrast + 0.30 * s.n_saturation
        s.hat_density = int(np.clip(round(hat_score * 4) + 1, 1, 5))  # min 1, always active

        # Ghost density: always at least 1 for texture
        s.ghost_density = max(1, s.drum_density - 1)

        # Chord progression: combine warmth, brightness, hue spread
        prog_score = 0.4 * s.n_warmth + 0.3 * s.n_hue_spread + 0.3 * s.n_brightness
        s.progression_idx = int(np.clip(round(prog_score * (len(PROGRESSIONS) - 1)), 0, len(PROGRESSIONS) - 1))

        # Filter cutoff: saturation + brightness
        s.filter_cutoff = 400 + (0.5 * s.n_saturation + 0.5 * s.n_brightness) * 3200

        # Gain: energy-based, high floor
        s.gain_mult = 0.6 + s.energy * 0.4

        # Room: inverse of motion (calm = reverby, busy = dry)
        s.room_amount = 0.6 - s.n_motion * 0.45

        # Musical arrangement flags - everything always active, just varying
        s.has_ride = s.hat_density >= 4
        s.has_crash = (i > 0 and abs(s.energy - sections[i-1].energy) > 0.1)
        s.has_arp = True  # always on
        s.has_melody = True  # always on
        s.is_breakdown = False  # no breakdowns, always full energy

    print(f"  Derived musical parameters for {n} sections")
    for i, s in enumerate(sections):
        flags = []
        if s.is_breakdown: flags.append("BREAK")
        if s.has_ride: flags.append("RIDE")
        if s.has_crash: flags.append("CRASH")
        if s.has_arp: flags.append("ARP")
        if s.has_melody: flags.append("MELODY")
        flag_str = " [" + ",".join(flags) + "]" if flags else ""
        print(f"  S{i+1}: E={s.energy:.2f} D={s.drum_density} B={s.bass_density} "
              f"H={s.hat_density} G={s.ghost_density} prog={s.progression_idx}{flag_str}")


def make_sub_bass_pattern(section: SectionAnalysis) -> str:
    """Generate a 4-bar sub bass pattern using the section's chord progression."""
    prog = PROGRESSIONS[section.progression_idx]
    rhythm = BASS_RHYTHMS[min(section.bass_density, len(BASS_RHYTHMS) - 1)]

    bars = []
    for chord_idx in prog:
        bass_note = CHORD_PALETTE[chord_idx][2]  # sub bass note
        parts = []
        for hit in rhythm:
            parts.append(bass_note if hit else "~")
        bars.append("[" + " ".join(parts) + "]")

    return "<" + " ".join(bars) + ">"


def make_mid_bass_pattern(section: SectionAnalysis) -> str:
    """Generate a 4-bar mid bass pattern."""
    prog = PROGRESSIONS[section.progression_idx]
    rhythm = BASS_RHYTHMS[min(section.bass_density, len(BASS_RHYTHMS) - 1)]

    bars = []
    for chord_idx in prog:
        mid_note = CHORD_PALETTE[chord_idx][3]  # mid bass note
        parts = []
        for hit in rhythm:
            parts.append(mid_note if hit else "~")
        bars.append("[" + " ".join(parts) + "]")

    return "<" + " ".join(bars) + ">"


def make_pad_pattern(section: SectionAnalysis) -> str:
    """Generate a 4-bar pad chord pattern."""
    prog = PROGRESSIONS[section.progression_idx]
    bars = []
    for chord_idx in prog:
        notes = CHORD_PALETTE[chord_idx][1]  # pad notes
        bars.append("[" + ",".join(notes) + "]")
    return "<" + " ".join(bars) + ">"


def make_stab_pattern(section: SectionAnalysis) -> str:
    """Generate chord stab pattern (offbeat hits)."""
    prog = PROGRESSIONS[section.progression_idx]
    bars = []
    for chord_idx in prog:
        notes = CHORD_PALETTE[chord_idx][1]
        chord_str = "[" + ",".join(notes) + "]"
        bars.append(f"[~ ~ ~ ~ ~ {chord_str} ~ ~]")
    return "<" + " ".join(bars) + ">"


def make_arp_pattern(section: SectionAnalysis) -> str:
    """Generate arpeggio patterns from the chord progression."""
    prog = PROGRESSIONS[section.progression_idx]
    bars = []
    for chord_idx in prog:
        arp_notes = CHORD_PALETTE[chord_idx][4]  # arp notes
        if section.energy > 0.6:
            # Fast arp - all 8 slots
            pattern = " ".join([arp_notes[i % len(arp_notes)] for i in range(8)])
        elif section.energy > 0.4:
            # Medium - sparse arp
            pattern = f"{arp_notes[0]} ~ {arp_notes[1]} ~ {arp_notes[2]} ~ {arp_notes[1]} ~"
        else:
            # Slow - very sparse
            pattern = f"{arp_notes[0]} ~ ~ ~ {arp_notes[2]} ~ ~ ~"
        bars.append("[" + pattern + "]")
    return "<" + " ".join(bars) + ">"


def make_melody_pattern(section: SectionAnalysis) -> str:
    """Generate breakdown melody pattern."""
    prog = PROGRESSIONS[section.progression_idx]
    bars = []
    for chord_idx in prog:
        arp_notes = CHORD_PALETTE[chord_idx][4]
        # Sparse melody with rests
        pattern = f"~ ~ {arp_notes[2]} ~ ~ ~ {arp_notes[1]} ~"
        bars.append("[" + pattern + "]")
    return "<" + " ".join(bars) + ">"


def generate_strudel(sections: List[SectionAnalysis], bpm: int = 174) -> str:
    """Generate complete Strudel DnB code from section analyses."""
    n = len(sections)
    total_cycles = n * 16
    cpm = bpm / 4

    lines = []
    lines.append("// ============================================================")
    lines.append("// VIDEO2DNB - Auto-generated DnB from video analysis")
    lines.append(f"// {n} sections x 16 cycles = {total_cycles} cycles total")
    lines.append(f"// BPM: {bpm} | Paste into Strudel and press Ctrl+Enter")
    lines.append("// ============================================================")
    lines.append("")
    lines.append(f"setcpm({cpm})")
    lines.append("")

    # Summary comments
    lines.append("// Section analysis:")
    for i, s in enumerate(sections):
        prog = PROGRESSIONS[s.progression_idx]
        chord_names = " -> ".join([CHORD_PALETTE[c][0] for c in prog])
        flags = []
        if s.is_breakdown: flags.append("BREAK")
        if s.has_ride: flags.append("RIDE")
        if s.has_crash: flags.append("CRASH")
        if s.has_arp: flags.append("ARP")
        if s.has_melody: flags.append("MEL")
        flag_str = " [" + ",".join(flags) + "]" if flags else ""
        lines.append(f"// S{i+1}: E={s.energy:.2f} D={s.drum_density} B={s.bass_density} "
                     f"H={s.hat_density} | {chord_names}{flag_str}")
    lines.append("")

    TC = total_cycles  # shorthand

    # ─── KICK ───
    lines.append("// ===================== KICK =====================")
    kick_items = []
    for s in sections:
        kick_items.append(f's("{KICK_PATTERNS[s.drum_density]}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in kick_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    gains = [f"{s.gain_mult * 1.1:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    shapes = [f"{min(s.energy * 0.35, 0.3):.2f}" for s in sections]
    lines.append(f"  .shape(cat({', '.join(shapes)}).slow({TC}))")
    lines.append("")

    # ─── SNARE ───
    lines.append("// ===================== SNARE =====================")
    snare_items = []
    for s in sections:
        snare_items.append(f's("{SNARE_PATTERNS[s.drum_density]}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in snare_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    gains = [f"{s.gain_mult:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .speed(1.05)")
    lines.append("  .room(.08)")
    lines.append("")

    # ─── GHOST SNARES ───
    lines.append("// ===================== GHOST SNARES =====================")
    ghost_items = []
    for s in sections:
        ghost_items.append(f's("{GHOST_PATTERNS[min(s.ghost_density, 5)]}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in ghost_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    gains = [f"{s.energy * 0.22:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("")

    # ─── CLOSED HATS ───
    lines.append("// ===================== CLOSED HATS =====================")
    hat_items = []
    hat_vel_items = []
    for s in sections:
        hat_items.append(f's("{HAT_PATTERNS[s.hat_density]}")')
        hat_vel_items.append(HAT_VELOCITIES[s.hat_density])
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in hat_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    lines.append(f"  .gain(cat(")
    for i, vel in enumerate(hat_vel_items):
        comma = "," if i < len(hat_vel_items) - 1 else ""
        lines.append(f'    "{vel}"{comma}')
    lines.append(f"  ).slow({TC}))")
    lines.append("  .speed(perlin.range(.95, 1.05))")
    lines.append("")

    # ─── OPEN HAT ───
    lines.append("// ===================== OPEN HAT =====================")
    oh_items = []
    for s in sections:
        idx = min(max(s.hat_density - 1, 0), len(OPEN_HAT_PATTERNS) - 1)
        oh_items.append(f's("{OPEN_HAT_PATTERNS[idx]}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in oh_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    gains = [f"{s.energy * 0.28:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .cut(1)")
    lines.append("")

    # ─── RIDE ───
    lines.append("// ===================== RIDE =====================")
    ride_items = []
    for s in sections:
        if s.has_ride:
            ride_items.append('s("~ ride ~ ride ~ ride ~ ride ~ ride ~ ride ~ ride ~ ride")')
        else:
            ride_items.append("silence")
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in ride_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    gains = [f"{0.1:.2f}" if s.has_ride else "0" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .speed(1.1)")
    lines.append("")

    # ─── CRASH ───
    lines.append("// ===================== CRASH =====================")
    crash_items = []
    for s in sections:
        if s.has_crash:
            crash_items.append('s("cp ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")')
        else:
            crash_items.append("silence")
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in crash_items))
    lines.append(f").slow({TC})")
    lines.append('  .bank("RolandTR909")')
    gains = [".18" if s.has_crash else "0" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .room(.35)")
    lines.append("")

    # ─── SUB BASS ───
    lines.append("// ===================== SUB BASS =====================")
    sub_items = []
    for s in sections:
        pattern = make_sub_bass_pattern(s)
        sub_items.append(f'note("{pattern}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in sub_items))
    lines.append(f").slow({TC})")
    lines.append('  .s("sine")')
    gains = [f"{s.gain_mult * 0.85:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    decays = [f"{max(0.1, 0.5 - s.energy * 0.35):.2f}" for s in sections]
    lines.append(f"  .decay(cat({', '.join(decays)}).slow({TC}))")
    sustains = [f"{max(0.05, 0.25 - s.energy * 0.2):.2f}" for s in sections]
    lines.append(f"  .sustain(cat({', '.join(sustains)}).slow({TC}))")
    lines.append("")

    # ─── MID BASS (REESE) ───
    lines.append("// ===================== MID BASS (REESE) =====================")
    mid_items = []
    for s in sections:
        pattern = make_mid_bass_pattern(s)
        mid_items.append(f'note("{pattern}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in mid_items))
    lines.append(f").slow({TC})")
    lines.append('  .s("sawtooth")')
    lines.append("  .superimpose(add(.04))")
    cutoffs = [f"{s.filter_cutoff:.0f}" for s in sections]
    lines.append(f"  .cutoff(sine.slow(4).range(300, cat({', '.join(cutoffs)}).slow({TC})))")
    res = [f"{min(s.n_saturation * 22 + 5, 22):.0f}" for s in sections]
    lines.append(f"  .resonance(cat({', '.join(res)}).slow({TC}))")
    gains = [f"{s.energy * 0.22:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .decay(.1).sustain(0)")
    shapes = [f"{min(s.energy * 0.3, 0.3):.2f}" for s in sections]
    lines.append(f"  .shape(cat({', '.join(shapes)}).slow({TC}))")
    lines.append("")

    # ─── PAD ───
    lines.append("// ===================== PAD =====================")
    pad_items = []
    for s in sections:
        pattern = make_pad_pattern(s)
        pad_items.append(f'note("{pattern}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in pad_items))
    lines.append(f").slow({TC})")
    lines.append('  .s("sawtooth")')
    cutoffs_pad = [f"{400 + s.n_brightness * 500:.0f}" for s in sections]
    lines.append(f"  .cutoff(sine.slow(8).range(200, cat({', '.join(cutoffs_pad)}).slow({TC})))")
    gains = [f"{0.06 + s.energy * 0.08:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .attack(.3).decay(.5).sustain(.4).release(.8)")
    rooms = [f"{s.room_amount:.2f}" for s in sections]
    lines.append(f"  .room(cat({', '.join(rooms)}).slow({TC}))")
    sizes = [f"{min(s.room_amount + 0.2, 0.9):.2f}" for s in sections]
    lines.append(f"  .size(cat({', '.join(sizes)}).slow({TC}))")
    lines.append("  .pan(sine.slow(6).range(.25, .75))")
    lines.append("")

    # ─── CHORD STABS ───
    lines.append("// ===================== CHORD STABS =====================")
    stab_items = []
    for s in sections:
        pattern = make_stab_pattern(s)
        stab_items.append(f'note("{pattern}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in stab_items))
    lines.append(f").slow({TC})")
    lines.append('  .s("sawtooth")')
    cutoffs_stab = [f"{800 + s.energy * 1200:.0f}" for s in sections]
    lines.append(f"  .cutoff(cat({', '.join(cutoffs_stab)}).slow({TC}))")
    gains = [f"{s.energy * 0.07:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .decay(.12).sustain(0)")
    lines.append("  .room(.3).size(.45)")
    lines.append("")

    # ─── ARP LEAD ───
    lines.append("// ===================== ARP LEAD =====================")
    arp_items = []
    for s in sections:
        pattern = make_arp_pattern(s)
        arp_items.append(f'note("{pattern}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in arp_items))
    lines.append(f").slow({TC})")
    lines.append('  .s("square")')
    lines.append("  .cutoff(sine.slow(3).range(800, 3500))")
    gains = [f"{s.energy * 0.06:.3f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .decay(.07).sustain(0)")
    lines.append("  .room(.2).size(.35)")
    lines.append("  .pan(sine.slow(2).range(.2, .8))")
    lines.append("")

    # ─── MELODY ───
    lines.append("// ===================== MELODY =====================")
    mel_items = []
    for s in sections:
        pattern = make_melody_pattern(s)
        mel_items.append(f'note("{pattern}")')
    lines.append("$: cat(")
    lines.append(",\n".join(f"  {item}" for item in mel_items))
    lines.append(f").slow({TC})")
    lines.append('  .s("triangle")')
    lines.append("  .cutoff(2000)")
    gains = [f"{0.08:.2f}" for s in sections]
    lines.append(f"  .gain(cat({', '.join(gains)}).slow({TC}))")
    lines.append("  .decay(.3).sustain(.2).release(.4)")
    lines.append("  .room(.5).size(.7)")
    lines.append("  .delay(.3).delaytime(.375).delayfeedback(.4)")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert video to DnB Strudel patterns")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--sections", type=int, default=8,
                        help="Number of musical sections (default: 8)")
    parser.add_argument("--bpm", type=int, default=174,
                        help="BPM (default: 174)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output .js file path")
    parser.add_argument("--json", action="store_true",
                        help="Also output analysis as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Analyze video
    print(f"\n🎬 Analyzing video: {args.video}")
    sections = analyze_video(args.video, args.sections)

    # Generate Strudel code
    print(f"\n🎵 Generating DnB patterns...")
    strudel_code = generate_strudel(sections, args.bpm)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, "..", f"video-dnb-{base}.js")

    # Write output
    with open(out_path, "w") as f:
        f.write(strudel_code)
    print(f"\n✅ Strudel code written to: {out_path}")
    print(f"   Paste into Strudel REPL and press Ctrl+Enter to play!")

    # Optionally write JSON analysis
    if args.json:
        json_path = out_path.replace(".js", "-analysis.json")
        analysis_data = []
        for i, s in enumerate(sections):
            analysis_data.append({
                "section": i + 1,
                "brightness": round(s.brightness, 3),
                "contrast": round(s.contrast, 3),
                "saturation": round(s.saturation, 3),
                "edge_density": round(s.edge_density, 3),
                "motion": round(s.motion, 3),
                "energy": round(s.energy, 3),
                "drum_density": s.drum_density,
                "bass_density": s.bass_density,
                "hat_density": s.hat_density,
                "progression": [CHORD_PALETTE[c][0] for c in PROGRESSIONS[s.progression_idx]],
                "filter_cutoff": round(s.filter_cutoff, 0),
                "room": round(s.room_amount, 3),
            })
        with open(json_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
        print(f"📊 Analysis JSON written to: {json_path}")


if __name__ == "__main__":
    main()
