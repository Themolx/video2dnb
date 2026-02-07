#!/usr/bin/env python3
"""
analyze_samples.py — Analyze extracted video frames to create visual guide for v3.

Extracts rhythmic patterns from visual data:
- Column brightness profiles → drum patterns
- Row brightness profiles → melodic contours  
- Motion detection → energy levels
- Color analysis → harmonic content

The guide will be used by v3 to make more rhythmic but still image-driven audio.
"""

import cv2
import numpy as np
from scipy import signal as sig
import json
import os
from pathlib import Path

def analyze_frame(frame_path):
    """Analyze single frame and return visual features."""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    
    # Resize for consistent analysis
    small = cv2.resize(img, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    # Basic stats
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 128.0
    sat = np.mean(hsv[:,:,1]) / 255.0
    
    # Color channels
    b = np.mean(small[:,:,0]) / 255.0
    g = np.mean(small[:,:,1]) / 255.0
    r = np.mean(small[:,:,2]) / 255.0
    
    # Hue (median of saturated pixels)
    sat_mask = hsv[:,:,1] > 20
    hue = float(np.median(hsv[:,:,0][sat_mask])) / 90.0 if np.any(sat_mask) else 0.0
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Column profile (for rhythm)
    col_profile = np.mean(gray, axis=0) / 255.0
    col_indices = np.linspace(0, len(col_profile)-1, 16).astype(int)
    col_16 = col_profile[col_indices]
    
    # Row profile (for melody)
    row_profile = np.mean(gray, axis=1) / 255.0
    row_indices = np.linspace(0, len(row_profile)-1, 32).astype(int)
    row_32 = row_profile[row_indices][::-1]
    
    # Quadrant energies
    h, w = gray.shape
    qtl = np.mean(gray[:h//2,:w//2]) / 255.0
    qtr = np.mean(gray[:h//2,w//2:]) / 255.0
    qbl = np.mean(gray[h//2:,:w//2]) / 255.0
    qbr = np.mean(gray[h//2:,w//2:]) / 255.0
    
    return {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'saturation': float(sat),
        'hue': float(hue),
        'red': float(r),
        'green': float(g),
        'blue': float(b),
        'edge_density': float(edge_density),
        'col_profile': col_16.tolist(),
        'row_profile': row_32.tolist(),
        'quadrants': [float(qtl), float(qtr), float(qbl), float(qbr)],
    }

def detect_motion(prev_gray, curr_gray):
    """Detect motion between frames."""
    if prev_gray is None:
        return 0.0
    
    diff = cv2.absdiff(curr_gray, prev_gray)
    motion = np.mean(diff) / 255.0
    return float(motion)

def extract_rhythmic_patterns(col_profiles, threshold=0.6):
    """Extract rhythmic patterns from column brightness profiles."""
    patterns = []
    
    for profile in col_profiles:
        # Find peaks (bright spots) and valleys (dark spots)
        profile = np.array(profile)
        
        # Normalize profile
        if np.max(profile) > np.min(profile):
            profile_norm = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))
        else:
            profile_norm = np.ones_like(profile) * 0.5
        
        # Detect kick positions (dark spots)
        kick_mask = profile_norm < (1.0 - threshold)
        kick_positions = np.where(kick_mask)[0].tolist()
        
        # Detect snare positions (bright spots)
        snare_mask = profile_norm > threshold
        snare_positions = np.where(snare_mask)[0].tolist()
        
        # Detect hat positions (medium brightness)
        hat_mask = (profile_norm > 0.3) & (profile_norm < 0.7)
        hat_positions = np.where(hat_mask)[0].tolist()
        
        patterns.append({
            'kicks': kick_positions,
            'snares': snare_positions,
            'hats': hat_positions,
            'profile': profile_norm.tolist(),
        })
    
    return patterns

def analyze_melodic_contours(row_profiles):
    """Analyze row profiles for melodic contours."""
    contours = []
    
    for profile in row_profiles:
        profile = np.array(profile)
        
        # Find peaks and valleys
        peaks, _ = sig.find_peaks(profile, height=0.6, distance=2)
        valleys, _ = sig.find_peaks(-profile, height=0.6, distance=2)
        
        # Calculate contour direction
        if len(profile) > 1:
            direction = np.mean(np.diff(profile))
        else:
            direction = 0.0
        
        contours.append({
            'peaks': peaks.tolist(),
            'valleys': valleys.tolist(),
            'direction': float(direction),
            'profile': profile.tolist(),
        })
    
    return contours

def create_visual_guide(frames_data):
    """Create a visual guide from analyzed frames."""
    guide = {
        'summary': {
            'total_frames': len(frames_data),
            'avg_brightness': np.mean([f['brightness'] for f in frames_data]),
            'avg_contrast': np.mean([f['contrast'] for f in frames_data]),
            'avg_edge_density': np.mean([f['edge_density'] for f in frames_data]),
            'avg_hue': np.mean([f['hue'] for f in frames_data]),
        },
        'rhythmic_patterns': extract_rhythmic_patterns([f['col_profile'] for f in frames_data]),
        'melodic_contours': analyze_melodic_contours([f['row_profile'] for f in frames_data]),
        'energy_curve': [f['edge_density'] for f in frames_data],
        'harmonic_progression': [f['hue'] for f in frames_data],
        'frames': frames_data,
    }
    
    return guide

def main():
    samples_dir = Path("/Users/martintomek/CascadeProjects/strudel/my-tracks/video2dnb/samples")
    output_path = Path("/Users/martintomek/CascadeProjects/strudel/my-tracks/video2dnb/visual_guide.json")
    
    print("Analyzing sample frames...")
    
    # Get sorted frame list
    frame_files = sorted([f for f in samples_dir.glob("*.jpg")])
    print(f"Found {len(frame_files)} frames")
    
    frames_data = []
    prev_gray = None
    
    for i, frame_file in enumerate(frame_files):
        if i % 50 == 0:
            print(f"  Processing frame {i+1}/{len(frame_files)}")
        
        # Analyze frame
        data = analyze_frame(str(frame_file))
        if data is None:
            continue
        
        # Detect motion
        img = cv2.imread(str(frame_file))
        small = cv2.resize(img, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        motion = detect_motion(prev_gray, gray)
        data['motion'] = motion
        
        frames_data.append(data)
        prev_gray = gray.copy()
    
    print(f"Analyzed {len(frames_data)} frames")
    
    # Create visual guide
    print("Creating visual guide...")
    guide = create_visual_guide(frames_data)
    
    # Save guide
    with open(output_path, 'w') as f:
        json.dump(guide, f, indent=2)
    
    print(f"Visual guide saved to {output_path}")
    
    # Print summary
    summary = guide['summary']
    print("\nVisual Summary:")
    print(f"  Avg brightness: {summary['avg_brightness']:.3f}")
    print(f"  Avg contrast: {summary['avg_contrast']:.3f}")
    print(f"  Avg edge density: {summary['avg_edge_density']:.3f}")
    print(f"  Avg hue: {summary['avg_hue']:.3f}")
    
    # Analyze rhythmic patterns
    patterns = guide['rhythmic_patterns']
    kick_density = np.mean([len(p['kicks']) for p in patterns])
    snare_density = np.mean([len(p['snares']) for p in patterns])
    hat_density = np.mean([len(p['hats']) for p in patterns])
    
    print(f"\nRhythmic Density (per 16th):")
    print(f"  Kicks: {kick_density:.1f}")
    print(f"  Snares: {snare_density:.1f}")
    print(f"  Hats: {hat_density:.1f}")

if __name__ == "__main__":
    main()
