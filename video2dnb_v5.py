#!/usr/bin/env python3
"""
video2dnb_v5.py — Pattern Library + Multi-Dimensional Image Selection DnB Generator

FAMU experimental sonification project.

Core principle: hardcoded musical patterns (proven to groove) are SELECTED by image
analysis, not generated from image data. The image doesn't create rhythm — it chooses
rhythm from a well-curated library, then modulates the sound.

Usage:
    python video2dnb_v5.py video.mp4 --bpm 174 --output out.wav
    python video2dnb_v5.py video.mp4 --stems          # export individual stems
    python video2dnb_v5.py video.mp4 --groove roller   # override groove type
    python video2dnb_v5.py video.mp4 --key Dm          # override key
"""

import cv2
import numpy as np
import soundfile as sf
from scipy import signal
import argparse, os, sys, time

SR = 48000
DEFAULT_BPM = 174
GROOVE_NAMES = ['halftime', 'twostep', 'roller', 'breakbeat', 'minimal']

# ═══ MUSICAL KEY SYSTEM ══════════════════════════════════════════════════════
KEYS = {
    'Cm': {
        'sub':'c1','mid':'c2',
        'pad':['c3','eb3','g3'],'pad7':['c3','eb3','g3','bb3'],
        'arp':['c4','eb4','g4','c5'],'mel':['c4','eb4','f4','g4','bb4'],
        'freqs':{'c1':32.70,'eb1':38.89,'f1':43.65,'g1':49.00,'ab1':51.91,'bb1':58.27,
                 'c2':65.41,'eb2':77.78,'f2':87.31,'g2':98.00,'ab2':103.83,'bb2':116.54,
                 'c3':130.81,'eb3':155.56,'f3':174.61,'g3':196.00,'ab3':207.65,'bb3':233.08,
                 'c4':261.63,'eb4':311.13,'f4':349.23,'g4':392.00,'ab4':415.30,'bb4':466.16,
                 'c5':523.25,'eb5':622.25,'f5':698.46,'g5':783.99},
    },
    'Dm': {
        'sub':'d1','mid':'d2',
        'pad':['d3','f3','a3'],'pad7':['d3','f3','a3','c4'],
        'arp':['d4','f4','a4','d5'],'mel':['d4','e4','f4','a4','c5'],
        'freqs':{'d1':36.71,'e1':41.20,'f1':43.65,'g1':49.00,'a1':55.00,'bb1':58.27,
                 'd2':73.42,'e2':82.41,'f2':87.31,'g2':98.00,'a2':110.00,'bb2':116.54,'c3':130.81,
                 'd3':146.83,'e3':164.81,'f3':174.61,'g3':196.00,'a3':220.00,'bb3':233.08,
                 'd4':293.66,'e4':329.63,'f4':349.23,'g4':392.00,'a4':440.00,'bb4':466.16,'c4':261.63,'c5':523.25,
                 'd5':587.33,'f5':698.46,'a5':880.00},
    },
    'Em': {
        'sub':'e1','mid':'e2',
        'pad':['e3','g3','b3'],'pad7':['e3','g3','b3','d4'],
        'arp':['e4','g4','b4','e5'],'mel':['e4','f#4','g4','b4','d5'],
        'freqs':{'e1':41.20,'f#1':46.25,'g1':49.00,'a1':55.00,'b1':61.74,
                 'e2':82.41,'f#2':92.50,'g2':98.00,'a2':110.00,'b2':123.47,'c3':130.81,'d3':146.83,
                 'e3':164.81,'f#3':185.00,'g3':196.00,'a3':220.00,'b3':246.94,
                 'e4':329.63,'f#4':369.99,'g4':392.00,'a4':440.00,'b4':493.88,'c4':261.63,'d4':293.66,
                 'e5':659.25,'g5':783.99,'b5':987.77,'d5':587.33},
    },
    'F#m': {
        'sub':'f#1','mid':'f#2',
        'pad':['f#3','a3','c#4'],'pad7':['f#3','a3','c#4','e4'],
        'arp':['f#4','a4','c#5','f#5'],'mel':['f#4','g#4','a4','c#5','e5'],
        'freqs':{'f#1':46.25,'g#1':51.91,'a1':55.00,'b1':61.74,
                 'f#2':92.50,'g#2':103.83,'a2':110.00,'b2':123.47,'c#3':138.59,'d3':146.83,'e3':164.81,
                 'f#3':185.00,'g#3':207.65,'a3':220.00,'b3':246.94,'c#4':277.18,'d4':293.66,'e4':329.63,
                 'f#4':369.99,'g#4':415.30,'a4':440.00,'b4':493.88,'c#5':554.37,'e5':659.25,
                 'f#5':739.99,'a5':880.00},
    },
    'Abm': {
        'sub':'ab1','mid':'ab2',
        'pad':['ab3','b3','eb4'],'pad7':['ab3','b3','eb4','gb4'],
        'arp':['ab4','b4','eb5','ab5'],'mel':['ab4','bb4','b4','eb5','gb5'],
        'freqs':{'ab1':51.91,'bb1':58.27,'b1':61.74,
                 'ab2':103.83,'bb2':116.54,'b2':123.47,'db3':138.59,'eb3':155.56,'e3':164.81,'gb3':185.00,
                 'ab3':207.65,'bb3':233.08,'b3':246.94,'db4':277.18,'eb4':311.13,'e4':329.63,'gb4':369.99,
                 'ab4':415.30,'bb4':466.16,'b4':493.88,'db5':554.37,'eb5':622.25,'gb5':739.99,
                 'ab5':830.61},
    },
    'Bbm': {
        'sub':'bb1','mid':'bb2',
        'pad':['bb3','db4','f4'],'pad7':['bb3','db4','f4','ab4'],
        'arp':['bb4','db5','f5','bb5'],'mel':['bb4','c5','db5','f5','ab5'],
        'freqs':{'bb1':58.27,'c2':65.41,'db2':69.30,
                 'bb2':116.54,'c3':130.81,'db3':138.59,'eb3':155.56,'f3':174.61,'gb3':185.00,'ab3':207.65,
                 'bb3':233.08,'c4':261.63,'db4':277.18,'eb4':311.13,'f4':349.23,'gb4':369.99,'ab4':415.30,
                 'bb4':466.16,'c5':523.25,'db5':554.37,'eb5':622.25,'f5':698.46,'ab5':830.61,
                 'bb5':932.33},
    },
}

# ═══ CHORD PROGRESSIONS — per key, 3 moods ═══════════════════════════════════
# Each entry: [sub_note, mid_note, pad_notes, arp_notes] × 4 bars
def _ch(key, sub, mid, pad, arp):
    """Helper: resolve note names to freqs for a chord."""
    freqs = KEYS[key]['freqs']
    return (freqs.get(sub,32.7), freqs.get(mid,65.4), 
            [freqs.get(n,200) for n in pad], [freqs.get(n,400) for n in arp])

PROGRESSIONS = {}
for k in KEYS:
    PROGRESSIONS[k] = {
        'dark': [], 'neutral': [], 'bright': []
    }

# Cm progressions
PROGRESSIONS['Cm']['dark'] = [
    [_ch('Cm','c1','c2',['c3','eb3','g3'],['c4','eb4','g4','c5']),
     _ch('Cm','c1','c2',['c3','eb3','g3'],['c4','eb4','g4','c5']),
     _ch('Cm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Cm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5'])],
    [_ch('Cm','c1','c2',['c3','eb3','g3'],['c4','eb4','g4','c5']),
     _ch('Cm','f1','f2',['f3','ab3','c4'],['f4','ab4','c5']),
     _ch('Cm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Cm','g1','g2',['g3','bb3','eb4'],['g4','bb4','eb5'])],
]
PROGRESSIONS['Cm']['neutral'] = [
    [_ch('Cm','c1','c2',['c3','eb3','g3'],['c4','eb4','g4','c5']),
     _ch('Cm','c1','c2',['c3','eb3','g3','bb3'],['c4','eb4','g4','bb4']),
     _ch('Cm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Cm','f1','f2',['f3','ab3','c4'],['f4','ab4','c5'])],
    [_ch('Cm','c1','c2',['c3','eb3','g3'],['c4','eb4','g4','c5']),
     _ch('Cm','eb1','eb2',['eb3','g3','bb3'],['eb4','g4','bb4']),
     _ch('Cm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Cm','bb1','bb2',['bb3','eb4','g4'],['bb4','eb5','g5'])],
]
PROGRESSIONS['Cm']['bright'] = [
    [_ch('Cm','c1','c2',['c3','eb3','g3','bb3'],['c4','eb4','g4','bb4']),
     _ch('Cm','eb1','eb2',['eb3','g3','bb3'],['eb4','g4','bb4']),
     _ch('Cm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Cm','bb1','bb2',['bb3','eb4','g4'],['bb4','eb5','g5'])],
]
# Dm progressions
PROGRESSIONS['Dm']['dark'] = [
    [_ch('Dm','d1','d2',['d3','f3','a3'],['d4','f4','a4','d5']),
     _ch('Dm','d1','d2',['d3','f3','a3'],['d4','f4','a4','d5']),
     _ch('Dm','bb1','bb2',['bb3','d4','f4'],['bb4','d5','f5']),
     _ch('Dm','bb1','bb2',['bb3','d4','f4'],['bb4','d5','f5'])],
]
PROGRESSIONS['Dm']['neutral'] = [
    [_ch('Dm','d1','d2',['d3','f3','a3'],['d4','f4','a4','d5']),
     _ch('Dm','f1','f2',['f3','a3','c4'],['f4','a4','c5']),
     _ch('Dm','bb1','bb2',['bb3','d4','f4'],['bb4','d5','f5']),
     _ch('Dm','g1','g2',['g3','bb3','d4'],['g4','bb4','d5'])],
]
PROGRESSIONS['Dm']['bright'] = [
    [_ch('Dm','f1','f2',['f3','a3','c4'],['f4','a4','c5']),
     _ch('Dm','g1','g2',['g3','bb3','d4'],['g4','bb4','d5']),
     _ch('Dm','bb1','bb2',['bb3','d4','f4'],['bb4','d5','f5']),
     _ch('Dm','d1','d2',['d3','f3','a3'],['d4','f4','a4','d5'])],
]
# Em progressions
PROGRESSIONS['Em']['dark'] = [
    [_ch('Em','e1','e2',['e3','g3','b3'],['e4','g4','b4','e5']),
     _ch('Em','e1','e2',['e3','g3','b3'],['e4','g4','b4','e5']),
     _ch('Em','c3','c4',['c3','e3','g3'],['c4','e4','g4']),
     _ch('Em','a1','a2',['a3','c4','e4'],['a4','c5','e5'])],
]
PROGRESSIONS['Em']['neutral'] = [
    [_ch('Em','e1','e2',['e3','g3','b3'],['e4','g4','b4','e5']),
     _ch('Em','c3','c4',['c3','e3','g3'],['c4','e4','g4']),
     _ch('Em','a1','a2',['a3','c4','e4'],['a4','c5','e5']),
     _ch('Em','b1','b2',['b3','d4','f#4'],['b4','d5','f#5'])],
]
PROGRESSIONS['Em']['bright'] = [
    [_ch('Em','c3','c4',['c3','e3','g3'],['c4','e4','g4']),
     _ch('Em','g1','g2',['g3','b3','d4'],['g4','b4','d5']),
     _ch('Em','a1','a2',['a3','c4','e4'],['a4','c5','e5']),
     _ch('Em','e1','e2',['e3','g3','b3'],['e4','g4','b4','e5'])],
]
# F#m progressions
PROGRESSIONS['F#m']['dark'] = [
    [_ch('F#m','f#1','f#2',['f#3','a3','c#4'],['f#4','a4','c#5','f#5']),
     _ch('F#m','f#1','f#2',['f#3','a3','c#4'],['f#4','a4','c#5','f#5']),
     _ch('F#m','d3','d4',['d3','f#3','a3'],['d4','f#4','a4']),
     _ch('F#m','b1','b2',['b3','d4','f#4'],['b4','d5','f#5'])],
]
PROGRESSIONS['F#m']['neutral'] = [
    [_ch('F#m','f#1','f#2',['f#3','a3','c#4'],['f#4','a4','c#5','f#5']),
     _ch('F#m','d3','d4',['d3','f#3','a3'],['d4','f#4','a4']),
     _ch('F#m','e3','e4',['e3','g#3','b3'],['e4','g#4','b4']),
     _ch('F#m','c#3','c#4',['c#3','e3','g#3'],['c#4','e4','g#4'])],
]
PROGRESSIONS['F#m']['bright'] = [
    [_ch('F#m','d3','d4',['d3','f#3','a3'],['d4','f#4','a4']),
     _ch('F#m','e3','e4',['e3','g#3','b3'],['e4','g#4','b4']),
     _ch('F#m','f#1','f#2',['f#3','a3','c#4'],['f#4','a4','c#5']),
     _ch('F#m','a1','a2',['a3','c#4','e4'],['a4','c#5','e5'])],
]
# Abm progressions
PROGRESSIONS['Abm']['dark'] = [
    [_ch('Abm','ab1','ab2',['ab3','b3','eb4'],['ab4','b4','eb5']),
     _ch('Abm','ab1','ab2',['ab3','b3','eb4'],['ab4','b4','eb5']),
     _ch('Abm','e3','e4',['e3','ab3','b3'],['e4','ab4','b4']),
     _ch('Abm','gb3','gb4',['gb3','bb3','db4'],['gb4','bb4','db5'])],
]
PROGRESSIONS['Abm']['neutral'] = [
    [_ch('Abm','ab1','ab2',['ab3','b3','eb4'],['ab4','b4','eb5']),
     _ch('Abm','e3','e4',['e3','ab3','b3'],['e4','ab4','b4']),
     _ch('Abm','b2','b3',['b3','eb4','gb4'],['b4','eb5','gb5']),
     _ch('Abm','gb3','gb4',['gb3','bb3','db4'],['gb4','bb4','db5'])],
]
PROGRESSIONS['Abm']['bright'] = [
    [_ch('Abm','e3','e4',['e3','ab3','b3'],['e4','ab4','b4']),
     _ch('Abm','b2','b3',['b3','eb4','gb4'],['b4','eb5','gb5']),
     _ch('Abm','ab1','ab2',['ab3','b3','eb4'],['ab4','b4','eb5']),
     _ch('Abm','e3','e4',['e3','ab3','b3'],['e4','ab4','b4'])],
]
# Bbm progressions
PROGRESSIONS['Bbm']['dark'] = [
    [_ch('Bbm','bb1','bb2',['bb3','db4','f4'],['bb4','db5','f5']),
     _ch('Bbm','bb1','bb2',['bb3','db4','f4'],['bb4','db5','f5']),
     _ch('Bbm','gb3','gb4',['gb3','bb3','db4'],['gb4','bb4','db5']),
     _ch('Bbm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5'])],
]
PROGRESSIONS['Bbm']['neutral'] = [
    [_ch('Bbm','bb1','bb2',['bb3','db4','f4'],['bb4','db5','f5']),
     _ch('Bbm','gb3','gb4',['gb3','bb3','db4'],['gb4','bb4','db5']),
     _ch('Bbm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Bbm','eb3','eb4',['eb3','gb3','bb3'],['eb4','gb4','bb4'])],
]
PROGRESSIONS['Bbm']['bright'] = [
    [_ch('Bbm','gb3','gb4',['gb3','bb3','db4'],['gb4','bb4','db5']),
     _ch('Bbm','ab1','ab2',['ab3','c4','eb4'],['ab4','c5','eb5']),
     _ch('Bbm','bb1','bb2',['bb3','db4','f4'],['bb4','db5','f5']),
     _ch('Bbm','db3','db4',['db3','f3','ab3'],['db4','f4','ab4'])],
]

# ═══ PATTERN LIBRARY ═════════════════════════════════════════════════════════
# PATTERNS[groove][instrument][density] = [varA, varB]
# 16-step arrays. Kick: beat1+beat3 area. Snare: steps 4,12.
P = {}

# ─── HALFTIME: sparse, heavy ────────────────────────────────────────────
P['halftime'] = {
 'kick':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
  4:[[1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]],
  5:[[1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0],[1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0]],
 },
 'snare':{
  1:[[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
  2:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  3:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1]],
  4:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],[0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0]],
  5:[[0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0],[0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0]],
 },
 'ghost':{
  1:[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  3:[[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]],
  4:[[0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1]],
  5:[[0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1],[0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1]],
 },
 'hat':{
  1:[[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  2:[[0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]],
  3:[[1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0],[0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0]],
  4:[[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],[1,0,1,1,0,0,1,0,1,0,1,0,0,0,1,0]],
  5:[[1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1],[1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0]],
 },
 'hat_vel':{
  1:[.3]*16, 2:[.4,.2,.3,.2]*4, 3:[.5,.2,.35,.2]*4,
  4:[.6,.25,.4,.25]*4, 5:[.7,.3,.5,.3]*4,
 },
 'bass':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0]],
  4:[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0]],
  5:[[1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0]],
 },
}
# ─── TWOSTEP: classic DnB ───────────────────────────────────────────────
P['twostep'] = {
 'kick':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]],
  3:[[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0]],
  4:[[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0]],
  5:[[1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0]],
 },
 'snare':{
  1:[[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
  2:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  3:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1]],
  4:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],[0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0]],
  5:[[0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1],[0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0]],
 },
 'ghost':{
  1:[[0]*16,[0]*16],
  2:[[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  3:[[0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]],
  4:[[0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1],[0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0]],
  5:[[0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,1],[0,1,0,1,0,1,1,0,0,0,1,0,0,1,1,0]],
 },
 'hat':{
  1:[[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  2:[[1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0],[0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0]],
  3:[[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],[1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0]],
  4:[[1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0],[1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0]],
  5:[[1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1],[1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0]],
 },
 'hat_vel':{
  1:[.3]*16, 2:[.5,.2,.35,.2]*4, 3:[.6,.25,.4,.25]*4,
  4:[.7,.25,.45,.3,.65,.25,.45,.3]*2, 5:[.8,.3,.5,.35,.75,.3,.5,.35]*2,
 },
 'bass':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]],
  4:[[1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]],
  5:[[1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0],[1,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0]],
 },
}
# ─── ROLLER: relentless rolling breaks ──────────────────────────────────
P['roller'] = {
 'kick':{
  1:[[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0]],
  4:[[1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0]],
  5:[[1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0],[1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0]],
 },
 'snare':{
  1:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  2:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  3:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1]],
  4:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],[0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0]],
  5:[[0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0],[0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1]],
 },
 'ghost':{
  1:[[0]*16,[0]*16],
  2:[[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]],
  3:[[0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],[0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1]],
  4:[[0,1,1,0,0,0,1,0,0,1,1,0,0,0,1,0],[0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,0]],
  5:[[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],[0,1,1,1,0,0,1,1,0,1,1,0,0,1,1,0]],
 },
 'hat':{
  1:[[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]],
  2:[[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],[1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0]],
  3:[[1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0],[1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0]],
  4:[[1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0],[1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0]],
  5:[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
 },
 'hat_vel':{
  1:[.5,.25,.4,.25]*4, 2:[.6,.25,.45,.25]*4, 3:[.7,.3,.5,.3]*4,
  4:[.75,.35,.55,.35]*4, 5:[.8,.4,.6,.35,.75,.35,.55,.4]*2,
 },
 'bass':{
  1:[[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0]],
  3:[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],[1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0]],
  4:[[1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0],[1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0]],
  5:[[1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0],[1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0]],
 },
}
# ─── BREAKBEAT: syncopated, complex ─────────────────────────────────────
P['breakbeat'] = {
 'kick':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],
  4:[[1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]],
  5:[[1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0],[1,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0]],
 },
 'snare':{
  1:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
  2:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  3:[[0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  4:[[0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0]],
  5:[[0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0],[0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,1]],
 },
 'ghost':{
  1:[[0]*16,[0]*16],
  2:[[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
  3:[[0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0]],
  4:[[0,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0],[0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1]],
  5:[[0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0],[0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1]],
 },
 'hat':{
  1:[[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]],
  2:[[1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],[0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0]],
  3:[[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],[1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1]],
  4:[[1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0],[1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1]],
  5:[[1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1],[1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1]],
 },
 'hat_vel':{
  1:[.35]*16, 2:[.5,.2,.4,.25]*4,
  3:[.6,.2,.45,.3,.55,.2,.45,.25]*2,
  4:[.7,.3,.5,.25,.65,.3,.5,.35]*2, 5:[.8,.35,.6,.3,.75,.35,.55,.4]*2,
 },
 'bass':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]],
  4:[[1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0],[1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0]],
  5:[[1,0,0,1,0,1,1,0,0,0,1,1,0,0,0,0],[1,0,1,0,0,0,1,1,0,1,0,1,0,0,0,0]],
 },
}
# ─── MINIMAL: stripped back ─────────────────────────────────────────────
P['minimal'] = {
 'kick':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]],
  4:[[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
  5:[[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0]],
 },
 'snare':{
  1:[[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
  2:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
  3:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]],
  4:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1]],
  5:[[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],[0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0]],
 },
 'ghost':{
  1:[[0]*16,[0]*16], 2:[[0]*16,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],
  3:[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  4:[[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]],
  5:[[0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1]],
 },
 'hat':{
  1:[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
  2:[[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0]],
  3:[[0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]],
  4:[[1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0],[1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0]],
  5:[[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]],
 },
 'hat_vel':{
  1:[.25]*16, 2:[.35,.15,.25,.15]*4, 3:[.4,.2,.3,.2]*4,
  4:[.5,.2,.35,.2]*4, 5:[.55,.25,.4,.25]*4,
 },
 'bass':{
  1:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  2:[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]],
  3:[[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
  4:[[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0]],
  5:[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0]],
 },
}

# ═══ SYNTHESIS PRIMITIVES ═════════════════════════════════════════════════════

def make_kick(sr=SR, duration=0.15, gain=1.0, body=1.0, click_amount=1.0):
    n = int(sr * duration); t = np.arange(n) / sr
    decay = 12 / max(body, 0.3)
    pitch_env = 150 * np.exp(-t * 40) + 40
    phase = np.cumsum(2 * np.pi * pitch_env / sr)
    osc = np.sin(phase) * np.exp(-t * decay)
    click = np.exp(-t * 200) * 0.5 * click_amount
    return np.tanh((osc + click) * 1.5 * gain) * 0.9

def make_snare(sr=SR, duration=0.1, gain=1.0, tone_freq=200, crack=1.0):
    n = int(sr * duration); t = np.arange(n) / sr
    noise = np.random.randn(n) * np.exp(-t * 20)
    hi = min(4000 + crack * 4000, sr/2 - 100)
    sos = signal.butter(2, [2000, hi], btype='band', fs=sr, output='sos')
    nf = signal.sosfilt(sos, noise)
    tone = np.sin(2 * np.pi * tone_freq * t) * np.exp(-t * 30)
    return np.tanh((nf * 0.7 * crack + tone * 0.5) * gain * 2) * 0.7

def make_ghost(sr=SR, gain=0.15):
    return make_snare(sr=sr, duration=0.06, gain=gain, crack=0.6)

def make_hihat(sr=SR, gain=0.3, is_open=False, brightness=1.0):
    dur = 0.15 if is_open else 0.04
    n = int(sr * dur); t = np.arange(n) / sr
    noise = np.random.randn(n)
    env = np.exp(-t * (8 if is_open else 40))
    cutoff = max(3000, min(4000 + brightness * 4000, sr/2 - 100))
    sos = signal.butter(2, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, noise) * env * gain

def make_click(sr=SR, gain=0.4, freq=4000):
    n = int(sr * 0.008); t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 500) * gain

def make_impact(sr=SR, gain=0.8):
    n = int(sr * 0.4); t = np.arange(n) / sr
    boom = np.sin(2*np.pi*40*t*np.exp(-t*3)) * np.exp(-t*5) * 0.8
    noise = np.random.randn(n) * np.exp(-t*8) * 0.3
    sos = signal.butter(2, 800, btype='low', fs=sr, output='sos')
    return (boom + signal.sosfilt(sos, noise)) * gain

def make_riser(sr=SR, duration=1.0, gain=0.3):
    n = int(sr * duration); t = np.arange(n) / sr
    noise = np.random.randn(n)
    sweep = 200 + (t/duration)**2 * 8000
    chunk = max(n//32, 64); out = np.zeros(n)
    for i in range(0, n, chunk):
        e = min(i+chunk, n)
        c = np.clip(np.mean(sweep[i:e]), 20, sr/2-100)
        sos = signal.butter(2, c, btype='low', fs=sr, output='sos')
        out[i:e] = signal.sosfilt(sos, noise[i:e])
    return out * np.linspace(0.1, 1.0, n)**2 * gain

def make_saw(freq, n_samples, sr=SR, num_harmonics=15, gain=1.0):
    t = np.arange(n_samples) / sr; wave = np.zeros(n_samples)
    for k in range(1, num_harmonics+1):
        if k*freq > sr/2: break
        wave += np.sin(2*np.pi*k*freq*t) / k
    return wave * gain * (2.0/np.pi)

def apply_lp(audio, cutoff, sr=SR):
    cutoff = np.clip(cutoff, 20, sr/2-100)
    return signal.sosfilt(signal.butter(2, cutoff, btype='low', fs=sr, output='sos'), audio)

def apply_hp(audio, cutoff, sr=SR):
    cutoff = np.clip(cutoff, 20, sr/2-100)
    return signal.sosfilt(signal.butter(2, cutoff, btype='high', fs=sr, output='sos'), audio)

def comb_reverb(audio, sr=SR, decay=0.3, delays_ms=(23,37,49,61)):
    out = audio.copy()
    for d_ms in delays_ms:
        d = int(d_ms*sr/1000)
        if d < len(audio):
            delayed = np.zeros_like(audio); delayed[d:] = audio[:-d]
            out += delayed * decay * 0.25
    return out

def env_adsr(n, sr=SR, a=0.01, d=0.05, s=0.7, r=0.05):
    ai,di,ri = int(a*sr),int(d*sr),int(r*sr); env = np.zeros(n); pos=0
    seg=min(ai,n)
    if seg>0: env[pos:pos+seg]=np.linspace(0,1,seg)
    pos+=seg; seg=min(di,n-pos)
    if seg>0: env[pos:pos+seg]=np.linspace(1,s,seg)
    pos+=seg; sus=max(0,n-pos-ri)
    if sus>0: env[pos:pos+sus]=s
    pos+=sus; seg=min(ri,n-pos)
    if seg>0: env[pos:pos+seg]=np.linspace(s,0,seg)
    return env

def place(buf, sound, pos):
    end = min(pos+len(sound), len(buf))
    ln = end-pos
    if ln>0 and pos>=0: buf[pos:end] += sound[:ln]


# ═══ VIDEO DNA FINGERPRINT ═══════════════════════════════════════════════════

def compute_video_dna(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Cannot open '{video_path}'"); sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    duration = total_frames / fps
    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    brights,contrasts,sats,hues_list,edges_list,motions = [],[],[],[],[],[]
    reds,greens,blues,sharps = [],[],[],[]
    frame_analyses = []; prev_small = None; idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        fa = {}
        fa['brightness'] = float(np.mean(gray))/255.0
        fa['contrast'] = min(float(np.std(gray))/128.0, 1.0)
        fa['red'] = float(np.mean(small[:,:,2]))/255.0
        fa['green'] = float(np.mean(small[:,:,1]))/255.0
        fa['blue'] = float(np.mean(small[:,:,0]))/255.0
        fa['saturation'] = float(np.mean(hsv[:,:,1]))/255.0
        edg = cv2.Canny(gray, 50, 150)
        fa['edge_density'] = float(np.sum(edg>0))/float(edg.size)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        fa['sharpness'] = min(float(np.var(lap))/5000.0, 1.0)
        sat_mask = hsv[:,:,1] > 20
        fa['hue'] = float(np.median(hsv[:,:,0][sat_mask]))*2.0 if np.any(sat_mask) else 0.0
        fa['hue_norm'] = fa['hue']/360.0
        warm = fa['red']+fa['green']*0.3; cool = fa['blue']+fa['green']*0.3
        fa['warmth'] = warm/(warm+cool) if (warm+cool)>0 else 0.5
        if prev_small is not None:
            fa['motion'] = float(np.mean(cv2.absdiff(gray, cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY))))/255.0
        else: fa['motion'] = 0.0
        h,w = gray.shape
        top=float(np.mean(gray[:h//2,:]))/255.0; bot=float(np.mean(gray[h//2:,:]))/255.0
        fa['vertical_balance'] = top/(top+bot) if (top+bot)>0 else 0.5
        left=float(np.mean(gray[:,:w//2]))/255.0; right_v=float(np.mean(gray[:,w//2:]))/255.0
        fa['horizontal_balance'] = left/(left+right_v) if (left+right_v)>0 else 0.5
        ctr = gray[h//4:3*h//4, w//4:3*w//4]
        fa['center_weight'] = min(float(np.mean(ctr))/255.0/max(fa['brightness'],0.01), 2.0)/2.0
        n_bands = 64; ri2 = np.linspace(0, h-1, n_bands).astype(int)
        fa['spectral_amps'] = np.mean(gray[ri2[::-1],:], axis=1)/255.0
        fa['spectral_freqs'] = np.logspace(np.log10(80), np.log10(8000), n_bands)
        fa['energy'] = 0.25*fa['brightness']+0.25*fa['motion']+0.20*fa['edge_density']+0.15*fa['contrast']+0.15*fa['saturation']
        frame_analyses.append(fa)
        brights.append(fa['brightness']); contrasts.append(fa['contrast'])
        sats.append(fa['saturation']); hues_list.append(fa['hue'])
        edges_list.append(fa['edge_density']); motions.append(fa['motion'])
        reds.append(fa['red']); greens.append(fa['green']); blues.append(fa['blue'])
        sharps.append(fa['sharpness'])
        prev_small = small; idx += 1
        if idx % 30 == 0:
            pct = idx/total_frames*100
            print(f"\r  Analyzing [{'#'*int(pct/2.5):40s}] {pct:.0f}%", end="", flush=True)
    cap.release()
    print(f"\r  Analyzing [{'#'*40}] 100%")
    print(f"  {len(frame_analyses)} frames analyzed")

    # Normalize energy
    energies = [f['energy'] for f in frame_analyses]
    e_min,e_max = min(energies),max(energies); e_rng = e_max-e_min if e_max>e_min else 1.0
    for f in frame_analyses: f['energy_norm'] = (f['energy']-e_min)/e_rng

    # Scene cuts
    motions_arr = np.array(motions)
    sc_thresh = np.mean(motions_arr)+2.5*np.std(motions_arr) if len(motions_arr)>1 else 0.5
    scene_cuts = []
    for i in range(len(frame_analyses)):
        if i>0 and motions_arr[i]>sc_thresh:
            scene_cuts.append(i); frame_analyses[i]['is_scene_cut'] = True
        else:
            frame_analyses[i]['is_scene_cut'] = False

    sats_arr = np.array(sats); hues_arr = np.array(hues_list)
    sat_hues = hues_arr[sats_arr > 0.1]
    brights_arr = np.array(brights)

    # Motion rhythmicity
    if len(motions_arr)>10:
        mn = motions_arr-np.mean(motions_arr)
        ac = np.correlate(mn, mn, mode='full'); ac = ac[len(ac)//2:]
        if ac[0]>0: ac /= ac[0]
        mr = float(np.max(ac[1:min(len(ac),120)])) if len(ac)>1 else 0.0
    else: mr = 0.0

    hue_spread = min(float(np.std(sat_hues))/180.0, 1.0) if len(sat_hues)>10 else 0.0
    if len(brights_arr)>2:
        slope = float(np.polyfit(np.arange(len(brights_arr)), brights_arr, 1)[0])*len(brights_arr)
        bt = np.clip(slope*5+0.5, 0, 1)
    else: bt = 0.5

    dna = {
        'avg_brightness': float(np.mean(brights_arr)),
        'brightness_variance': float(np.std(brights_arr)),
        'brightness_trend': bt,
        'dominant_hue': float(np.median(sat_hues)) if len(sat_hues)>0 else 0.0,
        'hue_spread': hue_spread,
        'saturation_avg': float(np.mean(sats_arr)),
        'warmth_ratio': float(np.mean([f['warmth'] for f in frame_analyses])),
        'chrominance_energy': float(np.mean(sats_arr)),
        'edge_density_avg': float(np.mean(edges_list)),
        'sharpness': float(np.mean(sharps)),
        'contrast_avg': float(np.mean(contrasts)),
        'motion_avg': float(np.mean(motions_arr)),
        'motion_variance': float(np.std(motions_arr)),
        'scene_cut_rate': min(len(scene_cuts)/(duration/60.0)/30.0, 1.0) if duration>0 else 0,
        'motion_rhythmicity': min(mr, 1.0),
        'vertical_balance': float(np.mean([f['vertical_balance'] for f in frame_analyses])),
        'horizontal_balance': float(np.mean([f['horizontal_balance'] for f in frame_analyses])),
        'center_weight': float(np.mean([f['center_weight'] for f in frame_analyses])),
        'scene_cuts': scene_cuts,
    }
    return dna, frame_analyses, fps, duration, total_frames


# ═══ MUSICAL DECISION ENGINE ═════════════════════════════════════════════════

def select_groove(dna, override=None):
    if override and override in GROOVE_NAMES: return override
    s = dna['saturation_avg']; e = dna['edge_density_avg']
    m = dna['motion_avg']; hs = dna['hue_spread']
    c = dna['contrast_avg']; b = dna['avg_brightness']
    scores = {
        'halftime': (1-s)*0.3+(1-b)*0.3+(1-m)*0.2+(1-e)*0.2,
        'twostep': (0.5-abs(s-0.4))*0.3+dna['warmth_ratio']*0.2+(0.5-abs(m-0.3))*0.2+(1-abs(c-0.4))*0.3,
        'roller': e*0.35+m*0.35+dna['motion_rhythmicity']*0.15+c*0.15,
        'breakbeat': hs*0.35+s*0.25+e*0.2+dna['scene_cut_rate']*0.2,
        'minimal': (1-c)*0.3+(1-s)*0.25+(1-m)*0.25+(1-e)*0.2,
    }
    return max(scores, key=scores.get)

def select_key(dna, override=None):
    if override:
        for k in KEYS:
            if override.lower().replace(' ','') in k.lower(): return k
        return 'Cm'
    if dna['saturation_avg'] < 0.12: return 'Cm'
    h = dna['dominant_hue']
    if h<60: return 'Cm'
    elif h<120: return 'Dm'
    elif h<180: return 'Em'
    elif h<240: return 'F#m'
    elif h<300: return 'Abm'
    else: return 'Bbm'

def select_mood(dna):
    score = dna['warmth_ratio']*0.4 + dna['brightness_trend']*0.3 + dna['avg_brightness']*0.3
    if score<0.4: return 'dark'
    elif score<0.6: return 'neutral'
    else: return 'bright'

def select_progression(key_name, mood, dna):
    progs = PROGRESSIONS.get(key_name, PROGRESSIONS['Cm']).get(mood, PROGRESSIONS['Cm']['neutral'])
    if not progs: progs = PROGRESSIONS['Cm']['neutral']
    idx = int(dna['brightness_variance']*10) % len(progs)
    return progs[idx]


# ═══ ARRANGEMENT ═════════════════════════════════════════════════════════════

def compute_arrangement(frame_analyses, fps, duration, bpm, dna):
    beat_dur = 60.0/bpm; bar_dur = beat_dur*4
    total_bars = int(duration/bar_dur)
    if total_bars == 0: return {}
    bar_features = []
    for bar in range(total_bars):
        fs = int(bar*bar_dur*fps); fe = int((bar+1)*bar_dur*fps)
        fs = max(0, min(fs, len(frame_analyses)-1))
        fe = max(fs+1, min(fe, len(frame_analyses)))
        fas = frame_analyses[fs:fe]
        bar_features.append({
            'energy': np.mean([f['energy_norm'] for f in fas]),
            'motion': np.mean([f['motion'] for f in fas]),
            'has_cut': any(f.get('is_scene_cut', False) for f in fas),
        })
    arr = {}
    intro_bars = max(2, min(8, total_bars//6))
    outro_bars = max(2, min(8, total_bars//6))
    for bar in range(total_bars):
        st = {'gain':1.0, 'density_offset':0, 'section':'main',
              'is_buildup':False, 'is_drop':False, 'is_breakdown':False}
        if bar < intro_bars:
            st['gain'] = 0.15+0.85*(bar/intro_bars)
            st['density_offset'] = -2 if bar<intro_bars//2 else -1
            st['section'] = 'intro'
        elif bar >= total_bars-outro_bars:
            rem = total_bars-bar
            st['gain'] = 0.15+0.85*(rem/outro_bars)
            st['density_offset'] = -1 if rem<outro_bars//2 else 0
            st['section'] = 'outro'
        else:
            bf = bar_features[bar]
            if bf['energy']<0.25 and bf['motion']<0.1:
                st['is_breakdown']=True; st['density_offset']=-2; st['section']='breakdown'
            elif bar>=2:
                prev_e = [bar_features[b]['energy'] for b in range(max(0,bar-3),bar)]
                if len(prev_e)>=2 and all(prev_e[i]<prev_e[i+1] for i in range(len(prev_e)-1)) and bf['energy']>0.5:
                    st['is_buildup']=True; st['density_offset']=1; st['section']='buildup'
            if bf['has_cut'] and bf['energy']>0.6:
                st['is_drop']=True; st['density_offset']=1; st['section']='drop'
        arr[bar] = st
    return arr


# ═══ SPECTRAL TEXTURE ════════════════════════════════════════════════════════

def render_spectral_texture(frame_analyses, fps, duration, sr=SR, seed=42):
    rng = np.random.RandomState(seed)
    total_samples = int(sr*duration); nf = len(frame_analyses)
    spf = total_samples//nf if nf>0 else 1
    n_bands = 64; phases = np.zeros(n_bands)
    out_L = np.zeros(total_samples+sr); out_R = np.zeros(total_samples+sr)
    for fi in range(nf):
        fa = frame_analyses[fi]
        freqs = fa['spectral_freqs']; amps = fa['spectral_amps']
        cs = fi*spf; ce = min(cs+spf, total_samples); cl = ce-cs
        if cl<=0: continue
        tex = np.zeros(cl)
        for i in range(n_bands):
            if amps[i]<0.04: continue
            freq = freqs[i]
            if freq>sr/2: continue
            pi2 = 2*np.pi*freq/sr
            ph = phases[i]+np.cumsum(np.ones(cl)*pi2)
            tex += np.sin(ph)*amps[i]; phases[i] = ph[-1]%(2*np.pi)
        pk = np.max(np.abs(tex))
        if pk>0: tex /= pk
        g = 0.04+fa['green']*0.08+fa['brightness']*0.04
        out_L[cs:ce] += tex*g*0.7; out_R[cs:ce] += tex*g*0.5
        d = int(0.005*sr)
        if ce+d <= len(out_R): out_R[cs+d:ce+d] += tex*g*0.3
    return out_L[:total_samples], out_R[:total_samples]


# ═══ MAIN RENDER ENGINE ═════════════════════════════════════════════════════

def render_video_to_dnb(video_path, bpm=DEFAULT_BPM, output_path=None,
                        groove_override=None, key_override=None,
                        no_sonification=False, sonification_gain=0.35,
                        export_stems=False, sr=SR):

    # 1. Analyze
    print("[1/6] Computing Video DNA fingerprint...")
    dna, frame_analyses, fps, duration, total_frames = compute_video_dna(video_path)

    # 2. Musical decisions
    print("\n[2/6] Making musical decisions...")
    groove = select_groove(dna, groove_override)
    key_name = select_key(dna, key_override)
    mood = select_mood(dna)
    progression = select_progression(key_name, mood, dna)
    key_freqs = KEYS[key_name]['freqs']
    dna_seed = int(abs(dna['avg_brightness']*1000+dna['dominant_hue']*100+dna['motion_avg']*10000))%(2**31)
    rng = np.random.RandomState(dna_seed)

    # Diagnostic report
    print(f"\n  {'─'*50}")
    print(f"  VIDEO DNA FINGERPRINT")
    print(f"  {'─'*50}")
    print(f"  Luminance:  bright={dna['avg_brightness']:.2f}  var={dna['brightness_variance']:.3f}  trend={dna['brightness_trend']:.2f}")
    print(f"  Color:      hue={dna['dominant_hue']:.0f}°  spread={dna['hue_spread']:.2f}  sat={dna['saturation_avg']:.2f}  warm={dna['warmth_ratio']:.2f}")
    print(f"  Texture:    edges={dna['edge_density_avg']:.3f}  sharp={dna['sharpness']:.3f}  contrast={dna['contrast_avg']:.2f}")
    print(f"  Motion:     avg={dna['motion_avg']:.3f}  var={dna['motion_variance']:.3f}  cuts/min={dna['scene_cut_rate']*30:.1f}  rhythm={dna['motion_rhythmicity']:.2f}")
    print(f"  Spatial:    vert={dna['vertical_balance']:.2f}  horiz={dna['horizontal_balance']:.2f}  center={dna['center_weight']:.2f}")
    print(f"  {'─'*50}")
    print(f"  MUSICAL CHOICES")
    print(f"  {'─'*50}")
    print(f"  Groove:     {groove.upper()}")
    print(f"  Key:        {key_name}")
    print(f"  Mood:       {mood}")
    print(f"  Scene cuts: {len(dna['scene_cuts'])}")
    print(f"  DNA seed:   {dna_seed}")
    print(f"  {'─'*50}")

    # 3. Setup
    total_samples = int(duration*sr); tail = sr; buf_len = total_samples+tail
    beat_dur = 60.0/bpm; s16 = int(beat_dur/4*sr); s16_sec = beat_dur/4
    bar_samples = s16*16; total_bars = int(duration/(beat_dur*4))
    print(f"\n  BPM: {bpm}, 16th={s16_sec*1000:.1f}ms, bars={total_bars}")
    nf = len(frame_analyses)
    def fa_at(pos):
        t = pos/sr; idx = min(int(t*fps), nf-1); return frame_analyses[max(0,idx)]

    arrangement = compute_arrangement(frame_analyses, fps, duration, bpm, dna)
    gp = P[groove]

    # 4. Render
    print("\n[3/6] Rendering DnB layers...")
    kick_bus=np.zeros(buf_len); snare_bus=np.zeros(buf_len); hat_bus=np.zeros(buf_len)
    click_bus=np.zeros(buf_len); sub_bus=np.zeros(buf_len); reese_bus=np.zeros(buf_len)
    pad_bus_L=np.zeros(buf_len); pad_bus_R=np.zeros(buf_len)
    arp_bus=np.zeros(buf_len); mel_bus=np.zeros(buf_len)
    impact_bus=np.zeros(buf_len); riser_bus=np.zeros(buf_len)

    # ─── DRUMS ───────────────────────────────────────────────────────
    print("  Drums...")
    impact_snd = make_impact(sr=sr)
    pos=0; beat_idx=0
    while pos < total_samples:
        s16b = beat_idx%16; bar_num = beat_idx//16
        fa = fa_at(pos); e = fa['energy_norm']
        arr = arrangement.get(bar_num, {'gain':1.0,'density_offset':0,'section':'main',
                                        'is_buildup':False,'is_drop':False,'is_breakdown':False})
        ag = arr['gain']
        dd = int(np.clip(round(e*4)+1+arr['density_offset'], 1, 5))
        vi = 0 if fa['brightness']<0.5 else 1
        kp = gp['kick'][dd][vi]; sp = gp['snare'][dd][vi]
        ghp = gp['ghost'][dd][vi]; hp = gp['hat'][dd][vi]
        hv = gp['hat_vel'][dd]; bp = gp['bass'][dd][vi]

        # Kick
        if kp[s16b]:
            body = 0.6+fa['brightness']*0.8; ca = 0.5+fa['sharpness']*1.5
            g = (0.7+e*0.3)*ag
            if fa['motion']>0.1: g *= 1.0+fa['motion']*0.5
            place(kick_bus, make_kick(sr=sr, gain=g, body=body, click_amount=ca), pos)
        # Snare
        if sp[s16b]:
            tf = 180+fa['edge_density']*80; cr = 0.6+fa['contrast']*0.8
            g = (0.6+e*0.4)*ag
            place(snare_bus, make_snare(sr=sr, gain=g, tone_freq=tf, crack=cr), pos)
        elif ghp[s16b]:
            g = (0.08+fa['edge_density']*0.12)*ag
            place(snare_bus, make_ghost(sr=sr, gain=g), pos)
        # Hat
        if hp[s16b]:
            is_open = s16b in (6,14) and e>0.5
            vel = hv[s16b%len(hv)]
            br = fa['blue']*2.0
            # Swing: horizontal balance shifts off-beat timing
            swing_offset = 0
            if s16b%2==1:
                swing_amt = (fa['horizontal_balance']-0.5)*0.3
                swing_offset = int(swing_amt*s16)
            h = make_hihat(sr=sr, gain=vel*ag*0.6, is_open=is_open, brightness=br)
            place(hat_bus, h, pos+swing_offset)
        # Click transient from sharpness
        if fa['sharpness']>0.3 and s16b%2==0:
            cg = (fa['sharpness']-0.3)*0.6*ag
            place(click_bus, make_click(sr=sr, gain=cg, freq=3000+fa['edge_density']*5000), pos)
        # Scene cut impact
        if fa.get('is_scene_cut', False) and s16b==0:
            place(impact_bus, impact_snd*ag, pos)
            # Also place a riser ~1 bar before this cut
            riser_pos = pos - bar_samples
            if riser_pos > 0:
                rs = make_riser(sr=sr, duration=beat_dur*4, gain=0.2*ag)
                place(riser_bus, rs, riser_pos)

        beat_idx+=1; pos+=s16

    # ─── SUB BASS ────────────────────────────────────────────────────
    print("  Sub bass...")
    sub_phase = 0.0
    for cs in range(0, total_samples, s16):
        ce = min(cs+s16, total_samples); cl = ce-cs
        if cl<=0: continue
        fa = fa_at(cs); s16b = (cs//s16)%16
        bar_num = (cs//s16)//16
        arr = arrangement.get(bar_num, {'gain':1.0,'density_offset':0})
        dd = int(np.clip(round(fa['energy_norm']*4)+1+arr.get('density_offset',0), 1, 5))
        vi = 0 if fa['brightness']<0.5 else 1
        bp = gp['bass'][dd][vi]
        if bp[s16b]:
            bar_in_prog = bar_num%4
            chord = progression[bar_in_prog%len(progression)]
            freq = chord[0]  # sub freq
            wobble = 1.0+np.sin(cs/sr*np.pi)*fa['motion']*0.02
            freq *= wobble
            t = np.arange(cl)/sr
            spa = sub_phase+np.cumsum(np.ones(cl)*2*np.pi*freq/sr)
            sub = np.sin(spa); sub_phase = spa[-1] if cl>0 else sub_phase
            sg = (0.3+fa['red']*0.5)*(0.5+fa['brightness']*0.5)*arr.get('gain',1.0)
            se = np.exp(-t*4)*0.7+0.3
            sub_bus[cs:ce] += sub*sg*se

    # ─── REESE BASS ──────────────────────────────────────────────────
    print("  Reese bass...")
    for cs in range(0, total_samples, s16):
        ce = min(cs+s16, total_samples); cl = ce-cs
        if cl<=0: continue
        fa = fa_at(cs); s16b = (cs//s16)%16
        bar_num = (cs//s16)//16
        arr = arrangement.get(bar_num, {'gain':1.0,'density_offset':0})
        dd = int(np.clip(round(fa['energy_norm']*4)+1+arr.get('density_offset',0), 1, 5))
        vi = 0 if fa['brightness']<0.5 else 1
        bp = gp['bass'][dd][vi]
        if bp[s16b]:
            bar_in_prog = bar_num%4
            chord = progression[bar_in_prog%len(progression)]
            freq = chord[1]  # mid freq
            dt = 1.001+fa['edge_density']*0.005
            s1 = make_saw(freq*dt, cl, sr=sr, gain=0.3)
            s2 = make_saw(freq/dt, cl, sr=sr, gain=0.3)
            reese = s1+s2
            cutoff = 200+fa['saturation']*2000+fa['brightness']*1500
            reese = apply_lp(reese, cutoff, sr)
            t = np.arange(cl)/sr
            re = np.exp(-t*6)*0.7+0.3
            rg = (0.1+fa['edge_density']*0.3)*arr.get('gain',1.0)
            reese_bus[cs:ce] += reese*rg*re

    # ─── CHORD PAD ───────────────────────────────────────────────────
    print("  Chord pad...")
    for bar_start in range(0, total_samples, bar_samples):
        bar_end = min(bar_start+bar_samples, total_samples); bl = bar_end-bar_start
        if bl<=0: continue
        fa = fa_at(bar_start); bar_num = bar_start//bar_samples
        arr = arrangement.get(bar_num, {'gain':1.0})
        bar_in_prog = bar_num%4
        chord = progression[bar_in_prog%len(progression)]
        pad_freqs = chord[2]  # list of pad freqs
        cutoff = 200+fa['brightness']*800+fa['saturation']*400
        pad = np.zeros(bl)
        for pf in pad_freqs:
            s1 = make_saw(pf*1.002, bl, sr=sr, gain=0.15, num_harmonics=8)
            s2 = make_saw(pf*0.998, bl, sr=sr, gain=0.15, num_harmonics=8)
            pad += s1+s2
        pad /= max(len(pad_freqs),1)
        pad = apply_lp(pad, cutoff, sr)
        an = int(0.3*sr); rn = int(0.5*sr)
        env = np.ones(bl)
        if bl>an: env[:an] = np.linspace(0,1,an)
        if bl>rn: env[-rn:] = np.linspace(1,0,rn)
        pg = (0.04+fa['green']*0.06)*arr.get('gain',1.0)
        pad_bus_L[bar_start:bar_end] += pad*pg*env
        pad_bus_R[bar_start:bar_end] += pad*pg*env*0.8

    # ─── ARP LEAD ────────────────────────────────────────────────────
    print("  Arp lead...")
    arp_counter = 0; pos=0; beat_idx=0
    while pos<total_samples:
        s16b = beat_idx%16; fa = fa_at(pos); e = fa['energy_norm']
        play = False
        if e>0.6: play = s16b%2==0
        elif e>0.35: play = s16b%4==0
        else: play = s16b%8==0
        if play:
            bar_num = beat_idx//16; bar_in_prog = bar_num%4
            chord = progression[bar_in_prog%len(progression)]
            arp_freqs = chord[3]
            note_idx = int(fa['hue_norm']*len(arp_freqs))%len(arp_freqs)
            freq = arp_freqs[note_idx]
            nl = int(s16_sec*1.5*sr); t = np.arange(nl)/sr
            note = np.sin(2*np.pi*freq*t)*env_adsr(nl, sr, a=0.002, d=0.04, s=0.0, r=0.02)
            arr = arrangement.get(bar_num, {'gain':1.0})
            g = e*0.06*arr.get('gain',1.0)
            place(arp_bus, note*g, pos); arp_counter+=1
        beat_idx+=1; pos+=s16

    # ─── MELODY ──────────────────────────────────────────────────────
    print("  Melody...")
    mel_notes = KEYS[key_name]['mel']
    mel_freqs_list = [key_freqs.get(n, 300) for n in mel_notes]
    pos=0; beat_idx=0
    while pos<total_samples:
        s16b = beat_idx%16; fa = fa_at(pos)
        if s16b in (2, 10):
            note_idx = int(fa['hue_norm']*len(mel_freqs_list))%len(mel_freqs_list)
            freq = mel_freqs_list[note_idx]
            nl = int(s16_sec*3*sr); t = np.arange(nl)/sr
            note = 2*np.abs(2*(t*freq-np.floor(t*freq+0.5)))-1
            note *= env_adsr(nl, sr, a=0.01, d=0.15, s=0.2, r=0.2)
            note = apply_lp(note, 2000, sr)
            bar_num = beat_idx//16
            arr = arrangement.get(bar_num, {'gain':1.0})
            g = 0.07*arr.get('gain',1.0)
            place(mel_bus, note*g, pos)
            delay_s = int(s16_sec*6*sr)
            place(mel_bus, note*g*0.35, pos+delay_s)
            place(mel_bus, note*g*0.12, pos+delay_s*2)
        beat_idx+=1; pos+=s16

    # ─── SPECTRAL TEXTURE ────────────────────────────────────────────
    if not no_sonification:
        print("\n[4/6] Rendering spectral texture...")
        tex_L, tex_R = render_spectral_texture(frame_analyses, fps, duration, sr=sr, seed=dna_seed)
    else:
        print("\n[4/6] Sonification skipped")
        tex_L = np.zeros(total_samples); tex_R = np.zeros(total_samples)

    # ─── BUS PROCESSING ──────────────────────────────────────────────
    print("\n[5/6] Processing & mixing...")
    sub_bus = apply_lp(sub_bus[:total_samples], 120, sr)
    sub_bus = np.tanh(sub_bus*1.3)
    reese_bus = apply_hp(reese_bus[:total_samples], 80, sr)
    reese_bus = np.tanh(reese_bus*1.5)
    snare_bus_p = comb_reverb(snare_bus[:total_samples], sr, decay=0.1, delays_ms=(15,))
    tex_L = comb_reverb(tex_L, sr, decay=0.2, delays_ms=(40,55))
    tex_R = comb_reverb(tex_R, sr, decay=0.2, delays_ms=(55,70))
    pad_bus_L_p = comb_reverb(pad_bus_L[:total_samples], sr, decay=0.3, delays_ms=(60,75))
    pad_bus_R_p = comb_reverb(pad_bus_R[:total_samples], sr, decay=0.3, delays_ms=(75,90))
    mel_bus_p = comb_reverb(mel_bus[:total_samples], sr, decay=0.3, delays_ms=(50,70))
    arp_bus_p = comb_reverb(arp_bus[:total_samples], sr, decay=0.15, delays_ms=(25,))

    def trim(buf): return buf[:total_samples] if len(buf)>=total_samples else np.pad(buf,(0,total_samples-len(buf)))

    # ─── STEMS DICT ──────────────────────────────────────────────────
    stems = {
        'kick': trim(kick_bus),
        'snare': trim(snare_bus_p),
        'hats': trim(hat_bus[:total_samples]),
        'clicks': trim(click_bus[:total_samples]),
        'sub': trim(sub_bus),
        'reese': trim(reese_bus),
        'pad_L': trim(pad_bus_L_p), 'pad_R': trim(pad_bus_R_p),
        'arp': trim(arp_bus_p),
        'melody': trim(mel_bus_p),
        'texture_L': trim(tex_L), 'texture_R': trim(tex_R),
        'impact': trim(impact_bus[:total_samples]),
        'riser': trim(riser_bus[:total_samples]),
    }

    # ─── STEREO MIX ──────────────────────────────────────────────────
    print("  Master processing...")
    mix_L = np.zeros(total_samples); mix_R = np.zeros(total_samples)
    # Center
    mix_L += stems['kick']*1.0;      mix_R += stems['kick']*1.0
    mix_L += stems['snare']*0.9;     mix_R += stems['snare']*0.9
    mix_L += stems['sub']*0.85;      mix_R += stems['sub']*0.85
    # Panned
    mix_L += stems['hats']*0.6;      mix_R += stems['hats']*0.45
    mix_L += stems['clicks']*0.4;    mix_R += stems['clicks']*0.35
    mix_L += stems['reese']*0.55;    mix_R += stems['reese']*0.55
    # Stereo
    mix_L += stems['pad_L'];         mix_R += stems['pad_R']
    mix_L += stems['texture_L']*sonification_gain
    mix_R += stems['texture_R']*sonification_gain
    # Arp/melody
    mix_L += stems['arp']*0.4;       mix_R += stems['arp']*0.35
    mix_L += stems['melody']*0.45;   mix_R += stems['melody']*0.5
    # FX
    mix_L += stems['impact']*0.8;    mix_R += stems['impact']*0.8
    mix_L += stems['riser']*0.5;     mix_R += stems['riser']*0.5

    # Master chain
    stereo = np.column_stack([mix_L, mix_R])
    peak = np.max(np.abs(stereo))
    if peak>0: stereo = stereo/peak*0.95
    threshold=0.6; ratio=3.0
    for ch in range(2):
        above = np.abs(stereo[:,ch])>threshold
        stereo[above,ch] = np.sign(stereo[above,ch])*(threshold+(np.abs(stereo[above,ch])-threshold)/ratio)
    stereo = np.clip(stereo, -0.98, 0.98)
    stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.98, neginf=-0.98)

    # ─── WRITE OUTPUT ────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{base}_v5.wav")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n[6/6] Writing output...")
    sf.write(output_path, stereo, sr, subtype='PCM_24')
    fsize = os.path.getsize(output_path)
    print(f"  Main mix: {output_path} ({fsize/1024/1024:.1f} MB)")

    # ─── STEMS EXPORT ────────────────────────────────────────────────
    if export_stems:
        stem_dir = os.path.splitext(output_path)[0] + '_stems'
        os.makedirs(stem_dir, exist_ok=True)
        stem_groups = {
            'drums': ['kick','snare','hats','clicks'],
            'bass': ['sub','reese'],
            'melodic': ['arp','melody'],
            'atmosphere': ['texture_L','texture_R'],
            'pads': ['pad_L','pad_R'],
            'fx': ['impact','riser'],
        }
        for group_name, stem_keys in stem_groups.items():
            if any('_L' in k or '_R' in k for k in stem_keys):
                # Stereo stem
                l_keys = [k for k in stem_keys if '_L' in k or '_R' not in k]
                r_keys = [k for k in stem_keys if '_R' in k or '_L' not in k]
                sl = np.zeros(total_samples); sr_ch = np.zeros(total_samples)
                for k in stem_keys:
                    if k.endswith('_L'): sl += stems[k]
                    elif k.endswith('_R'): sr_ch += stems[k]
                    else: sl += stems[k]; sr_ch += stems[k]
                st = np.column_stack([sl, sr_ch])
            else:
                # Mono summed to stereo
                mono = np.zeros(total_samples)
                for k in stem_keys: mono += stems[k]
                st = np.column_stack([mono, mono])
            # Normalize stem
            pk = np.max(np.abs(st))
            if pk>0: st = st/pk*0.95
            st = np.nan_to_num(st, nan=0.0)
            stem_path = os.path.join(stem_dir, f"{group_name}.wav")
            sf.write(stem_path, st, sr, subtype='PCM_24')
            print(f"  Stem: {stem_path}")
        print(f"  Stems exported to: {stem_dir}/")

    print(f"\n{'='*60}")
    print(f"  Output:   {output_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Size:     {fsize/1024/1024:.1f} MB")
    print(f"  Format:   {sr}Hz / 24-bit / Stereo")
    print(f"  Groove:   {groove.upper()} | Key: {key_name} | Mood: {mood}")
    print(f"{'='*60}")
    return output_path


# ═══ CLI ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="video2dnb v5 — Pattern Library DnB Renderer")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--groove", type=str, default=None, choices=GROOVE_NAMES,
                        help="Override groove type")
    parser.add_argument("--key", type=str, default=None,
                        help="Override key (Cm, Dm, Em, F#m, Abm, Bbm)")
    parser.add_argument("--no-sonification", action="store_true")
    parser.add_argument("--sonification-gain", type=float, default=0.35)
    parser.add_argument("--stems", action="store_true",
                        help="Export individual stems (drums, bass, melodic, atmosphere, pads, fx)")
    parser.add_argument("--sample-rate", type=int, default=SR)
    args = parser.parse_args()
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  VIDEO2DNB v5 — PATTERN LIBRARY + IMAGE SELECTION")
    print(f"  Video: {args.video}")
    print(f"  BPM:   {args.bpm}")
    print(f"  SR:    {args.sample_rate}")
    if args.groove: print(f"  Groove override: {args.groove}")
    if args.key: print(f"  Key override: {args.key}")
    print(f"  Stems: {'YES' if args.stems else 'NO'}")
    print(f"  Sonification: {'OFF' if args.no_sonification else f'ON (gain={args.sonification_gain})'}")
    print(f"{'='*60}\n")

    t0 = time.time()
    render_video_to_dnb(
        args.video, args.bpm, args.output,
        groove_override=args.groove, key_override=args.key,
        no_sonification=args.no_sonification,
        sonification_gain=args.sonification_gain,
        export_stems=args.stems, sr=args.sample_rate,
    )
    print(f"  Render time: {time.time()-t0:.1f}s\n")

if __name__ == "__main__":
    main()
