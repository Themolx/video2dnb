// ============================================================
// JUNGLE TRACK — Chopped Amen Breaks
// Paste into Strudel (strudel.cc) and press Ctrl+Enter
// BPM: 170 | Classic jungle vibes with amen chops
// ============================================================

// Load amen chops from GitHub
samples({
  amen: {
    '0':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/00_kick_ride.wav',
    '1':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/01_ride_ghost.wav',
    '2':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/02_snare_ride.wav',
    '3':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/03_hat_tick.wav',
    '4':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/04_kick_hat.wav',
    '5':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/05_ride_soft.wav',
    '6':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/06_snare_ghost.wav',
    '7':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/07_hat_ride.wav',
    '8':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/08_kick_ride2.wav',
    '9':  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/09_hat_soft.wav',
    '10': 'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/10_snare_ride2.wav',
    '11': 'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/11_ride_ring.wav',
    '12': 'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/12_kick_short.wav',
    '13': 'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/13_ride_tick.wav',
    '14': 'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/14_snare_hat.wav',
    '15': 'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/15_ride_end.wav',
  },
  amenhit: {
    kick:   'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/hit_kick.wav',
    snare:  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/hit_snare.wav',
    ghost:  'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/hit_snare_ghost.wav',
    hat:    'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/hit_hat.wav',
    hatopen:'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/hit_hat_open.wav',
    ride:   'https://raw.githubusercontent.com/Themolx/video2dnb/main/samples/amen/hit_ride.wav',
  }
})

setcpm(170/4)

// ===================== AMEN BREAK — main chopped loop =====================
// Classic jungle: play the amen chops in sequence, then mangle them
// Original pattern: 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
// S1: straight amen | S2: chopped/rearranged | S3: halftime | S4: double-time chaos
$: cat(
  s("amen:0 amen:1 amen:2 amen:3 amen:4 amen:5 amen:6 amen:7 amen:8 amen:9 amen:10 amen:11 amen:12 amen:13 amen:14 amen:15"),
  s("amen:0 amen:2 amen:0 amen:2 amen:4 amen:10 amen:8 amen:14 amen:0 amen:6 amen:2 amen:14 amen:8 amen:2 amen:10 amen:14"),
  s("amen:0 ~ amen:2 ~ amen:4 ~ amen:10 ~ amen:8 ~ amen:14 ~ amen:0 ~ amen:2 ~"),
  s("amen:0 amen:2 amen:10 amen:2 amen:0 amen:14 amen:2 amen:0 amen:8 amen:2 amen:14 amen:10 [amen:2 amen:14] [amen:0 amen:10] [amen:2 amen:2] [amen:14 amen:0]")
).slow(64)
  .gain(1.1)
  .speed(cat(1, 1, 0.85, 1.15).slow(64))

// ===================== EXTRA KICK — reinforcement =====================
$: cat(
  s("amenhit:kick ~ ~ ~ ~ ~ ~ ~ ~ ~ amenhit:kick ~ ~ ~ ~ ~"),
  s("amenhit:kick ~ ~ amenhit:kick ~ ~ ~ ~ ~ ~ amenhit:kick ~ ~ ~ ~ ~"),
  s("amenhit:kick ~ ~ ~ ~ ~ ~ ~ amenhit:kick ~ ~ ~ ~ ~ ~ ~"),
  s("amenhit:kick ~ ~ amenhit:kick ~ ~ amenhit:kick ~ ~ ~ amenhit:kick ~ ~ ~ ~ ~")
).slow(64)
  .gain(0.85)

// ===================== EXTRA SNARE — accent 2 & 4 =====================
$: cat(
  s("~ ~ ~ ~ amenhit:snare ~ ~ ~ ~ ~ ~ ~ amenhit:snare ~ ~ ~"),
  s("~ ~ ~ ~ amenhit:snare ~ ~ ~ ~ ~ ~ ~ amenhit:snare ~ ~ amenhit:ghost"),
  s("~ ~ ~ ~ amenhit:snare ~ ~ ~ ~ ~ ~ ~ amenhit:snare ~ ~ ~"),
  s("~ ~ ~ ~ amenhit:snare ~ amenhit:ghost ~ ~ ~ ~ amenhit:ghost amenhit:snare ~ amenhit:snare ~")
).slow(64)
  .gain(0.7)

// ===================== SUB BASS — deep jungle weight =====================
// Am pentatonic: A1=55 C2=65 D2=73 E2=82 G2=98
$: cat(
  note("a1 ~ ~ ~ ~ ~ ~ ~ a1 ~ ~ ~ ~ ~ ~ ~"),
  note("a1 ~ ~ ~ ~ ~ c2 ~ a1 ~ ~ ~ ~ ~ ~ ~"),
  note("a1 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
  note("a1 ~ ~ a1 ~ ~ c2 ~ d2 ~ ~ a1 ~ ~ ~ ~")
).slow(64)
  .s("sawtooth")
  .cutoff(200)
  .resonance(8)
  .decay(.15)
  .sustain(0)
  .gain(0.9)

// ===================== REESE BASS — dark jungle mid-bass =====================
$: cat(
  note("a2 ~ ~ ~ ~ ~ ~ ~ a2 ~ ~ ~ ~ ~ ~ ~"),
  note("a2 ~ ~ ~ ~ ~ c3 ~ a2 ~ ~ ~ ~ ~ ~ ~"),
  note("a2 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
  note("a2 ~ ~ a2 ~ ~ c3 ~ d3 ~ ~ a2 ~ ~ ~ ~")
).slow(64)
  .s("sawtooth")
  .cutoff(sine.range(300, 1200).slow(8))
  .resonance(12)
  .decay(.2)
  .sustain(.3)
  .release(.1)
  .gain(0.35)

// ===================== PAD — atmospheric jungle chords =====================
// Am → C → Dm → Am
$: cat(
  note("[a3,c4,e4]"),
  note("[c3,e3,g3]"),
  note("[d3,f3,a3]"),
  note("[a3,c4,e4]")
).slow(64)
  .s("sawtooth")
  .cutoff(sine.range(400, 1500).slow(16))
  .resonance(5)
  .attack(0.5)
  .decay(2)
  .sustain(0.6)
  .release(1)
  .gain(0.12)
  .room(0.6)
  .roomsize(4)

// ===================== STAB — jungle chord stabs =====================
$: cat(
  s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
  note("~ ~ [a4,c5,e5] ~ ~ ~ ~ ~ ~ ~ [a4,c5,e5] ~ ~ ~ ~ ~"),
  s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
  note("[a4,c5,e5] ~ ~ ~ ~ ~ [c5,e5,g5] ~ ~ ~ ~ ~ [d5,f5,a5] ~ ~ ~")
).slow(64)
  .s("square")
  .cutoff(2500)
  .decay(.08)
  .sustain(0)
  .gain(0.2)
  .room(0.4)

// ===================== RIDE LAYER — extra shimmer =====================
$: s("amenhit:ride*4")
  .gain(sine.range(0.05, 0.18).slow(8))
  .speed(rand.range(0.95, 1.05))
  .pan(sine.range(0.3, 0.7).slow(3))

// ===================== VOCAL CHOP SIMULATION — pitched amen hits =====================
$: cat(
  s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
  s("~ ~ ~ ~ ~ ~ amen:6 ~ ~ ~ ~ ~ ~ ~ amen:6 ~"),
  s("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
  s("~ amen:6 ~ ~ amen:6 ~ ~ amen:6 ~ ~ ~ ~ amen:6 ~ ~ ~")
).slow(64)
  .speed(cat(1.5, 2, 1.75, 2.5).slow(64))
  .gain(0.25)
  .room(0.5)
  .pan(rand.range(0.2, 0.8))
