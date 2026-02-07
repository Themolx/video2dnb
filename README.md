# video2dnb

**Convert video into Drum & Bass audio — feel the image in the music.**

A Python tool that reads every frame of a video and synthesizes a DnB track where visual content directly drives the audio. This isn't just "mapping parameters to templates" — the pixel data itself becomes sound through additive synthesis, spectral reconstruction, and per-frame modulation.

## How It Works

Every frame of your video drives the synthesis in real-time:

| Visual Property | Audio Effect |
|---|---|
| **Image rows** (brightness per row) | Additive spectral synthesis — you literally hear the image |
| **Red channel** | Sub bass intensity & pitch wobble |
| **Green channel** | Pad/atmosphere gain |
| **Blue channel** | Hi-hat brightness & density |
| **Edge density** | Drum hit intensity, snare fills, hat pattern density |
| **Motion** (frame diff) | Ghost snares, energy spikes |
| **Saturation** | Reese bass filter cutoff |
| **Dominant hue** | Chord root (warm=minor, cool=major) |
| **Brightness** | Overall amplitude, filter openness |
| **Contrast** | Dynamic range of percussion |

## DnB Structure (174 BPM)

- **Kick** — beat 1, with motion/edge-driven offbeats
- **Snare** — beat 2, extra hits from edge density
- **Ghost snares** — triggered by inter-frame motion
- **Hi-hats** — 8th or 16th note density from edge detail
- **Sub bass** (sine) — follows chord roots from dominant hue
- **Reese bass** (detuned saws) — saturation-driven filter
- **Spectral texture** — img2sound-style layer: image rows → frequency bands
- **Chord pad** (detuned saws) — brightness-driven filter

## Installation

```bash
# Clone
git clone https://github.com/Themolx/video2dnb.git
cd video2dnb

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Video
```bash
python video2dnb_render.py input_video.mp4 --bpm 174 --output output.wav
```

### Batch Render
```bash
# Edit paths in batch_render.sh, then:
chmod +x batch_render.sh
./batch_render.sh
```

### Merge Audio with Video
```bash
ffmpeg -i input.mp4 -i output.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest output_with_audio.mp4
```

## Options

| Argument | Default | Description |
|---|---|---|
| `video` | (required) | Input video file path |
| `--bpm` | 174 | Beats per minute |
| `--output` / `-o` | `<video>_dnb.wav` | Output WAV path |

## Output Format

- **Sample rate**: 48,000 Hz
- **Bit depth**: 24-bit PCM
- **Channels**: Stereo
- **Duration**: Matches input video exactly

## Examples

Render a glitch art video:
```bash
python video2dnb_render.py glitch_feedback.mp4 --bpm 174 -o glitch_dnb.wav
```

Render at slower tempo:
```bash
python video2dnb_render.py ambient_video.mp4 --bpm 140 -o ambient_dnb.wav
```

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- SciPy
- SoundFile

## License

MIT
