#!/bin/bash

# Batch render all glitch videos to DnB
# Usage: ./batch_render.sh

GLITCH_DIR="/Users/martintomek/.gemini/antigravity/scratch/martin-tomek-portfolio/public/projects/glitch"
OUTPUT_DIR="/Users/martintomek/Documents/osobni_projekty/sticker"
RENDER_SCRIPT="/Users/martintomek/CascadeProjects/strudel/my-tracks/video2dnb/video2dnb_render.py"

# Activate virtual environment
source "/Users/martintomek/CascadeProjects/strudel/my-tracks/video2dnb/venv/bin/activate"

# Videos to skip (already rendered)
SKIP=(
    "240110_v1_00100159.mp4"
    "240225_Feedback_12.mp4"
    "240225_Feedback_13.mp4"
    "240225_Feedback_1.mp4"
)

# Function to check if video should be skipped
should_skip() {
    local video="$1"
    for skip in "${SKIP[@]}"; do
        if [[ "$video" == "$skip" ]]; then
            return 0
        fi
    done
    return 1
}

# Process each video
for video in "$GLITCH_DIR"/*.mp4; do
    filename=$(basename "$video")
    
    if should_skip "$filename"; then
        echo "⏭️  Skipping $filename (already done)"
        continue
    fi
    
    echo "🎬 Processing $filename..."
    
    # Extract base name
    basename="${filename%.mp4}"
    wav_path="$OUTPUT_DIR/${basename}_dnb.wav"
    mp4_path="$OUTPUT_DIR/${basename}_with_dnb.mp4"
    
    # Render to WAV
    echo "  📊 Rendering to WAV..."
    python3 "$RENDER_SCRIPT" "$video" --bpm 174 --output "$wav_path"
    
    if [ $? -eq 0 ]; then
        # Merge with video
        echo "  🎵 Merging with video..."
        ffmpeg -i "$video" -i "$wav_path" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "$mp4_path" -y
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Done: $mp4_path"
        else
            echo "  ❌ Failed to merge $filename"
        fi
    else
        echo "  ❌ Failed to render $filename"
    fi
    
    echo ""
done

echo "🎉 Batch render complete!"
