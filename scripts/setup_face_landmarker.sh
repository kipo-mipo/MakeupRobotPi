#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
MODEL_PATH="$MODEL_DIR/face_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_PATH" ]]; then
    echo "Face Landmarker model already exists: $MODEL_PATH"
    exit 0
fi

if command -v curl >/dev/null 2>&1; then
    curl -fL "$MODEL_URL" -o "$MODEL_PATH"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "Need curl or wget to download the MediaPipe Face Landmarker model." >&2
    exit 1
fi

echo "Installed Face Landmarker model: $MODEL_PATH"
