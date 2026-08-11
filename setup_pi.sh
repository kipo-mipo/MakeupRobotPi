#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-venv wget

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p models
wget -q --show-progress \
  -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

echo "Setup complete."
echo "Run: source .venv/bin/activate && python main.py"
