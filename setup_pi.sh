#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y rpicam-apps wget curl

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12

rm -rf .venv
uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python -r requirements.txt

mkdir -p models
wget -q --show-progress \
  -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

echo "Setup complete."
echo "Python: $(.venv/bin/python --version)"
echo "Run: source .venv/bin/activate && python main.py"
