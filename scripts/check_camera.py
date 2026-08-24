import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_camera import camera_status


if __name__ == "__main__":
    status = camera_status()
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if status["ready"] else 1)
