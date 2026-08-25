from __future__ import annotations

import importlib.metadata
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

from gemini_camera import CAPTURE_DIR
from gemini_orientation import (
    capture_display_rotation,
    get_mount_rotation_degrees,
    rotate_image_for_display,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
MODEL_MINIMUM_CONFIDENCE = 0.50
_CAPTURE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Left/right are image-relative in the upright, unmirrored calibration preview.
# MediaPipe mesh indices are anatomical, so eye/mouth indices are intentionally
# mapped to preserve the app/robot convention. Only visually targetable points
# are selected; generic mesh vertices add manual measurement noise.
CALIBRATION_LANDMARKS: tuple[tuple[str, str, int, bool], ...] = (
    ("left_outer_eye", "Left outer eye", 33, True),
    ("left_iris_center", "Left iris center", 468, False),
    ("left_inner_eye", "Left inner eye", 133, True),
    ("right_inner_eye", "Right inner eye", 362, True),
    ("right_iris_center", "Right iris center", 473, False),
    ("right_outer_eye", "Right outer eye", 263, True),
    ("nose_bridge", "Nose bridge", 168, True),
    ("nose_tip", "Nose tip", 1, True),
    ("left_mouth_corner", "Left mouth corner", 61, True),
    ("upper_lip_center", "Upper lip center", 0, True),
    ("right_mouth_corner", "Right mouth corner", 291, True),
    ("lower_lip_center", "Lower lip center", 17, True),
    ("chin", "Chin", 152, True),
)


class GeminiLandmarkError(RuntimeError):
    pass


class GeminiLandmarkUnavailable(GeminiLandmarkError):
    pass


class GeminiFaceNotFound(GeminiLandmarkError):
    pass


class GeminiFaceLandmarker:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(
            model_path
            or os.getenv("FACE_LANDMARKER_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self._lock = threading.Lock()
        self._landmarker: Any | None = None

    def status(self) -> dict[str, Any]:
        return {
            "ready": cv2 is not None and mp is not None and self.model_path.is_file(),
            "mediapipe_installed": mp is not None,
            "opencv_installed": cv2 is not None,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_file(),
            "mediapipe_version": self._mediapipe_version(),
            "landmark_count_requested": len(CALIBRATION_LANDMARKS),
            "landmark_ids": [item[0] for item in CALIBRATION_LANDMARKS],
            "configured_display_rotation_degrees": get_mount_rotation_degrees(),
        }

    def detect_capture(self, capture_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_ready()
            metadata = self._load_metadata(capture_id)
            rotation_degrees = capture_display_rotation(metadata)
            color = metadata.get("color") or {}
            filename = color.get("filename")
            width = int(color.get("width") or 0)
            height = int(color.get("height") or 0)
            if not filename or width <= 1 or height <= 1:
                raise GeminiLandmarkError("Capture metadata is missing valid RGB image information.")

            image_path = CAPTURE_DIR / Path(str(filename)).name
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise GeminiLandmarkError(f"Could not decode Gemini RGB capture {image_path.name}.")
            if image_bgr.shape[1] != width or image_bgr.shape[0] != height:
                raise GeminiLandmarkError(
                    "Gemini RGB dimensions do not match capture metadata: "
                    f"image={image_bgr.shape[1]}x{image_bgr.shape[0]}, metadata={width}x{height}."
                )

            raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            display_rgb = rotate_image_for_display(raw_rgb, rotation_degrees)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=display_rgb)
            try:
                result = self._landmarker.detect(mp_image)
            except Exception as exc:
                raise GeminiLandmarkError(f"MediaPipe face landmark detection failed: {exc}") from exc

            if not result.face_landmarks:
                raise GeminiFaceNotFound("MediaPipe did not find a face in the upright Gemini RGB frame.")

            face = result.face_landmarks[0]
            selected: list[dict[str, Any]] = []
            omitted: list[dict[str, Any]] = []
            for landmark_id, display_name, source_index, required in CALIBRATION_LANDMARKS:
                if source_index >= len(face):
                    if required:
                        raise GeminiLandmarkError(
                            f"MediaPipe returned {len(face)} landmarks, but required landmark "
                            f"{landmark_id} needs index {source_index}."
                        )
                    omitted.append(
                        {
                            "id": landmark_id,
                            "display_name": display_name,
                            "source_index": source_index,
                            "reason": "index_not_available",
                        }
                    )
                    continue

                point = face[source_index]
                u_px = min(max(float(point.x) * (width - 1), 0.0), width - 1.0)
                v_px = min(max(float(point.y) * (height - 1), 0.0), height - 1.0)
                selected.append(
                    {
                        "id": landmark_id,
                        "display_name": display_name,
                        "u_px": u_px,
                        "v_px": v_px,
                        "confidence": MODEL_MINIMUM_CONFIDENCE,
                        "source_index": source_index,
                    }
                )

            return {
                "status": "ok",
                "capture_id": capture_id,
                "width": width,
                "height": height,
                "display_rotation_degrees": rotation_degrees,
                "pixel_coordinate_system": "upright_display_pixels",
                "landmarks": selected,
                "omitted_landmarks": omitted,
                "detector": {
                    "name": "MediaPipe Face Landmarker",
                    "version": self._mediapipe_version(),
                    "landmark_set": "makeuprobot_gemini_rigid_v3",
                    "left_right_convention": "image_relative_upright_unmirrored",
                    "model_path": str(self.model_path),
                },
            }

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                finally:
                    self._landmarker = None

    def _ensure_ready(self) -> None:
        if cv2 is None:
            raise GeminiLandmarkUnavailable("OpenCV is not installed in the Pi Python environment.")
        if mp is None:
            raise GeminiLandmarkUnavailable(
                "MediaPipe is not installed in the Pi Python environment. Run pip install -r requirements.txt."
            )
        if not self.model_path.is_file():
            raise GeminiLandmarkUnavailable(
                f"Face Landmarker model not found at {self.model_path}. "
                "Run scripts/setup_face_landmarker.sh."
            )
        if self._landmarker is None:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=MODEL_MINIMUM_CONFIDENCE,
                min_face_presence_confidence=MODEL_MINIMUM_CONFIDENCE,
                min_tracking_confidence=MODEL_MINIMUM_CONFIDENCE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            try:
                self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            except Exception as exc:
                raise GeminiLandmarkUnavailable(
                    f"Could not initialize MediaPipe Face Landmarker: {exc}"
                ) from exc

    @staticmethod
    def _load_metadata(capture_id: str) -> dict[str, Any]:
        if not _CAPTURE_ID_RE.fullmatch(capture_id):
            raise GeminiLandmarkError("Invalid capture ID.")
        metadata_path = CAPTURE_DIR / f"{capture_id}_metadata.json"
        if not metadata_path.is_file():
            raise GeminiLandmarkError(f"Capture metadata not found for {capture_id}.")
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GeminiLandmarkError(f"Could not read capture metadata: {exc}") from exc

    @staticmethod
    def _mediapipe_version() -> str | None:
        try:
            return importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError:
            return None


FACE_LANDMARKER = GeminiFaceLandmarker()
