from __future__ import annotations

import base64
import importlib.metadata
import io
import math
import os
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


RAW_IMAGE_WIDTH = 1280
RAW_IMAGE_HEIGHT = 720
DETECTOR_ROTATION_DEGREES_CCW = 90
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")
MODEL_MINIMUM_CONFIDENCE = 0.50

# MediaPipe landmark indices used by the app's mannequin-calibration mock.
# Left/right are anatomical: the mannequin/person's left and right.
CALIBRATION_LANDMARKS: tuple[tuple[str, int], ...] = (
    ("nose_tip", 1),
    ("left_inner_eye", 362),
    ("right_inner_eye", 133),
    ("left_mouth_corner", 291),
    ("right_mouth_corner", 61),
    ("chin", 152),
)


class LandmarkCaptureError(RuntimeError):
    """Base error for Pi face-landmark capture failures."""


class LandmarkCaptureUnavailable(LandmarkCaptureError):
    """Raised when camera/model dependencies are not ready."""


class FaceNotFound(LandmarkCaptureError):
    """Raised when no usable face is found in the captured frame."""


def upright_normalized_to_raw_pixel(
    x_normalized: float,
    y_normalized: float,
    *,
    raw_width: int = RAW_IMAGE_WIDTH,
    raw_height: int = RAW_IMAGE_HEIGHT,
) -> tuple[float, float]:
    """Map an upright detector point back into the raw sideways Pi image."""

    if raw_width <= 1 or raw_height <= 1:
        raise ValueError("raw image dimensions must both be greater than one")

    upright_width = raw_height
    upright_height = raw_width

    upright_x = float(x_normalized) * (upright_width - 1)
    upright_y = float(y_normalized) * (upright_height - 1)

    raw_u = (raw_width - 1) - upright_y
    raw_v = upright_x

    raw_u = min(max(raw_u, 0.0), raw_width - 1.0)
    raw_v = min(max(raw_v, 0.0), raw_height - 1.0)

    return raw_u, raw_v


def encode_raw_rgb_as_jpeg_base64(
    raw_rgb: np.ndarray,
    *,
    quality: int = 90,
) -> str:
    """Encode the unrotated raw Pi frame as base64 JPEG."""

    if raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
        raise ValueError("raw RGB image must have shape (height, width, 3)")

    output = io.BytesIO()
    Image.fromarray(raw_rgb.astype(np.uint8, copy=False), mode="RGB").save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True,
    )
    return base64.b64encode(output.getvalue()).decode("ascii")


def _finite_probability(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return min(max(number, 0.0), 1.0)


def landmark_confidence(
    landmark: Any,
    *,
    fallback: float = MODEL_MINIMUM_CONFIDENCE,
) -> tuple[float, str]:
    """Return a conservative confidence and describe where it came from."""

    candidates: list[tuple[str, float]] = []

    presence = _finite_probability(getattr(landmark, "presence", None))
    if presence is not None:
        candidates.append(("presence", presence))

    visibility = _finite_probability(getattr(landmark, "visibility", None))
    if visibility is not None:
        candidates.append(("visibility", visibility))

    if candidates:
        source, score = min(candidates, key=lambda item: item[1])
        if len(candidates) == 2:
            source = "min_presence_visibility"
        return score, source

    return min(max(float(fallback), 0.0), 1.0), "model_acceptance_floor"


class FaceLandmarkCapture:
    """Serialized rpicam capture plus MediaPipe Face Landmarker detection."""

    def __init__(
        self,
        *,
        raw_width: int = RAW_IMAGE_WIDTH,
        raw_height: int = RAW_IMAGE_HEIGHT,
        model_path: str | Path | None = None,
    ) -> None:
        self.raw_width = raw_width
        self.raw_height = raw_height
        self.model_path = Path(
            model_path
            or os.getenv("FACE_LANDMARKER_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self._lock = threading.Lock()
        self._landmarker: Any | None = None

    def status(self) -> dict[str, Any]:
        camera_command = shutil.which("rpicam-jpeg")
        return {
            "raw_image_width_px": self.raw_width,
            "raw_image_height_px": self.raw_height,
            "raw_orientation": "sideways",
            "rotation_degrees_ccw_for_detection": DETECTOR_ROTATION_DEGREES_CCW,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_file(),
            "camera_backend": "rpicam-jpeg",
            "camera_command_found": camera_command is not None,
            "landmarker_initialized": self._landmarker is not None,
            "calibration_landmark_ids": [
                landmark_id for landmark_id, _ in CALIBRATION_LANDMARKS
            ],
        }

    def capture_for_calibration(
        self,
        *,
        request_id: str,
        return_image: bool,
        minimum_confidence: float,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_ready()

            captured_at = datetime.now(timezone.utc).isoformat()
            raw_rgb = self._capture_raw_rgb()
            self._validate_frame(raw_rgb)

            # Physical camera is sideways. Rotate only detector input.
            upright_rgb = np.ascontiguousarray(np.rot90(raw_rgb, k=1))

            try:
                import mediapipe as mp
            except ImportError as exc:
                raise LandmarkCaptureUnavailable(
                    "MediaPipe is not installed in the Pi Python environment."
                ) from exc

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=upright_rgb,
            )

            try:
                result = self._landmarker.detect(mp_image)
            except Exception as exc:
                raise LandmarkCaptureError(
                    f"Face landmark detection failed: {exc}"
                ) from exc

            if not result.face_landmarks:
                raise FaceNotFound("No face detected in the camera frame.")

            face = result.face_landmarks[0]
            selected_landmarks: list[dict[str, Any]] = []

            for landmark_id, source_index in CALIBRATION_LANDMARKS:
                if source_index >= len(face):
                    raise LandmarkCaptureError(
                        f"Detector returned {len(face)} landmarks, but "
                        f"{landmark_id} requires MediaPipe index {source_index}."
                    )

                landmark = face[source_index]
                confidence, confidence_source = landmark_confidence(landmark)

                if confidence < minimum_confidence:
                    continue

                raw_u, raw_v = upright_normalized_to_raw_pixel(
                    landmark.x,
                    landmark.y,
                    raw_width=self.raw_width,
                    raw_height=self.raw_height,
                )

                selected_landmarks.append(
                    {
                        "id": landmark_id,
                        "u_px": raw_u,
                        "v_px": raw_v,
                        "confidence": confidence,
                        "source_index": source_index,
                        "confidence_source": confidence_source,
                    }
                )

            if not selected_landmarks:
                raise FaceNotFound(
                    "A face was detected, but no calibration landmarks met "
                    f"minimum_confidence={minimum_confidence:.2f}."
                )

            return {
                "status": "ok",
                "request_id": request_id,
                "captured_at": captured_at,
                "camera": {
                    "model": "Pi Camera Module 3 Standard",
                    "raw_width_px": self.raw_width,
                    "raw_height_px": self.raw_height,
                    "rotation_degrees_ccw": DETECTOR_ROTATION_DEGREES_CCW,
                },
                "image_jpeg_base64": (
                    encode_raw_rgb_as_jpeg_base64(raw_rgb)
                    if return_image
                    else None
                ),
                "landmarks": selected_landmarks,
                "detector": {
                    "name": "MediaPipe Face Landmarker",
                    "version": self._mediapipe_version(),
                    "landmark_set": "makeuprobot_mannequin_v1",
                    "left_right_convention": "anatomical_subject",
                    "confidence_mode": (
                        "landmark presence/visibility when supplied; otherwise "
                        "the configured model acceptance floor"
                    ),
                },
            }

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                finally:
                    self._landmarker = None

    def _capture_raw_rgb(self) -> np.ndarray:
        command = shutil.which("rpicam-jpeg")
        if command is None:
            raise LandmarkCaptureUnavailable(
                "rpicam-jpeg is not installed or not on PATH. Run setup_pi.sh."
            )

        with tempfile.TemporaryDirectory(prefix="makeuprobot-camera-") as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"
            args = [
                command,
                "--output",
                str(image_path),
                "--timeout",
                "800",
                "--width",
                str(self.raw_width),
                "--height",
                str(self.raw_height),
                "--autofocus-mode",
                "default",
                "--nopreview",
            ]

            try:
                completed = subprocess.run(
                    args,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=12,
                )
            except subprocess.TimeoutExpired as exc:
                raise LandmarkCaptureUnavailable(
                    "Pi camera capture timed out."
                ) from exc

            if completed.returncode != 0 or not image_path.is_file():
                detail = (completed.stderr or completed.stdout or "").strip()
                raise LandmarkCaptureUnavailable(
                    "rpicam-jpeg could not capture a frame"
                    + (f": {detail}" if detail else ".")
                )

            try:
                with Image.open(image_path) as image:
                    return np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                raise LandmarkCaptureError(
                    f"Captured JPEG could not be decoded: {exc}"
                ) from exc

    def _validate_frame(self, raw_rgb: np.ndarray) -> None:
        expected_shape = (self.raw_height, self.raw_width, 3)
        if raw_rgb.shape != expected_shape:
            raise LandmarkCaptureError(
                f"Unexpected camera frame shape {raw_rgb.shape}; "
                f"expected {expected_shape}."
            )

    def _ensure_ready(self) -> None:
        if shutil.which("rpicam-jpeg") is None:
            raise LandmarkCaptureUnavailable(
                "rpicam-jpeg is unavailable. Run setup_pi.sh."
            )

        if self._landmarker is None:
            self._initialize_landmarker()

    def _initialize_landmarker(self) -> None:
        if not self.model_path.is_file():
            raise LandmarkCaptureUnavailable(
                f"Face Landmarker model not found at {self.model_path}. "
                "Run setup_pi.sh or set FACE_LANDMARKER_MODEL_PATH."
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise LandmarkCaptureUnavailable(
                "MediaPipe is not installed in the Pi Python environment."
            ) from exc

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=MODEL_MINIMUM_CONFIDENCE,
            min_face_presence_confidence=MODEL_MINIMUM_CONFIDENCE,
            min_tracking_confidence=MODEL_MINIMUM_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        try:
            self._landmarker = (
                mp.tasks.vision.FaceLandmarker.create_from_options(options)
            )
        except Exception as exc:
            raise LandmarkCaptureUnavailable(
                f"Could not initialize Face Landmarker: {exc}"
            ) from exc

    @staticmethod
    def _mediapipe_version() -> str | None:
        try:
            return importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError:
            return None
