from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RAW_IMAGE_WIDTH = 1280
RAW_IMAGE_HEIGHT = 720
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


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
    """Map an upright detector point back into the raw sideways Pi image.

    The FaceCapture iOS calibration preview rotates the raw Pi image 90 degrees
    counterclockwise. For a raw point (U, V):

        upright_x = V
        upright_y = raw_width - 1 - U

    MediaPipe returns normalized coordinates in that upright image. This
    function applies the inverse transform so the API returns the same raw U/V
    coordinate system used by the iOS calibration homography.
    """

    if raw_width <= 1 or raw_height <= 1:
        raise ValueError("raw image dimensions must both be greater than one")

    upright_width = raw_height
    upright_height = raw_width

    upright_x = float(x_normalized) * (upright_width - 1)
    upright_y = float(y_normalized) * (upright_height - 1)

    raw_u = (raw_width - 1) - upright_y
    raw_v = upright_x

    # MediaPipe can occasionally report points a tiny amount outside [0, 1].
    raw_u = min(max(raw_u, 0.0), raw_width - 1.0)
    raw_v = min(max(raw_v, 0.0), raw_height - 1.0)

    return raw_u, raw_v


class FaceLandmarkCapture:
    """Lazy, serialized access to the Pi camera and MediaPipe Face Landmarker."""

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
        self._camera: Any | None = None
        self._landmarker: Any | None = None

    def status(self) -> dict[str, Any]:
        return {
            "raw_image_width_px": self.raw_width,
            "raw_image_height_px": self.raw_height,
            "raw_orientation": "sideways",
            "detector_rotation": "90_degrees_counterclockwise",
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_file(),
            "camera_initialized": self._camera is not None,
            "landmarker_initialized": self._landmarker is not None,
        }

    def capture(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_ready()

            captured_at = datetime.now(timezone.utc).isoformat()
            raw_rgb = self._camera.capture_array("main")

            expected_shape = (self.raw_height, self.raw_width)
            if raw_rgb.ndim != 3 or raw_rgb.shape[:2] != expected_shape:
                raise LandmarkCaptureError(
                    "Unexpected camera frame shape "
                    f"{raw_rgb.shape}; expected ({self.raw_height}, "
                    f"{self.raw_width}, 3)."
                )

            # The physical camera is sideways. Rotate only for detection; raw
            # U/V coordinates remain authoritative for camera-to-robot mapping.
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
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                raise FaceNotFound("No face detected in the camera frame.")

            face = result.face_landmarks[0]
            landmarks = []

            for index, landmark in enumerate(face):
                raw_u, raw_v = upright_normalized_to_raw_pixel(
                    landmark.x,
                    landmark.y,
                    raw_width=self.raw_width,
                    raw_height=self.raw_height,
                )

                landmarks.append(
                    {
                        "index": index,
                        "u_px": raw_u,
                        "v_px": raw_v,
                        # MediaPipe z is relative depth, not calibrated mm.
                        "normalized_z": float(landmark.z),
                    }
                )

            return {
                "status": "ok",
                "captured_at": captured_at,
                "face_count": len(result.face_landmarks),
                "landmark_count": len(landmarks),
                "raw_image": {
                    "width_px": self.raw_width,
                    "height_px": self.raw_height,
                    "orientation": "sideways",
                },
                "detector_image": {
                    "width_px": self.raw_height,
                    "height_px": self.raw_width,
                    "rotation_from_raw": "90_degrees_counterclockwise",
                },
                "coordinate_contract": {
                    "u_px": "raw Pi-image horizontal pixel coordinate",
                    "v_px": "raw Pi-image vertical pixel coordinate",
                    "robot_mapping": "apply the iOS camera-to-robot homography",
                },
                "landmarks": landmarks,
            }

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                finally:
                    self._landmarker = None

            if self._camera is not None:
                try:
                    self._camera.stop()
                finally:
                    self._camera.close()
                    self._camera = None

    def _ensure_ready(self) -> None:
        if self._landmarker is None:
            self._initialize_landmarker()

        if self._camera is None:
            self._initialize_camera()

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
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
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

    def _initialize_camera(self) -> None:
        try:
            from libcamera import controls
            from picamera2 import Picamera2
        except ImportError as exc:
            raise LandmarkCaptureUnavailable(
                "Picamera2/libcamera is not available. Install "
                "python3-picamera2 from Raspberry Pi OS."
            ) from exc

        camera = None
        try:
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={
                    "size": (self.raw_width, self.raw_height),
                    "format": "RGB888",
                }
            )
            camera.configure(config)
            camera.start()
            camera.set_controls(
                {
                    "AfMode": controls.AfModeEnum.Continuous,
                }
            )

            # Give AE/AWB/autofocus a short startup window before first capture.
            time.sleep(0.5)
        except Exception as exc:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            raise LandmarkCaptureUnavailable(
                f"Could not initialize Pi camera: {exc}"
            ) from exc

        self._camera = camera
