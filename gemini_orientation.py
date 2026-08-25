from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from gemini_camera import CAPTURE_DIR

CONFIG_DIR = Path(__file__).resolve().parent / "config"
MOUNT_CONFIG_PATH = CONFIG_DIR / "gemini_mount_orientation.json"
DEFAULT_ROTATION_DEGREES = 180
SUPPORTED_ROTATIONS = (0, 180)


class GeminiOrientationError(RuntimeError):
    pass


def _validate_rotation(value: Any) -> int:
    try:
        rotation = int(value)
    except (TypeError, ValueError) as exc:
        raise GeminiOrientationError("Gemini mount rotation must be 0 or 180 degrees.") from exc
    if rotation not in SUPPORTED_ROTATIONS:
        raise GeminiOrientationError("Gemini mount rotation must be 0 or 180 degrees.")
    return rotation


def get_mount_rotation_degrees() -> int:
    if not MOUNT_CONFIG_PATH.is_file():
        return DEFAULT_ROTATION_DEGREES
    try:
        payload = json.loads(MOUNT_CONFIG_PATH.read_text(encoding="utf-8"))
        return _validate_rotation(payload.get("rotation_degrees"))
    except GeminiOrientationError:
        raise
    except Exception as exc:
        raise GeminiOrientationError(f"Could not read Gemini mount orientation: {exc}") from exc


def set_mount_rotation_degrees(rotation_degrees: int) -> dict[str, Any]:
    rotation = _validate_rotation(rotation_degrees)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "rotation_degrees": rotation,
        "meaning": "Rotate raw Gemini RGB this many degrees for upright display and MediaPipe detection.",
    }
    temporary = MOUNT_CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(MOUNT_CONFIG_PATH)
    return mount_orientation_status()


def mount_orientation_status() -> dict[str, Any]:
    rotation = get_mount_rotation_degrees()
    return {
        "status": "ok",
        "rotation_degrees": rotation,
        "supported_rotations": list(SUPPORTED_ROTATIONS),
        "default_rotation_degrees": DEFAULT_ROTATION_DEGREES,
        "config_path": MOUNT_CONFIG_PATH.name,
        "raw_geometry_unchanged": True,
        "applies_to_new_captures": True,
    }


def capture_display_rotation(metadata: dict[str, Any]) -> int:
    display = metadata.get("display")
    if not isinstance(display, dict):
        return 0
    return _validate_rotation(display.get("rotation_degrees", 0))


def rotate_image_for_display(image: Any, rotation_degrees: int) -> Any:
    rotation = _validate_rotation(rotation_degrees)
    if rotation == 0:
        return image
    return cv2.rotate(image, cv2.ROTATE_180)


def display_to_raw_pixel(
    u_px: float,
    v_px: float,
    *,
    width: int,
    height: int,
    rotation_degrees: int,
) -> tuple[float, float]:
    rotation = _validate_rotation(rotation_degrees)
    if rotation == 0:
        return float(u_px), float(v_px)
    return (float(width - 1) - float(u_px), float(height - 1) - float(v_px))


def raw_to_display_pixel(
    u_px: float,
    v_px: float,
    *,
    width: int,
    height: int,
    rotation_degrees: int,
) -> tuple[float, float]:
    return display_to_raw_pixel(
        u_px,
        v_px,
        width=width,
        height=height,
        rotation_degrees=rotation_degrees,
    )


def prepare_capture_display(
    *,
    capture_id: str,
    color_filename: str,
    metadata_filename: str,
    capture_dir: Path = CAPTURE_DIR,
) -> dict[str, Any]:
    rotation = get_mount_rotation_degrees()
    color_path = capture_dir / Path(color_filename).name
    metadata_path = capture_dir / Path(metadata_filename).name
    if not color_path.is_file() or not metadata_path.is_file():
        raise GeminiOrientationError("Gemini capture is missing its RGB image or metadata.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GeminiOrientationError(f"Could not read Gemini capture metadata: {exc}") from exc

    if rotation == 0:
        display_filename = color_path.name
    else:
        image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise GeminiOrientationError("Could not decode the raw Gemini RGB image for display rotation.")
        display_image = rotate_image_for_display(image, rotation)
        display_filename = f"{capture_id}_color_display.png"
        display_path = capture_dir / display_filename
        if not cv2.imwrite(str(display_path), display_image):
            raise GeminiOrientationError("Could not save the upright Gemini display image.")

    metadata["display"] = {
        "rotation_degrees": rotation,
        "color_filename": display_filename,
        "coordinate_system": "upright_display_pixels",
        "raw_color_filename": color_path.name,
        "raw_depth_and_intrinsics_unchanged": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "rotation_degrees": rotation,
        "display_color_filename": display_filename,
    }
