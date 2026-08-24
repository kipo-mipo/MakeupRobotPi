from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gemini_camera import CAPTURE_DIR


class CalibrationDepthError(RuntimeError):
    pass


def _capture_paths(capture_id: str, capture_dir: Path = CAPTURE_DIR) -> tuple[Path, Path]:
    safe_id = Path(capture_id).name
    if safe_id != capture_id or not safe_id:
        raise CalibrationDepthError("Invalid capture ID.")

    depth_path = capture_dir / f"{safe_id}_depth_raw.png"
    metadata_path = capture_dir / f"{safe_id}_metadata.json"
    if not depth_path.is_file() or not metadata_path.is_file():
        raise CalibrationDepthError(f"Capture {safe_id} is missing depth data or metadata.")
    return depth_path, metadata_path


def sample_capture_depth(
    capture_id: str,
    points: list[dict[str, Any]],
    *,
    radius_px: int = 2,
    capture_dir: Path = CAPTURE_DIR,
) -> dict[str, Any]:
    if not points:
        raise CalibrationDepthError("At least one depth sample point is required.")
    if radius_px < 0 or radius_px > 10:
        raise CalibrationDepthError("Depth sample radius must be between 0 and 10 pixels.")

    depth_path, metadata_path = _capture_paths(capture_id, capture_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    try:
        scale_mm = float(metadata["depth"]["scale_mm_per_unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationDepthError("Capture metadata is missing a valid depth scale.") from exc

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise CalibrationDepthError("The aligned raw depth image could not be decoded.")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)

    height, width = depth.shape
    samples: list[dict[str, Any]] = []

    for point in points:
        point_id = str(point.get("id", ""))
        try:
            u = float(point["u_px"])
            v = float(point["v_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationDepthError(f"Depth point {point_id or '<unnamed>'} has invalid U/V coordinates.") from exc

        if not np.isfinite(u) or not np.isfinite(v):
            raise CalibrationDepthError(f"Depth point {point_id or '<unnamed>'} has non-finite U/V coordinates.")

        center_u = int(round(u))
        center_v = int(round(v))
        if center_u < 0 or center_u >= width or center_v < 0 or center_v >= height:
            raise CalibrationDepthError(
                f"Depth point {point_id or '<unnamed>'} is outside the aligned image ({width}x{height})."
            )

        u0 = max(0, center_u - radius_px)
        u1 = min(width, center_u + radius_px + 1)
        v0 = max(0, center_v - radius_px)
        v1 = min(height, center_v + radius_px + 1)
        window = depth[v0:v1, u0:u1]
        valid = window[window > 0]

        if valid.size:
            raw_value = float(np.median(valid))
            depth_mm = raw_value * scale_mm
        else:
            raw_value = None
            depth_mm = None

        samples.append(
            {
                "id": point_id,
                "u_px": u,
                "v_px": v,
                "depth_raw": raw_value,
                "depth_mm": depth_mm,
                "valid_sample_count": int(valid.size),
                "radius_px": radius_px,
            }
        )

    return {
        "status": "ok",
        "capture_id": capture_id,
        "width": width,
        "height": height,
        "depth_scale_mm": scale_mm,
        "samples": samples,
    }
