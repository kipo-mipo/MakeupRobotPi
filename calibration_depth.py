from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camera_geometry import CameraGeometryError, read_active_camera_geometry
from gemini_camera import CAPTURE_DIR
from gemini_orientation import capture_display_rotation, display_to_raw_pixel


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


def _validated_point(
    point: dict[str, Any],
    width: int,
    height: int,
    rotation_degrees: int,
) -> dict[str, Any]:
    point_id = str(point.get("id", ""))
    try:
        display_u = float(point["u_px"])
        display_v = float(point["v_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationDepthError(
            f"Depth point {point_id or '<unnamed>'} has invalid U/V coordinates."
        ) from exc

    if not np.isfinite(display_u) or not np.isfinite(display_v):
        raise CalibrationDepthError(
            f"Depth point {point_id or '<unnamed>'} has non-finite U/V coordinates."
        )

    display_center_u = int(round(display_u))
    display_center_v = int(round(display_v))
    if (
        display_center_u < 0
        or display_center_u >= width
        or display_center_v < 0
        or display_center_v >= height
    ):
        raise CalibrationDepthError(
            f"Depth point {point_id or '<unnamed>'} is outside the upright display image ({width}x{height})."
        )

    raw_u, raw_v = display_to_raw_pixel(
        display_u,
        display_v,
        width=width,
        height=height,
        rotation_degrees=rotation_degrees,
    )
    center_u = int(round(raw_u))
    center_v = int(round(raw_v))
    if center_u < 0 or center_u >= width or center_v < 0 or center_v >= height:
        raise CalibrationDepthError(
            f"Depth point {point_id or '<unnamed>'} maps outside the raw aligned image ({width}x{height})."
        )

    return {
        "id": point_id,
        "display_u_px": display_u,
        "display_v_px": display_v,
        "raw_u_px": raw_u,
        "raw_v_px": raw_v,
        "center_u": center_u,
        "center_v": center_v,
    }


def _window_values(
    depth: np.ndarray,
    center_u: int,
    center_v: int,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    u0 = max(0, center_u - radius_px)
    u1 = min(width, center_u + radius_px + 1)
    v0 = max(0, center_v - radius_px)
    v1 = min(height, center_v + radius_px + 1)

    window = depth[v0:v1, u0:u1]
    rows, columns = np.nonzero(window > 0)
    if rows.size == 0:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    values = window[rows, columns]
    absolute_u = columns.astype(np.int32) + u0
    absolute_v = rows.astype(np.int32) + v0
    return values, absolute_u, absolute_v


def _direct_sample(
    depth: np.ndarray,
    center_u: int,
    center_v: int,
    radius_px: int,
) -> tuple[float | None, int]:
    values, _, _ = _window_values(depth, center_u, center_v, radius_px)
    if values.size == 0:
        return None, 0
    return float(np.median(values)), int(values.size)


def _face_consistent_fallback(
    depth: np.ndarray,
    *,
    center_u: int,
    center_v: int,
    scale_mm: float,
    reference_depth_mm: float,
    initial_radius_px: int,
) -> tuple[float | None, int, int | None]:
    tolerance_mm = max(40.0, reference_depth_mm * 0.05)
    radii = sorted({max(initial_radius_px + 2, 4), 6, 8})

    for radius_px in radii:
        if radius_px <= initial_radius_px:
            continue

        values, absolute_u, absolute_v = _window_values(
            depth,
            center_u,
            center_v,
            radius_px,
        )
        if values.size == 0:
            continue

        values_mm = values.astype(np.float64) * scale_mm
        consistent = np.abs(values_mm - reference_depth_mm) <= tolerance_mm
        if not np.any(consistent):
            continue

        candidate_values = values[consistent]
        candidate_u = absolute_u[consistent]
        candidate_v = absolute_v[consistent]

        distances_sq = (
            (candidate_u.astype(np.float64) - center_u) ** 2
            + (candidate_v.astype(np.float64) - center_v) ** 2
        )
        order = np.argsort(distances_sq)
        nearest_count = min(9, candidate_values.size)
        nearest_values = candidate_values[order[:nearest_count]]
        return (
            float(np.median(nearest_values)),
            int(nearest_values.size),
            radius_px,
        )

    return None, 0, None


def _camera_geometry_for_capture(
    metadata: dict[str, Any],
    metadata_path: Path,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    geometry = metadata.get("camera_geometry")
    if not isinstance(geometry, dict):
        try:
            geometry = read_active_camera_geometry()
        except CameraGeometryError as exc:
            raise CalibrationDepthError(
                "Gemini intrinsics are required for rigid 3D calibration but could not be read: "
                + str(exc)
            ) from exc

        captured_serial = metadata.get("device", {}).get("serial_number")
        geometry_serial = geometry.get("serial_number")
        if captured_serial and geometry_serial and captured_serial != geometry_serial:
            raise CalibrationDepthError(
                "The connected Gemini serial number does not match the camera that created this capture."
            )

        metadata["camera_geometry"] = geometry
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        intrinsic = geometry["rgb_intrinsic"]
        fx = float(intrinsic["fx"])
        fy = float(intrinsic["fy"])
        intrinsic_width = int(intrinsic["width"])
        intrinsic_height = int(intrinsic["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationDepthError("Stored Gemini RGB intrinsics are incomplete.") from exc

    if fx <= 0 or fy <= 0 or intrinsic_width <= 1 or intrinsic_height <= 1:
        raise CalibrationDepthError("Stored Gemini RGB intrinsics are invalid.")

    capture_aspect = width / height
    intrinsic_aspect = intrinsic_width / intrinsic_height
    if abs(capture_aspect - intrinsic_aspect) > 0.01:
        raise CalibrationDepthError(
            "Gemini RGB intrinsics do not match the captured image aspect ratio."
        )

    return geometry


def _deproject_color_pixel(
    *,
    u_px: float,
    v_px: float,
    depth_mm: float,
    geometry: dict[str, Any],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    intrinsic = geometry["rgb_intrinsic"]
    distortion = geometry.get("rgb_distortion", {})

    intrinsic_width = float(intrinsic["width"])
    intrinsic_height = float(intrinsic["height"])
    scale_x = width / intrinsic_width
    scale_y = height / intrinsic_height

    fx = float(intrinsic["fx"]) * scale_x
    fy = float(intrinsic["fy"]) * scale_y
    cx = float(intrinsic["cx"]) * scale_x
    cy = float(intrinsic["cy"]) * scale_y

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coefficients = np.array(
        [
            float(distortion.get("k1", 0.0)),
            float(distortion.get("k2", 0.0)),
            float(distortion.get("p1", 0.0)),
            float(distortion.get("p2", 0.0)),
            float(distortion.get("k3", 0.0)),
            float(distortion.get("k4", 0.0)),
            float(distortion.get("k5", 0.0)),
            float(distortion.get("k6", 0.0)),
        ],
        dtype=np.float64,
    )

    pixel = np.array([[[u_px, v_px]]], dtype=np.float64)
    normalized = cv2.undistortPoints(pixel, camera_matrix, coefficients)
    x_normalized = float(normalized[0, 0, 0])
    y_normalized = float(normalized[0, 0, 1])

    return (
        x_normalized * depth_mm,
        y_normalized * depth_mm,
        depth_mm,
    )


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
    rotation_degrees = capture_display_rotation(metadata)

    try:
        scale_mm = float(metadata["depth"]["scale_mm_per_unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationDepthError("Capture metadata is missing a valid depth scale.") from exc

    if not np.isfinite(scale_mm) or scale_mm <= 0:
        raise CalibrationDepthError("Capture metadata contains an invalid depth scale.")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise CalibrationDepthError("The aligned raw depth image could not be decoded.")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)

    height, width = depth.shape
    geometry = _camera_geometry_for_capture(
        metadata,
        metadata_path,
        width=width,
        height=height,
    )
    validated = [
        _validated_point(point, width, height, rotation_degrees)
        for point in points
    ]

    initial_samples: list[dict[str, Any]] = []
    reliable_face_depths_mm: list[float] = []

    for point in validated:
        raw_value, valid_count = _direct_sample(
            depth,
            point["center_u"],
            point["center_v"],
            radius_px,
        )
        depth_mm = raw_value * scale_mm if raw_value is not None else None

        initial_samples.append(
            {
                **point,
                "depth_raw": raw_value,
                "depth_mm": depth_mm,
                "valid_sample_count": valid_count,
            }
        )

        if depth_mm is not None and valid_count >= 3:
            reliable_face_depths_mm.append(depth_mm)

    reference_depth_mm = (
        float(np.median(np.asarray(reliable_face_depths_mm, dtype=np.float64)))
        if len(reliable_face_depths_mm) >= 3
        else None
    )

    samples: list[dict[str, Any]] = []
    for sample in initial_samples:
        raw_value = sample["depth_raw"]
        depth_mm = sample["depth_mm"]
        valid_count = sample["valid_sample_count"]
        effective_radius_px = radius_px
        sample_method = "direct"

        if raw_value is None and reference_depth_mm is not None:
            fallback_raw, fallback_count, fallback_radius = _face_consistent_fallback(
                depth,
                center_u=sample["center_u"],
                center_v=sample["center_v"],
                scale_mm=scale_mm,
                reference_depth_mm=reference_depth_mm,
                initial_radius_px=radius_px,
            )
            if fallback_raw is not None and fallback_radius is not None:
                raw_value = fallback_raw
                depth_mm = fallback_raw * scale_mm
                valid_count = fallback_count
                effective_radius_px = fallback_radius
                sample_method = "face_consistent_fallback"

        camera_xyz = None
        if depth_mm is not None:
            camera_xyz = _deproject_color_pixel(
                u_px=sample["raw_u_px"],
                v_px=sample["raw_v_px"],
                depth_mm=depth_mm,
                geometry=geometry,
                width=width,
                height=height,
            )

        samples.append(
            {
                "id": sample["id"],
                "u_px": sample["display_u_px"],
                "v_px": sample["display_v_px"],
                "raw_u_px": sample["raw_u_px"],
                "raw_v_px": sample["raw_v_px"],
                "depth_raw": raw_value,
                "depth_mm": depth_mm,
                "camera_x_mm": camera_xyz[0] if camera_xyz else None,
                "camera_y_mm": camera_xyz[1] if camera_xyz else None,
                "camera_z_mm": camera_xyz[2] if camera_xyz else None,
                "valid_sample_count": int(valid_count),
                "radius_px": radius_px,
                "effective_radius_px": effective_radius_px,
                "sample_method": sample_method,
                "face_reference_depth_mm": reference_depth_mm,
            }
        )

    return {
        "status": "ok",
        "capture_id": capture_id,
        "width": width,
        "height": height,
        "depth_scale_mm": scale_mm,
        "display_rotation_degrees": rotation_degrees,
        "pixel_coordinate_system": "upright_display_pixels",
        "raw_pixel_coordinate_system": "native_gemini_rgb_pixels",
        "camera_geometry": geometry,
        "camera_coordinate_convention": {
            "x": "right_mm_in_raw_rgb_optical_frame",
            "y": "down_mm_in_raw_rgb_optical_frame",
            "z": "forward_optical_depth_mm",
        },
        "samples": samples,
    }
