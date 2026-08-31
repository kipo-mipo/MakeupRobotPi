from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibration_depth import CalibrationDepthError, sample_capture_depth
from camera_geometry import CameraGeometryError, read_active_camera_geometry
from gemini_camera import CAPTURE_DIR, CameraCaptureError, CameraUnavailableError, capture_calibration
from gemini_orientation import (
    GeminiOrientationError,
    capture_display_rotation,
    display_to_raw_pixel,
    prepare_capture_display,
)
from robot_motion import RobotMotionUnavailable, _request_json, _result


DEFAULT_X_MIN_MM = 0.0
DEFAULT_X_MAX_MM = 220.0
DEFAULT_Z_MIN_MM = 0.0
DEFAULT_Z_MAX_MM = 70.0
DEFAULT_X_COUNT = 5
DEFAULT_Z_COUNT = 4
DEFAULT_MARKER_ID = 0
DEFAULT_MARKER_SIZE_MM = 30.0
DEFAULT_TRAVEL_SPEED_MM_S = 35.0
DEFAULT_SETTLE_SECONDS = 0.35
DEFAULT_DEPTH_RADIUS_PX = 2
DEFAULT_MAX_PLANE_RMS_MM = 2.0
DEFAULT_ACCEPTABLE_ERROR_MM = 5.0
MIN_CALIBRATION_POINTS = 8

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_OUTPUT_PATH = CONFIG_DIR / "aruco_robot_calibration_latest.json"


class ArucoCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GridPoint:
    index: int
    x_mm: float
    z_mm: float


@dataclass(frozen=True)
class RigidFit:
    rotation: np.ndarray
    translation: np.ndarray
    residuals_mm: np.ndarray
    rms_mm: float
    maximum_mm: float


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArucoCalibrationError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ArucoCalibrationError(f"{name} must be finite.")
    return result


def build_grid(
    *,
    x_min_mm: float = DEFAULT_X_MIN_MM,
    x_max_mm: float = DEFAULT_X_MAX_MM,
    z_min_mm: float = DEFAULT_Z_MIN_MM,
    z_max_mm: float = DEFAULT_Z_MAX_MM,
    x_count: int = DEFAULT_X_COUNT,
    z_count: int = DEFAULT_Z_COUNT,
) -> list[GridPoint]:
    x_min = _finite(x_min_mm, "X minimum")
    x_max = _finite(x_max_mm, "X maximum")
    z_min = _finite(z_min_mm, "Z minimum")
    z_max = _finite(z_max_mm, "Z maximum")
    if x_max <= x_min or z_max <= z_min:
        raise ArucoCalibrationError("Calibration X/Z maximums must exceed minimums.")
    if x_count < 2 or z_count < 2:
        raise ArucoCalibrationError("Calibration grid needs at least two X and two Z positions.")

    xs = np.linspace(x_min, x_max, x_count).tolist()
    zs = np.linspace(z_min, z_max, z_count).tolist()
    result: list[GridPoint] = []
    index = 1
    for row, z_mm in enumerate(zs):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x_mm in row_xs:
            result.append(GridPoint(index=index, x_mm=float(x_mm), z_mm=float(z_mm)))
            index += 1
    return result


def build_xz_move_commands(point: GridPoint, travel_speed_mm_s: float) -> list[str]:
    speed = _finite(travel_speed_mm_s, "travel speed")
    if speed <= 0:
        raise ArucoCalibrationError("Travel speed must be positive.")
    return [
        "M400",
        "G90",
        f"G0 X{point.x_mm:.3f} Z{point.z_mm:.3f} F{speed * 60.0:.0f}",
        "M400",
    ]


def marker_robot_coordinate(
    command_xyz_mm: np.ndarray,
    marker_center_minus_nozzle_robot_mm: np.ndarray,
) -> np.ndarray:
    command = np.asarray(command_xyz_mm, dtype=np.float64).reshape(3)
    offset = np.asarray(marker_center_minus_nozzle_robot_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(offset)):
        raise ArucoCalibrationError("Robot command and marker offset must be finite.")
    return command + offset


def fit_rigid_transform(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray) -> RigidFit:
    camera = np.asarray(camera_points_mm, dtype=np.float64)
    robot = np.asarray(robot_points_mm, dtype=np.float64)
    if camera.ndim != 2 or camera.shape != robot.shape or camera.shape[1:] != (3,):
        raise ArucoCalibrationError("Rigid fit requires matching N x 3 point arrays.")
    if camera.shape[0] < 3:
        raise ArucoCalibrationError("Rigid fit requires at least three points.")
    if not np.all(np.isfinite(camera)) or not np.all(np.isfinite(robot)):
        raise ArucoCalibrationError("Rigid fit points must be finite.")

    camera_center = camera.mean(axis=0)
    robot_center = robot.mean(axis=0)
    camera_zero = camera - camera_center
    robot_zero = robot - robot_center
    if np.linalg.matrix_rank(camera_zero, tol=1e-7) < 2:
        raise ArucoCalibrationError("Camera calibration points are collinear.")
    if np.linalg.matrix_rank(robot_zero, tol=1e-7) < 2:
        raise ArucoCalibrationError("Robot calibration points are collinear.")

    u, _, vt = np.linalg.svd(camera_zero.T @ robot_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = robot_center - rotation @ camera_center

    predicted = (rotation @ camera.T).T + translation
    residuals = np.linalg.norm(predicted - robot, axis=1)
    return RigidFit(
        rotation=rotation,
        translation=translation,
        residuals_mm=residuals,
        rms_mm=float(np.sqrt(np.mean(residuals**2))),
        maximum_mm=float(np.max(residuals)),
    )


def leave_one_out_errors(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray) -> np.ndarray:
    camera = np.asarray(camera_points_mm, dtype=np.float64)
    robot = np.asarray(robot_points_mm, dtype=np.float64)
    if camera.shape[0] < 4:
        raise ArucoCalibrationError("Leave-one-out validation needs at least four points.")
    errors: list[float] = []
    for held_out in range(camera.shape[0]):
        keep = np.ones(camera.shape[0], dtype=bool)
        keep[held_out] = False
        fit = fit_rigid_transform(camera[keep], robot[keep])
        predicted = fit.rotation @ camera[held_out] + fit.translation
        errors.append(float(np.linalg.norm(predicted - robot[held_out])))
    return np.asarray(errors, dtype=np.float64)


def _printer_status() -> dict[str, Any]:
    printer_info = _result(_request_json("GET", "/printer/info"))
    query = _result(
        _request_json(
            "POST",
            "/printer/objects/query",
            {
                "objects": {
                    "toolhead": ["homed_axes", "position", "axis_minimum", "axis_maximum"],
                    "print_stats": ["state"],
                }
            },
        )
    )
    status = query.get("status") or {}
    toolhead = status.get("toolhead") or {}
    print_stats = status.get("print_stats") or {}
    return {
        "printer_state": str(printer_info.get("state") or ""),
        "homed_axes": str(toolhead.get("homed_axes") or ""),
        "axis_minimum": toolhead.get("axis_minimum"),
        "axis_maximum": toolhead.get("axis_maximum"),
        "print_state": str(print_stats.get("state") or ""),
    }


def validate_robot_for_grid(status: dict[str, Any], grid: list[GridPoint]) -> None:
    if status.get("printer_state", "").lower() != "ready":
        raise ArucoCalibrationError("Klipper is not ready.")
    homed = status.get("homed_axes", "").lower()
    missing = [axis.upper() for axis in ("x", "z") if axis not in homed]
    if missing:
        raise ArucoCalibrationError("Home X/Z before calibration; missing " + ", ".join(missing) + ".")
    if status.get("print_state", "").lower() in {"printing", "paused"}:
        raise ArucoCalibrationError("Calibration is blocked while a print is active or paused.")

    minimum = status.get("axis_minimum")
    maximum = status.get("axis_maximum")
    if not isinstance(minimum, list) or not isinstance(maximum, list) or len(minimum) < 3 or len(maximum) < 3:
        raise ArucoCalibrationError("Klipper did not report XYZ axis limits.")
    for point in grid:
        if not float(minimum[0]) <= point.x_mm <= float(maximum[0]):
            raise ArucoCalibrationError(f"X={point.x_mm:.3f} is outside Klipper limits.")
        if not float(minimum[2]) <= point.z_mm <= float(maximum[2]):
            raise ArucoCalibrationError(f"Z={point.z_mm:.3f} is outside Klipper limits.")


def _move_xz(point: GridPoint, travel_speed_mm_s: float) -> None:
    script = "\n".join(build_xz_move_commands(point, travel_speed_mm_s))
    _result(
        _request_json(
            "POST",
            "/printer/gcode/script",
            {"script": script},
            timeout_seconds=90.0,
        )
    )


def _detector() -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise ArucoCalibrationError("OpenCV ArUco support is unavailable; install opencv-contrib-python.")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def detect_marker_corners(image_bgr: np.ndarray, marker_id: int = DEFAULT_MARKER_ID) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = _detector()
    if isinstance(detector, tuple):
        dictionary, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    else:
        corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        raise ArucoCalibrationError(f"ArUco marker ID {marker_id} was not detected.")
    matches = [i for i, value in enumerate(ids.reshape(-1)) if int(value) == int(marker_id)]
    if len(matches) != 1:
        if len(matches) > 1:
            raise ArucoCalibrationError(f"Multiple ID {marker_id} markers are visible; remove duplicate printouts.")
        raise ArucoCalibrationError(f"ArUco marker ID {marker_id} was not detected.")
    result = np.asarray(corners[matches[0]], dtype=np.float64).reshape(4, 2)
    sides = [np.linalg.norm(result[(i + 1) % 4] - result[i]) for i in range(4)]
    if min(sides) < 20.0:
        raise ArucoCalibrationError(f"Marker is too small in the RGB image ({min(sides):.1f}px minimum side).")
    return result


def _quad_point(corners: np.ndarray, u: float, v: float) -> np.ndarray:
    c0, c1, c2, c3 = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return (
        (1 - u) * (1 - v) * c0
        + u * (1 - v) * c1
        + u * v * c2
        + (1 - u) * v * c3
    )


def _scaled_rgb_camera_model(
    geometry: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        intrinsic = geometry["rgb_intrinsic"]
        distortion = geometry.get("rgb_distortion", {})
        intrinsic_width = float(intrinsic["width"])
        intrinsic_height = float(intrinsic["height"])
        scale_x = float(width) / intrinsic_width
        scale_y = float(height) / intrinsic_height
        camera_matrix = np.asarray(
            [
                [float(intrinsic["fx"]) * scale_x, 0.0, float(intrinsic["cx"]) * scale_x],
                [0.0, float(intrinsic["fy"]) * scale_y, float(intrinsic["cy"]) * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        coefficients = np.asarray(
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
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ArucoCalibrationError("Gemini RGB camera geometry is incomplete for ArUco pose estimation.") from exc

    if (
        not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(coefficients))
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
    ):
        raise ArucoCalibrationError("Gemini RGB camera geometry is invalid for ArUco pose estimation.")
    return camera_matrix, coefficients


def estimate_marker_center_camera_xyz_pnp(
    corners_display_px: np.ndarray,
    *,
    marker_size_mm: float,
    geometry: dict[str, Any],
    width: int,
    height: int,
    rotation_degrees: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = _finite(marker_size_mm, "marker size")
    if size <= 0:
        raise ArucoCalibrationError("Marker size must be positive for RGB pose estimation.")

    corners_display = np.asarray(corners_display_px, dtype=np.float64).reshape(4, 2)
    raw_corners = np.asarray(
        [
            display_to_raw_pixel(
                float(point[0]),
                float(point[1]),
                width=width,
                height=height,
                rotation_degrees=rotation_degrees,
            )
            for point in corners_display
        ],
        dtype=np.float64,
    )
    camera_matrix, coefficients = _scaled_rgb_camera_model(
        geometry,
        width=width,
        height=height,
    )

    half = size / 2.0
    object_points = np.asarray(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        raw_corners.reshape(4, 1, 2),
        camera_matrix,
        coefficients,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise ArucoCalibrationError("OpenCV could not solve the 30 mm ArUco marker pose.")

    center_xyz = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_xyz)) or center_xyz[2] <= 0:
        raise ArucoCalibrationError("ArUco RGB pose returned an invalid marker-center Camera XYZ.")

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        coefficients,
    )
    projected = np.asarray(projected, dtype=np.float64).reshape(4, 2)
    reprojection = np.linalg.norm(projected - raw_corners, axis=1)
    reprojection_rms = float(np.sqrt(np.mean(reprojection**2)))
    side_lengths = [
        float(np.linalg.norm(raw_corners[(index + 1) % 4] - raw_corners[index]))
        for index in range(4)
    ]

    return center_xyz, {
        "method": "rgb_aruco_pnp_ippe_square",
        "marker_size_mm": size,
        "reprojection_rms_px": reprojection_rms,
        "reprojection_max_px": float(np.max(reprojection)),
        "minimum_marker_side_px": min(side_lengths),
        "raw_marker_corners_px": raw_corners.tolist(),
        "rotation_vector": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
    }


def _depth_request_points(corners: np.ndarray) -> list[dict[str, Any]]:
    fractions = (0.30, 0.50, 0.70)
    points: list[dict[str, Any]] = []
    for row, v in enumerate(fractions):
        for column, u in enumerate(fractions):
            pixel = _quad_point(corners, u, v)
            points.append(
                {
                    "id": f"tag_{row}_{column}",
                    "u_px": float(pixel[0]),
                    "v_px": float(pixel[1]),
                }
            )
    return points


def _center_ray(depth_result: dict[str, Any], center_sample: dict[str, Any]) -> np.ndarray:
    intrinsic = depth_result["camera_geometry"]["rgb_intrinsic"]
    width = float(depth_result["width"])
    height = float(depth_result["height"])
    sx = width / float(intrinsic["width"])
    sy = height / float(intrinsic["height"])
    fx = float(intrinsic["fx"]) * sx
    fy = float(intrinsic["fy"]) * sy
    cx = float(intrinsic["cx"]) * sx
    cy = float(intrinsic["cy"]) * sy
    u = float(center_sample["aligned_u_px"])
    v = float(center_sample["aligned_v_px"])
    return np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)


def estimate_marker_center_camera_xyz(
    capture_id: str,
    corners: np.ndarray,
    *,
    depth_radius_px: int = DEFAULT_DEPTH_RADIUS_PX,
    max_plane_rms_mm: float = DEFAULT_MAX_PLANE_RMS_MM,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = sample_capture_depth(
        capture_id,
        _depth_request_points(corners),
        radius_px=depth_radius_px,
    )
    valid: list[list[float]] = []
    center_sample: dict[str, Any] | None = None
    for sample in result["samples"]:
        if sample["id"] == "tag_1_1":
            center_sample = sample
        xyz = [sample["camera_x_mm"], sample["camera_y_mm"], sample["camera_z_mm"]]
        if all(value is not None and math.isfinite(float(value)) for value in xyz):
            valid.append([float(value) for value in xyz])
    if center_sample is None:
        raise ArucoCalibrationError("Depth sampling did not return the tag center.")
    if len(valid) < 5:
        raise ArucoCalibrationError(f"Only {len(valid)}/9 tag depth samples were valid.")

    xyz = np.asarray(valid, dtype=np.float64)
    centroid = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centroid, full_matrices=False)
    normal = vt[-1]
    plane_rms = float(np.sqrt(np.mean(((xyz - centroid) @ normal) ** 2)))
    if plane_rms > max_plane_rms_mm:
        raise ArucoCalibrationError(
            f"Tag depth plane RMS {plane_rms:.2f} mm exceeds {max_plane_rms_mm:.2f} mm."
        )

    ray = _center_ray(result, center_sample)
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-5:
        raise ArucoCalibrationError("Tag is too edge-on for a stable center estimate.")
    scale = float((normal @ centroid) / denominator)
    if scale <= 0:
        raise ArucoCalibrationError("Computed tag center is behind the camera.")
    center_xyz = ray * scale
    return center_xyz, {
        "valid_depth_samples": len(valid),
        "plane_rms_mm": plane_rms,
        "center_plane_depth_mm": float(center_xyz[2]),
        "center_direct_depth_mm": center_sample.get("camera_z_mm"),
    }


def _inject_geometry(metadata_filename: str, geometry: dict[str, Any]) -> None:
    path = CAPTURE_DIR / Path(metadata_filename).name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["camera_geometry"] = geometry
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def capture_marker_point(
    *,
    marker_id: int = DEFAULT_MARKER_ID,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
    geometry: dict[str, Any] | None = None,
    depth_radius_px: int = DEFAULT_DEPTH_RADIUS_PX,
    max_plane_rms_mm: float = DEFAULT_MAX_PLANE_RMS_MM,
) -> dict[str, Any]:
    active_geometry = geometry or read_active_camera_geometry()
    capture = capture_calibration()
    display = prepare_capture_display(
        capture_id=capture.capture_id,
        color_filename=capture.color_filename,
        metadata_filename=capture.metadata_filename,
    )
    _inject_geometry(capture.metadata_filename, active_geometry)
    display_path = CAPTURE_DIR / display["display_color_filename"]
    image = cv2.imread(str(display_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ArucoCalibrationError("Could not decode the upright Gemini RGB capture.")
    corners = detect_marker_corners(image, marker_id)

    metadata_path = CAPTURE_DIR / Path(capture.metadata_filename).name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rotation_degrees = capture_display_rotation(metadata)
    pnp_xyz, pnp_diagnostics = estimate_marker_center_camera_xyz_pnp(
        corners,
        marker_size_mm=marker_size_mm,
        geometry=active_geometry,
        width=capture.width,
        height=capture.height,
        rotation_degrees=rotation_degrees,
    )

    depth_xyz: np.ndarray | None = None
    depth_diagnostics: dict[str, Any]
    try:
        depth_xyz, depth_details = estimate_marker_center_camera_xyz(
            capture.capture_id,
            corners,
            depth_radius_px=depth_radius_px,
            max_plane_rms_mm=max_plane_rms_mm,
        )
        depth_diagnostics = {
            "available": True,
            **depth_details,
            "pnp_vs_depth_center_difference_mm": float(np.linalg.norm(pnp_xyz - depth_xyz)),
        }
    except (ArucoCalibrationError, CalibrationDepthError) as exc:
        depth_diagnostics = {
            "available": False,
            "error": str(exc),
        }

    if depth_xyz is not None:
        camera_xyz = depth_xyz
        position_method = "aligned_depth_plane"
    else:
        camera_xyz = pnp_xyz
        position_method = "rgb_aruco_pnp_fallback"

    return {
        "capture_id": capture.capture_id,
        "display_filename": display["display_color_filename"],
        "camera_xyz_mm": camera_xyz.tolist(),
        "camera_xyz_method": position_method,
        "marker_corners_display_px": corners.tolist(),
        "pnp_diagnostics": pnp_diagnostics,
        "depth_diagnostics": depth_diagnostics,
    }


def run_calibration(
    *,
    grid: list[GridPoint],
    marker_offset_robot_mm: np.ndarray,
    robot_y_mm: float,
    travel_speed_mm_s: float,
    settle_seconds: float,
    marker_id: int,
    marker_size_mm: float,
    depth_radius_px: int,
    max_plane_rms_mm: float,
    acceptable_error_mm: float,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    status = _printer_status()
    validate_robot_for_grid(status, grid)
    geometry = read_active_camera_geometry()

    offset = np.asarray(marker_offset_robot_mm, dtype=np.float64).reshape(3)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for point in grid:
        print(f"[{point.index:02d}/{len(grid):02d}] X={point.x_mm:.2f}, Z={point.z_mm:.2f}", flush=True)
        _move_xz(point, travel_speed_mm_s)
        time.sleep(settle_seconds)
        try:
            measured = capture_marker_point(
                marker_id=marker_id,
                marker_size_mm=marker_size_mm,
                geometry=geometry,
                depth_radius_px=depth_radius_px,
                max_plane_rms_mm=max_plane_rms_mm,
            )
            command_xyz = np.asarray([point.x_mm, robot_y_mm, point.z_mm], dtype=np.float64)
            marker_xyz = marker_robot_coordinate(command_xyz, offset)
            measured["grid_index"] = point.index
            measured["command_robot_xyz_mm"] = command_xyz.tolist()
            measured["marker_robot_xyz_mm"] = marker_xyz.tolist()
            records.append(measured)
            detail = measured["camera_xyz_method"]
            if measured["depth_diagnostics"].get("available"):
                detail += (
                    f"; plane RMS={measured['depth_diagnostics']['plane_rms_mm']:.2f} mm"
                )
            else:
                detail += (
                    f"; PnP reproj RMS={measured['pnp_diagnostics']['reprojection_rms_px']:.2f}px"
                )
            print(
                "  OK camera XYZ="
                + ", ".join(f"{value:.2f}" for value in measured["camera_xyz_mm"])
                + f"; {detail}"
            )
        except (
            ArucoCalibrationError,
            CalibrationDepthError,
            CameraCaptureError,
            CameraUnavailableError,
            GeminiOrientationError,
        ) as exc:
            failures.append(
                {
                    "grid_index": point.index,
                    "command_x_mm": point.x_mm,
                    "command_z_mm": point.z_mm,
                    "reason": str(exc),
                }
            )
            print(f"  SKIP: {exc}")

    if len(records) < MIN_CALIBRATION_POINTS:
        raise ArucoCalibrationError(
            f"Only {len(records)} valid points were captured; at least {MIN_CALIBRATION_POINTS} are required."
        )
    if len({round(r["marker_robot_xyz_mm"][0], 6) for r in records}) < 3:
        raise ArucoCalibrationError("Successful points do not span at least three X positions.")
    if len({round(r["marker_robot_xyz_mm"][2], 6) for r in records}) < 3:
        raise ArucoCalibrationError("Successful points do not span at least three Z positions.")

    camera = np.asarray([r["camera_xyz_mm"] for r in records], dtype=np.float64)
    robot = np.asarray([r["marker_robot_xyz_mm"] for r in records], dtype=np.float64)
    fit = fit_rigid_transform(camera, robot)
    loo = leave_one_out_errors(camera, robot)
    loo_rms = float(np.sqrt(np.mean(loo**2)))
    loo_max = float(np.max(loo))
    passed = (
        fit.rms_mm <= acceptable_error_mm
        and loo_rms <= acceptable_error_mm
        and fit.maximum_mm <= 2.0 * acceptable_error_mm
        and loo_max <= 2.0 * acceptable_error_mm
    )

    for record, train_error, loo_error in zip(records, fit.residuals_mm, loo):
        record["training_error_mm"] = float(train_error)
        record["leave_one_out_error_mm"] = float(loo_error)

    payload = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_type": "aruco_hybrid_camera_to_robot_rigid",
        "coordinate_model": "undistorted_rgb_pinhole_camera_xyz_mm_to_robot_xyz_mm_rigid",
        "marker": {
            "dictionary": "DICT_4X4_50",
            "id": int(marker_id),
            "nominal_size_mm": float(marker_size_mm),
            "center_minus_nozzle_tip_robot_xyz_mm": offset.tolist(),
        },
        "camera": {
            "device_name": geometry.get("device_name"),
            "serial_number": geometry.get("serial_number"),
            "model": geometry.get("model"),
        },
        "transform": {
            "rotation_row_major": fit.rotation.reshape(-1).tolist(),
            "translation_mm": fit.translation.tolist(),
        },
        "metrics": {
            "successful_points": len(records),
            "failed_candidates": len(failures),
            "training_rms_mm": fit.rms_mm,
            "training_maximum_mm": fit.maximum_mm,
            "leave_one_out_rms_mm": loo_rms,
            "leave_one_out_maximum_mm": loo_max,
            "camera_depth_span_mm": float(np.ptp(camera[:, 2])),
            "acceptable_error_mm": float(acceptable_error_mm),
            "quality_pass": bool(passed),
        },
        "correspondences": records,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload
