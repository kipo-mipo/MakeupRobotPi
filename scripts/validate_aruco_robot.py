from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aruco_robot_calibration import (
    ArucoCalibrationError,
    DEFAULT_OUTPUT_PATH as DEFAULT_CALIBRATION_PATH,
    GridPoint,
    _move_xz,
    _printer_status,
    capture_marker_point,
    validate_robot_for_grid,
)
from camera_geometry import CameraGeometryError, read_active_camera_geometry
from gemini_camera import CameraCaptureError, CameraUnavailableError
from gemini_orientation import GeminiOrientationError
from robot_motion import RobotMotionUnavailable


DEFAULT_VALIDATION_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "aruco_robot_validation_latest.json"
)
DEFAULT_TRAVEL_SPEED_MM_S = 35.0
DEFAULT_SETTLE_SECONDS = 0.35
DEFAULT_RMS_LIMIT_MM = 3.0
DEFAULT_MAX_LIMIT_MM = 5.0


def default_validation_points() -> list[GridPoint]:
    coordinates = [
        (27.5, 11.67),
        (82.5, 35.0),
        (137.5, 58.33),
        (192.5, 11.67),
        (27.5, 58.33),
        (82.5, 11.67),
        (137.5, 35.0),
        (192.5, 58.33),
    ]
    return [
        GridPoint(index=index + 1, x_mm=x_mm, z_mm=z_mm)
        for index, (x_mm, z_mm) in enumerate(coordinates)
    ]


def apply_transform(
    camera_xyz_mm: np.ndarray,
    rotation_row_major: list[float],
    translation_mm: list[float],
) -> np.ndarray:
    camera = np.asarray(camera_xyz_mm, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_row_major, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_mm, dtype=np.float64).reshape(3)
    if (
        not np.all(np.isfinite(camera))
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
    ):
        raise ArucoCalibrationError("Saved transform and Camera XYZ must be finite.")
    return rotation @ camera + translation


def residual_stats(errors_mm: list[float]) -> tuple[float, float]:
    if not errors_mm:
        raise ArucoCalibrationError("Validation requires at least one successful point.")
    values = np.asarray(errors_mm, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ArucoCalibrationError("Validation residuals must be finite.")
    return float(np.sqrt(np.mean(values**2))), float(np.max(values))


def load_calibration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArucoCalibrationError(f"Calibration file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArucoCalibrationError(f"Could not read calibration file: {exc}") from exc

    if not isinstance(payload, dict):
        raise ArucoCalibrationError("Calibration file has an unexpected structure.")
    if not payload.get("metrics", {}).get("quality_pass"):
        raise ArucoCalibrationError(
            "Saved calibration did not pass its quality gate; do not validate it for use."
        )
    try:
        rotation = payload["transform"]["rotation_row_major"]
        translation = payload["transform"]["translation_mm"]
        marker = payload["marker"]
        offset = marker["center_minus_nozzle_tip_robot_xyz_mm"]
        marker_id = int(marker["id"])
        marker_size_mm = float(marker["nominal_size_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArucoCalibrationError("Saved calibration is missing transform/marker fields.") from exc

    if len(rotation) != 9 or len(translation) != 3 or len(offset) != 3:
        raise ArucoCalibrationError("Saved calibration transform or marker offset has invalid dimensions.")
    if not math.isfinite(marker_size_mm) or marker_size_mm <= 0:
        raise ArucoCalibrationError("Saved marker size is invalid.")
    return payload


def infer_robot_y_mm(payload: dict[str, Any]) -> float:
    correspondences = payload.get("correspondences")
    if not isinstance(correspondences, list) or not correspondences:
        raise ArucoCalibrationError("Saved calibration has no correspondences.")
    y_values: list[float] = []
    for record in correspondences:
        try:
            xyz = record["command_robot_xyz_mm"]
            y = float(xyz[1])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ArucoCalibrationError(
                "Saved calibration correspondence is missing command Robot XYZ."
            ) from exc
        if not math.isfinite(y):
            raise ArucoCalibrationError("Saved calibration contains non-finite Robot Y.")
        y_values.append(y)
    if max(y_values) - min(y_values) > 1e-6:
        raise ArucoCalibrationError("Saved calibration was not collected at one fixed Robot Y.")
    return y_values[0]


def validate_saved_calibration(
    *,
    calibration_path: Path,
    output_path: Path,
    points: list[GridPoint],
    travel_speed_mm_s: float,
    settle_seconds: float,
    rms_limit_mm: float,
    max_limit_mm: float,
) -> dict[str, Any]:
    payload = load_calibration(calibration_path)
    status = _printer_status()
    validate_robot_for_grid(status, points)

    geometry = read_active_camera_geometry()
    saved_serial = payload.get("camera", {}).get("serial_number")
    active_serial = geometry.get("serial_number")
    if saved_serial and active_serial and saved_serial != active_serial:
        raise ArucoCalibrationError(
            f"Connected Gemini serial {active_serial} does not match calibration camera {saved_serial}."
        )

    robot_y_mm = infer_robot_y_mm(payload)
    marker = payload["marker"]
    marker_id = int(marker["id"])
    marker_size_mm = float(marker["nominal_size_mm"])
    offset = np.asarray(
        marker["center_minus_nozzle_tip_robot_xyz_mm"],
        dtype=np.float64,
    ).reshape(3)
    rotation = payload["transform"]["rotation_row_major"]
    translation = payload["transform"]["translation_mm"]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for point in points:
        print(
            f"[{point.index:02d}/{len(points):02d}] "
            f"X={point.x_mm:.2f}, Z={point.z_mm:.2f}",
            flush=True,
        )
        _move_xz(point, travel_speed_mm_s)
        time.sleep(settle_seconds)

        try:
            measured = capture_marker_point(
                marker_id=marker_id,
                marker_size_mm=marker_size_mm,
                geometry=geometry,
            )
            camera_xyz = np.asarray(measured["camera_xyz_mm"], dtype=np.float64)
            predicted_robot = apply_transform(camera_xyz, rotation, translation)
            expected_robot = (
                np.asarray([point.x_mm, robot_y_mm, point.z_mm], dtype=np.float64)
                + offset
            )
            delta = predicted_robot - expected_robot
            error = float(np.linalg.norm(delta))

            record = {
                "grid_index": point.index,
                "command_robot_xyz_mm": [point.x_mm, robot_y_mm, point.z_mm],
                "expected_marker_robot_xyz_mm": expected_robot.tolist(),
                "camera_xyz_mm": camera_xyz.tolist(),
                "camera_xyz_method": measured["camera_xyz_method"],
                "predicted_marker_robot_xyz_mm": predicted_robot.tolist(),
                "delta_xyz_mm": delta.tolist(),
                "error_mm": error,
                "pnp_reprojection_rms_px": measured["pnp_diagnostics"]["reprojection_rms_px"],
                "pnp_minimum_marker_side_px": measured["pnp_diagnostics"]["minimum_marker_side_px"],
                "depth_diagnostics": measured["depth_diagnostics"],
                "capture_id": measured["capture_id"],
            }
            records.append(record)
            print(
                "  ΔXYZ="
                + ", ".join(f"{value:+.2f}" for value in delta.tolist())
                + f" mm; |Δ|={error:.2f} mm; "
                + f"PnP RMS={record['pnp_reprojection_rms_px']:.3f}px"
            )
        except (
            ArucoCalibrationError,
            CameraGeometryError,
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

    if len(records) < 6:
        raise ArucoCalibrationError(
            f"Only {len(records)} independent validation points succeeded; at least 6 are required."
        )

    errors = [float(record["error_mm"]) for record in records]
    rms_mm, maximum_mm = residual_stats(errors)

    delta_matrix = np.asarray([record["delta_xyz_mm"] for record in records], dtype=np.float64)
    axis_mean = delta_matrix.mean(axis=0)
    axis_rms = np.sqrt(np.mean(delta_matrix**2, axis=0))

    passed = rms_mm <= rms_limit_mm and maximum_mm <= max_limit_mm
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_type": "heldout_off_grid_aruco_camera_to_robot",
        "source_calibration_path": str(calibration_path),
        "source_calibration_created_at": payload.get("created_at"),
        "camera_serial_number": active_serial,
        "marker": marker,
        "metrics": {
            "successful_points": len(records),
            "failed_points": len(failures),
            "rms_error_mm": rms_mm,
            "maximum_error_mm": maximum_mm,
            "mean_delta_xyz_mm": axis_mean.tolist(),
            "axis_rms_xyz_mm": axis_rms.tolist(),
            "rms_limit_mm": float(rms_limit_mm),
            "maximum_limit_mm": float(max_limit_mm),
            "quality_pass": bool(passed),
        },
        "points": records,
        "failures": failures,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the saved ArUco camera-to-robot calibration at eight "
            "off-grid X/Z positions without refitting the transform."
        )
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION_OUTPUT_PATH)
    parser.add_argument("--travel-speed", type=float, default=DEFAULT_TRAVEL_SPEED_MM_S)
    parser.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--rms-limit-mm", type=float, default=DEFAULT_RMS_LIMIT_MM)
    parser.add_argument("--max-limit-mm", type=float, default=DEFAULT_MAX_LIMIT_MM)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move X/Z and capture validation points. Default is preview-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for name, value in {
            "travel speed": args.travel_speed,
            "settle seconds": args.settle_seconds,
            "RMS limit": args.rms_limit_mm,
            "maximum limit": args.max_limit_mm,
        }.items():
            if not math.isfinite(float(value)):
                raise ArucoCalibrationError(f"{name} must be finite.")
        if args.travel_speed <= 0:
            raise ArucoCalibrationError("Travel speed must be positive.")
        if args.settle_seconds < 0:
            raise ArucoCalibrationError("Settle seconds cannot be negative.")
        if args.rms_limit_mm <= 0 or args.max_limit_mm <= 0:
            raise ArucoCalibrationError("Validation limits must be positive.")

        payload = load_calibration(args.calibration)
        points = default_validation_points()
        robot_y_mm = infer_robot_y_mm(payload)
        offset = payload["marker"]["center_minus_nozzle_tip_robot_xyz_mm"]

        print("\nIndependent ArUco calibration validation")
        print("----------------------------------------")
        print(f"Calibration: {args.calibration}")
        print(f"Marker offset XYZ: {offset}")
        print(f"Robot Y: {robot_y_mm:.3f} mm")
        print(f"Validation points: {len(points)}")
        print("These points are between the 5×4 calibration grid lines; the transform is NOT refit.")
        print("Motion: X/Z only. Y, airbrush servo, and solenoid are never commanded.")
        for point in points:
            print(f"  {point.index:02d}: X={point.x_mm:7.2f}  Z={point.z_mm:6.2f}")

        if not args.execute:
            print("\nPREVIEW ONLY — no motion or camera capture was started.")
            print("Re-run with --execute after checking the eight positions.")
            return 0

        result = validate_saved_calibration(
            calibration_path=args.calibration,
            output_path=args.output,
            points=points,
            travel_speed_mm_s=float(args.travel_speed),
            settle_seconds=float(args.settle_seconds),
            rms_limit_mm=float(args.rms_limit_mm),
            max_limit_mm=float(args.max_limit_mm),
        )

        metrics = result["metrics"]
        print("\nIndependent validation result")
        print("-----------------------------")
        print(f"Successful points: {metrics['successful_points']} / {len(points)}")
        print(f"RMS error:         {metrics['rms_error_mm']:.3f} mm")
        print(f"Maximum error:     {metrics['maximum_error_mm']:.3f} mm")
        print(
            "Mean ΔXYZ:         "
            + "  ".join(f"{value:+.3f}" for value in metrics["mean_delta_xyz_mm"])
            + " mm"
        )
        print(
            "Axis RMS XYZ:      "
            + "  ".join(f"{value:.3f}" for value in metrics["axis_rms_xyz_mm"])
            + " mm"
        )
        print(
            f"Quality gate:      {'PASS' if metrics['quality_pass'] else 'FAIL'} "
            f"(RMS ≤ {metrics['rms_limit_mm']:.1f} mm, "
            f"max ≤ {metrics['maximum_limit_mm']:.1f} mm)"
        )
        print(f"Saved: {args.output}")
        return 0 if metrics["quality_pass"] else 2

    except (
        ArucoCalibrationError,
        CameraGeometryError,
        CameraCaptureError,
        CameraUnavailableError,
        GeminiOrientationError,
        RobotMotionUnavailable,
        KeyboardInterrupt,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
