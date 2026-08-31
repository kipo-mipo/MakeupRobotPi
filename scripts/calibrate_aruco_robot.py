from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aruco_robot_calibration import (
    ArucoCalibrationError,
    DEFAULT_ACCEPTABLE_ERROR_MM,
    DEFAULT_DEPTH_RADIUS_PX,
    DEFAULT_MARKER_ID,
    DEFAULT_MARKER_SIZE_MM,
    DEFAULT_MAX_PLANE_RMS_MM,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SETTLE_SECONDS,
    DEFAULT_TRAVEL_SPEED_MM_S,
    DEFAULT_X_COUNT,
    DEFAULT_X_MAX_MM,
    DEFAULT_X_MIN_MM,
    DEFAULT_Z_COUNT,
    DEFAULT_Z_MAX_MM,
    DEFAULT_Z_MIN_MM,
    build_grid,
    capture_marker_point,
    run_calibration,
)
from calibration_depth import CalibrationDepthError
from camera_geometry import CameraGeometryError
from gemini_camera import CameraCaptureError, CameraUnavailableError
from gemini_orientation import GeminiOrientationError
from robot_motion import RobotMotionUnavailable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatic Gemini-to-robot rigid calibration using ArUco "
            "DICT_4X4_50 marker ID 0. X/Z only; no Y, servo, or solenoid commands."
        )
    )
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN_MM)
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX_MM)
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN_MM)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX_MM)
    parser.add_argument("--x-count", type=int, default=DEFAULT_X_COUNT)
    parser.add_argument("--z-count", type=int, default=DEFAULT_Z_COUNT)
    parser.add_argument("--robot-y", type=float, default=0.0)
    parser.add_argument("--travel-speed", type=float, default=DEFAULT_TRAVEL_SPEED_MM_S)
    parser.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size-mm", type=float, default=DEFAULT_MARKER_SIZE_MM)
    parser.add_argument("--depth-radius-px", type=int, default=DEFAULT_DEPTH_RADIUS_PX)
    parser.add_argument("--max-plane-rms-mm", type=float, default=DEFAULT_MAX_PLANE_RMS_MM)
    parser.add_argument("--acceptable-error-mm", type=float, default=DEFAULT_ACCEPTABLE_ERROR_MM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument(
        "--marker-offset-x",
        type=float,
        default=None,
        help="Marker center X minus nozzle tip X in Robot coordinates, mm.",
    )
    parser.add_argument(
        "--marker-offset-y",
        type=float,
        default=None,
        help="Marker center Y minus nozzle tip Y in Robot coordinates, mm.",
    )
    parser.add_argument(
        "--marker-offset-z",
        type=float,
        default=None,
        help="Marker center Z minus nozzle tip Z in Robot coordinates, mm.",
    )
    parser.add_argument(
        "--check-marker",
        action="store_true",
        help="Capture and verify the current marker/depth without moving the robot.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move X/Z and run the calibration. Default is preview-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        grid = build_grid(
            x_min_mm=args.x_min,
            x_max_mm=args.x_max,
            z_min_mm=args.z_min,
            z_max_mm=args.z_max,
            x_count=args.x_count,
            z_count=args.z_count,
        )
        numeric_checks = {
            "marker size": args.marker_size_mm,
            "travel speed": args.travel_speed,
            "settle seconds": args.settle_seconds,
            "maximum plane RMS": args.max_plane_rms_mm,
            "acceptable error": args.acceptable_error_mm,
            "Robot Y": args.robot_y,
        }
        for name, value in numeric_checks.items():
            if not np.isfinite(float(value)):
                raise ArucoCalibrationError(f"{name} must be finite.")
        if args.marker_size_mm <= 0:
            raise ArucoCalibrationError("Marker size must be positive.")
        if args.travel_speed <= 0:
            raise ArucoCalibrationError("Travel speed must be positive.")
        if args.settle_seconds < 0:
            raise ArucoCalibrationError("Settle seconds cannot be negative.")
        if args.max_plane_rms_mm <= 0 or args.acceptable_error_mm <= 0:
            raise ArucoCalibrationError("Plane RMS and acceptable-error limits must be positive.")
        if not 0 <= args.depth_radius_px <= 10:
            raise ArucoCalibrationError("Depth radius must be between 0 and 10 pixels.")

        print("\nGemini ↔ robot ArUco calibration")
        print("---------------------------------")
        print(f"Marker: DICT_4X4_50 ID {args.marker_id}, nominal {args.marker_size_mm:.2f} mm")
        print(f"Grid: {args.x_count} × {args.z_count} = {len(grid)} candidate points")
        print(f"X range: {args.x_min:.2f} … {args.x_max:.2f} mm")
        print(f"Z range: {args.z_min:.2f} … {args.z_max:.2f} mm")
        print("Motion: X/Z only. Y, airbrush servo, and solenoid are never commanded.")
        for point in grid:
            print(f"  {point.index:02d}: X={point.x_mm:7.2f}  Z={point.z_mm:6.2f}")

        if args.check_marker and args.execute:
            raise ArucoCalibrationError("Use either --check-marker or --execute, not both.")

        if args.check_marker:
            measured = capture_marker_point(
                marker_id=args.marker_id,
                depth_radius_px=args.depth_radius_px,
                max_plane_rms_mm=args.max_plane_rms_mm,
            )
            xyz = measured["camera_xyz_mm"]
            diag = measured["depth_diagnostics"]
            print("\nMARKER CHECK PASS")
            print("Camera XYZ: " + ", ".join(f"{value:.2f}" for value in xyz) + " mm")
            print(f"Valid tag depth samples: {diag['valid_depth_samples']}/9")
            print(f"Tag depth-plane RMS: {diag['plane_rms_mm']:.3f} mm")
            print(f"Capture: {measured['capture_id']}")
            return 0

        offsets = (args.marker_offset_x, args.marker_offset_y, args.marker_offset_z)
        if not args.execute:
            print("\nPREVIEW ONLY — no robot motion or camera capture was started.")
            print("Run with --check-marker first to verify the mounted tag.")
            if any(value is None for value in offsets):
                print(
                    "Before --execute, provide --marker-offset-x/y/z. "
                    "These are marker-center minus nozzle-tip offsets in Robot coordinates."
                )
            return 0

        if any(value is None for value in offsets):
            raise ArucoCalibrationError(
                "--execute requires --marker-offset-x, --marker-offset-y, and --marker-offset-z. "
                "A fixed marker offset changes the calibration translation; zero is valid only "
                "on axes where the marker center and nozzle tip are truly coincident."
            )

        marker_offset = np.asarray([float(value) for value in offsets], dtype=np.float64)
        print(
            "\nMarker center - nozzle tip Robot XYZ: "
            + ", ".join(f"{value:.3f}" for value in marker_offset.tolist())
            + " mm"
        )
        payload = run_calibration(
            grid=grid,
            marker_offset_robot_mm=marker_offset,
            robot_y_mm=float(args.robot_y),
            travel_speed_mm_s=float(args.travel_speed),
            settle_seconds=float(args.settle_seconds),
            marker_id=int(args.marker_id),
            marker_size_mm=float(args.marker_size_mm),
            depth_radius_px=int(args.depth_radius_px),
            max_plane_rms_mm=float(args.max_plane_rms_mm),
            acceptable_error_mm=float(args.acceptable_error_mm),
            output_path=Path(args.output),
        )

        metrics = payload["metrics"]
        transform = payload["transform"]
        print("\nCalibration result")
        print("------------------")
        print(f"Successful points: {metrics['successful_points']} / {len(grid)}")
        print(
            "Training RMS / max: "
            f"{metrics['training_rms_mm']:.3f} / {metrics['training_maximum_mm']:.3f} mm"
        )
        print(
            "LOO RMS / max:      "
            f"{metrics['leave_one_out_rms_mm']:.3f} / {metrics['leave_one_out_maximum_mm']:.3f} mm"
        )
        print(f"Camera Z span:      {metrics['camera_depth_span_mm']:.3f} mm")
        print(
            f"Quality gate:       {'PASS' if metrics['quality_pass'] else 'FAIL'} "
            f"(tolerance {metrics['acceptable_error_mm']:.2f} mm)"
        )
        rotation = transform["rotation_row_major"]
        print("Rotation row-major:")
        for row in range(3):
            values = rotation[row * 3 : row * 3 + 3]
            print("  " + "  ".join(f"{value:+.9f}" for value in values))
        print(
            "Translation mm: "
            + "  ".join(f"{value:+.4f}" for value in transform["translation_mm"])
        )
        print(f"Saved: {args.output}")
        if not metrics["quality_pass"]:
            print("Do not use this transform for physical targeting yet.")
            return 2
        return 0

    except (
        ArucoCalibrationError,
        CalibrationDepthError,
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
