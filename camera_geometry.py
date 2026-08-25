from __future__ import annotations

import time
from typing import Any

import numpy as np

from gemini_camera import (
    CameraCaptureError,
    CameraUnavailableError,
    Config,
    OBFrameAggregateOutputMode,
    Pipeline,
    _require_dependencies,
    _safe_info_call,
    _select_color_profile,
    _select_depth_profile,
)


class CameraGeometryError(RuntimeError):
    pass


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraGeometryError(f"Camera calibration field {name} is invalid.") from exc
    if not np.isfinite(result):
        raise CameraGeometryError(f"Camera calibration field {name} is non-finite.")
    return result


def _serialize_intrinsic(intrinsic: Any) -> dict[str, Any]:
    payload = {
        "fx": _finite_float(intrinsic.fx, "fx"),
        "fy": _finite_float(intrinsic.fy, "fy"),
        "cx": _finite_float(intrinsic.cx, "cx"),
        "cy": _finite_float(intrinsic.cy, "cy"),
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
    }
    if payload["fx"] <= 0 or payload["fy"] <= 0:
        raise CameraGeometryError("Gemini intrinsics contain a non-positive focal length.")
    if payload["width"] <= 1 or payload["height"] <= 1:
        raise CameraGeometryError("Gemini intrinsics contain invalid image dimensions.")
    return payload


def _serialize_distortion(distortion: Any) -> dict[str, float]:
    return {
        key: _finite_float(getattr(distortion, key, 0.0), key)
        for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    }


def _try_serialize_extrinsic(extrinsic: Any) -> tuple[dict[str, list[float]] | None, str | None]:
    """Best-effort diagnostic only.

    Some pyorbbecsdk builds expose a broken or non-iterable OBExtrinsic Python
    binding. Rigid calibration does not depend on reading this object directly:
    AlignFilter already performs depth-to-color registration, and camera XYZ is
    deprojected in the RGB optical frame using RGB intrinsics/distortion.
    """
    try:
        rotation = np.asarray(extrinsic.rot, dtype=np.float64).reshape(-1).tolist()
        translation = np.asarray(extrinsic.transform, dtype=np.float64).reshape(-1).tolist()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if len(rotation) != 9 or len(translation) != 3:
        return None, (
            "Unexpected depth-to-color extrinsic shape: "
            f"rotation={len(rotation)}, translation={len(translation)}"
        )
    if not all(np.isfinite(value) for value in rotation + translation):
        return None, "Depth-to-color extrinsic contains non-finite values."

    return (
        {
            "rotation_row_major": [float(value) for value in rotation],
            "translation_mm": [float(value) for value in translation],
        },
        None,
    )


def read_active_camera_geometry(*, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Read calibration for the exact stream profiles used by capture_calibration()."""
    try:
        _require_dependencies()
    except CameraUnavailableError as exc:
        raise CameraGeometryError(str(exc)) from exc

    pipeline = Pipeline()
    config = Config()
    color_profile = _select_color_profile(pipeline)
    depth_profile = _select_depth_profile(pipeline)
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    try:
        pipeline.start(config)
        deadline = time.monotonic() + timeout_seconds
        frame_set = None
        while time.monotonic() < deadline:
            frame_set = pipeline.wait_for_frames(1000)
            if frame_set is not None:
                break

        if frame_set is None:
            raise CameraGeometryError("Timed out waiting for Gemini frames before reading camera intrinsics.")

        device_info = pipeline.get_device().get_device_info()
        camera_param = pipeline.get_camera_param()
        rgb_intrinsic = _serialize_intrinsic(camera_param.rgb_intrinsic)
        depth_intrinsic = _serialize_intrinsic(camera_param.depth_intrinsic)
        rgb_distortion = _serialize_distortion(camera_param.rgb_distortion)
        depth_distortion = _serialize_distortion(camera_param.depth_distortion)
        depth_to_color, depth_to_color_error = _try_serialize_extrinsic(camera_param.transform)

        return {
            "model": "pinhole_with_distortion",
            "device_name": _safe_info_call(device_info, "get_name"),
            "serial_number": _safe_info_call(device_info, "get_serial_number"),
            "rgb_intrinsic": rgb_intrinsic,
            "rgb_distortion": rgb_distortion,
            "depth_intrinsic": depth_intrinsic,
            "depth_distortion": depth_distortion,
            "depth_to_color": depth_to_color,
            "depth_to_color_error": depth_to_color_error,
            "depth_to_color_required_for_rigid_calibration": False,
        }
    except CameraGeometryError:
        raise
    except CameraCaptureError as exc:
        raise CameraGeometryError(str(exc)) from exc
    except Exception as exc:
        raise CameraGeometryError(f"Failed to read Gemini camera geometry: {exc}") from exc
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
