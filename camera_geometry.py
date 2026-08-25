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
        raise CameraGeometryError("Gemini RGB intrinsics contain a non-positive focal length.")
    if payload["width"] <= 1 or payload["height"] <= 1:
        raise CameraGeometryError("Gemini RGB intrinsics contain invalid image dimensions.")
    return payload


def _serialize_distortion(distortion: Any) -> dict[str, float]:
    return {
        key: _finite_float(getattr(distortion, key, 0.0), key)
        for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    }


def _serialize_extrinsic(extrinsic: Any) -> dict[str, list[float]]:
    try:
        rotation = [float(value) for value in extrinsic.rot]
        translation = [float(value) for value in extrinsic.transform]
    except Exception as exc:
        raise CameraGeometryError("Gemini depth-to-color extrinsic could not be read.") from exc

    if len(rotation) != 9 or len(translation) != 3:
        raise CameraGeometryError("Gemini depth-to-color extrinsic has an unexpected shape.")
    if not all(np.isfinite(value) for value in rotation + translation):
        raise CameraGeometryError("Gemini depth-to-color extrinsic contains non-finite values.")

    return {
        "rotation_row_major": rotation,
        "translation_mm": translation,
    }


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
        depth_to_color = _serialize_extrinsic(camera_param.transform)

        return {
            "model": "pinhole_with_distortion",
            "device_name": _safe_info_call(device_info, "get_name"),
            "serial_number": _safe_info_call(device_info, "get_serial_number"),
            "rgb_intrinsic": rgb_intrinsic,
            "rgb_distortion": rgb_distortion,
            "depth_intrinsic": depth_intrinsic,
            "depth_distortion": depth_distortion,
            "depth_to_color": depth_to_color,
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
