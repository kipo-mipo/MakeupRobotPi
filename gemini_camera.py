from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import pyorbbecsdk
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
except ImportError:
    pyorbbecsdk = None
    AlignFilter = None
    Config = None
    Context = None
    OBFormat = None
    OBFrameAggregateOutputMode = None
    OBSensorType = None
    OBStreamType = None
    Pipeline = None

CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
_CAPTURE_LOCK = threading.Lock()


class CameraUnavailableError(RuntimeError):
    pass


class CameraCaptureError(RuntimeError):
    pass


@dataclass
class CaptureResult:
    capture_id: str
    color_filename: str
    depth_filename: str
    metadata_filename: str
    width: int
    height: int
    depth_scale_mm: float
    device_name: str | None
    serial_number: str | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _sdk_version() -> str | None:
    if pyorbbecsdk is None:
        return None
    getter = getattr(pyorbbecsdk, "get_version", None)
    if getter is None:
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _safe_info_call(info: Any, method_name: str) -> Any | None:
    method = getattr(info, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def camera_status() -> dict[str, Any]:
    dependencies = {
        "pyorbbecsdk": pyorbbecsdk is not None,
        "numpy": np is not None,
        "opencv": cv2 is not None,
    }
    status: dict[str, Any] = {
        "ready": False,
        "sdk_version": _sdk_version(),
        "dependencies": dependencies,
        "device_count": 0,
        "devices": [],
        "error": None,
    }

    if not all(dependencies.values()):
        missing = [name for name, installed in dependencies.items() if not installed]
        status["error"] = f"Missing camera dependencies: {', '.join(missing)}"
        return status

    try:
        context = Context()
        device_list = context.query_devices()
        status["device_count"] = device_list.get_count()
        devices: list[dict[str, Any]] = []

        for index in range(status["device_count"]):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            devices.append(
                {
                    "index": index,
                    "name": _safe_info_call(info, "get_name"),
                    "serial_number": _safe_info_call(info, "get_serial_number"),
                    "firmware_version": _safe_info_call(info, "get_firmware_version"),
                    "hardware_version": _safe_info_call(info, "get_hardware_version"),
                    "uid": _safe_info_call(info, "get_uid"),
                    "vid": _safe_info_call(info, "get_vid"),
                    "pid": _safe_info_call(info, "get_pid"),
                }
            )

        status["devices"] = devices
        status["ready"] = status["device_count"] > 0
        if not status["ready"]:
            status["error"] = "Orbbec SDK is installed, but no Orbbec camera is connected."
        return status
    except Exception as exc:
        status["error"] = f"Orbbec device discovery failed: {exc}"
        return status


def _require_dependencies() -> None:
    missing: list[str] = []
    if pyorbbecsdk is None:
        missing.append("pyorbbecsdk2")
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python-headless")
    if missing:
        raise CameraUnavailableError("Missing camera dependencies: " + ", ".join(missing))


def _select_color_profile(pipeline: Any) -> Any:
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        return profiles.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    except Exception:
        return profiles.get_default_video_stream_profile()


def _select_depth_profile(pipeline: Any) -> Any:
    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    return profiles.get_default_video_stream_profile()


def _frame_to_bgr(frame: Any) -> Any:
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)

    bgr_format = getattr(OBFormat, "BGR", None)
    if bgr_format is not None and fmt == bgr_format:
        return data.reshape((height, width, 3)).copy()

    mjpg_format = getattr(OBFormat, "MJPG", None)
    if mjpg_format is not None and fmt == mjpg_format:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    yuyv_format = getattr(OBFormat, "YUYV", None)
    if yuyv_format is not None and fmt == yuyv_format:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

    uyvy_format = getattr(OBFormat, "UYVY", None)
    if uyvy_format is not None and fmt == uyvy_format:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)

    raise CameraCaptureError(f"Unsupported color frame format: {fmt}")


def _as_frameset(frame: Any) -> Any:
    converter = getattr(frame, "as_frame_set", None)
    if converter is None:
        return frame
    try:
        return converter()
    except Exception:
        return frame


def capture_calibration(
    *,
    capture_dir: Path = CAPTURE_DIR,
    timeout_seconds: float = 10.0,
    warmup_frames: int = 5,
) -> CaptureResult:
    _require_dependencies()

    with _CAPTURE_LOCK:
        context = Context()
        if context.query_devices().get_count() == 0:
            raise CameraUnavailableError("No Orbbec camera is connected.")

        pipeline = Pipeline()
        config = Config()
        color_profile = _select_color_profile(pipeline)
        depth_profile = _select_depth_profile(pipeline)
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        try:
            pipeline.enable_frame_sync()
        except Exception:
            pass

        try:
            pipeline.start(config)
            device_info = pipeline.get_device().get_device_info()
            device_name = _safe_info_call(device_info, "get_name")
            serial_number = _safe_info_call(device_info, "get_serial_number")

            deadline = time.monotonic() + timeout_seconds
            accepted_frames = 0
            color_frame = None
            depth_frame = None

            while time.monotonic() < deadline:
                frames = pipeline.wait_for_frames(1000)
                if frames is None:
                    continue

                aligned = align_filter.process(frames)
                if aligned is None:
                    continue
                aligned = _as_frameset(aligned)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue

                accepted_frames += 1
                if accepted_frames > warmup_frames:
                    break

            if color_frame is None or depth_frame is None:
                raise CameraCaptureError(
                    f"Timed out after {timeout_seconds:.1f}s waiting for aligned color/depth frames."
                )

            color_bgr = _frame_to_bgr(color_frame)
            if color_bgr is None:
                raise CameraCaptureError("Failed to decode the Gemini color frame.")

            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            try:
                depth_raw = depth_raw.reshape((depth_height, depth_width))
            except ValueError as exc:
                raise CameraCaptureError(
                    "Depth frame byte count did not match its reported dimensions."
                ) from exc

            color_height, color_width = color_bgr.shape[:2]
            if (depth_width, depth_height) != (color_width, color_height):
                raise CameraCaptureError(
                    "Depth-to-color alignment returned mismatched dimensions: "
                    f"color={color_width}x{color_height}, depth={depth_width}x{depth_height}."
                )

            depth_scale_mm = float(depth_frame.get_depth_scale())
            timestamp = datetime.now(timezone.utc)
            capture_id = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
            capture_dir.mkdir(parents=True, exist_ok=True)

            color_filename = f"{capture_id}_color.png"
            depth_filename = f"{capture_id}_depth_raw.png"
            metadata_filename = f"{capture_id}_metadata.json"
            color_path = capture_dir / color_filename
            depth_path = capture_dir / depth_filename
            metadata_path = capture_dir / metadata_filename

            if not cv2.imwrite(str(color_path), color_bgr):
                raise CameraCaptureError(f"Failed to save color image to {color_path}")
            if not cv2.imwrite(str(depth_path), depth_raw):
                raise CameraCaptureError(f"Failed to save depth image to {depth_path}")

            valid_depth = depth_raw[depth_raw > 0]
            metadata = {
                "capture_id": capture_id,
                "captured_at": timestamp.isoformat(),
                "device": {"name": device_name, "serial_number": serial_number},
                "color": {
                    "filename": color_filename,
                    "width": color_width,
                    "height": color_height,
                    "format": str(color_frame.get_format()),
                },
                "depth": {
                    "filename": depth_filename,
                    "width": depth_width,
                    "height": depth_height,
                    "format": str(depth_frame.get_format()),
                    "scale_mm_per_unit": depth_scale_mm,
                    "min_valid_mm": float(valid_depth.min()) * depth_scale_mm if valid_depth.size else None,
                    "max_valid_mm": float(valid_depth.max()) * depth_scale_mm if valid_depth.size else None,
                },
                "alignment": "depth_to_color_software",
                "warmup_frames": warmup_frames,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            return CaptureResult(
                capture_id=capture_id,
                color_filename=color_filename,
                depth_filename=depth_filename,
                metadata_filename=metadata_filename,
                width=color_width,
                height=color_height,
                depth_scale_mm=depth_scale_mm,
                device_name=device_name,
                serial_number=serial_number,
            )
        except (CameraUnavailableError, CameraCaptureError):
            raise
        except Exception as exc:
            raise CameraCaptureError(f"Gemini calibration capture failed: {exc}") from exc
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
