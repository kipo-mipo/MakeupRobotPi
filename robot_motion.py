from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_MOONRAKER_URL = "http://127.0.0.1:7125"
DEFAULT_SAFE_Y_MM = 0.0
DEFAULT_RETRACT_FEED_MM_MIN = 900.0
DEFAULT_TRAVEL_FEED_MM_MIN = 1800.0
DEFAULT_APPROACH_FEED_MM_MIN = 600.0


class RobotMotionError(RuntimeError):
    pass


class RobotMotionUnavailable(RobotMotionError):
    pass


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RobotMotionError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise RobotMotionError(f"{name} must be a finite number.")
    return result


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return _finite(raw, name)


def moonraker_base_url() -> str:
    return os.getenv("MOONRAKER_URL", DEFAULT_MOONRAKER_URL).rstrip("/")


def safe_y_mm() -> float:
    return _env_float("ROBOT_TEST_SAFE_Y_MM", DEFAULT_SAFE_Y_MM)


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    url = moonraker_base_url() + path
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RobotMotionUnavailable(
            f"Moonraker returned HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RobotMotionUnavailable(
            f"Could not reach Moonraker at {moonraker_base_url()}: {exc}"
        ) from exc

    try:
        decoded = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RobotMotionUnavailable("Moonraker returned invalid JSON.") from exc

    if isinstance(decoded, dict) and "error" in decoded:
        raise RobotMotionUnavailable(f"Moonraker error: {decoded['error']}")
    if not isinstance(decoded, dict):
        raise RobotMotionUnavailable("Moonraker returned an unexpected response.")
    return decoded


def _result(payload: dict[str, Any]) -> Any:
    if "result" not in payload:
        raise RobotMotionUnavailable("Moonraker response is missing result.")
    return payload["result"]


def robot_status() -> dict[str, Any]:
    try:
        printer_info = _result(_request_json("GET", "/printer/info"))
        query_result = _result(
            _request_json(
                "POST",
                "/printer/objects/query",
                {
                    "objects": {
                        "toolhead": [
                            "homed_axes",
                            "position",
                            "axis_minimum",
                            "axis_maximum",
                        ],
                        "gcode_move": ["absolute_coordinates", "position"],
                        "print_stats": ["state"],
                    }
                },
            )
        )
    except RobotMotionUnavailable as exc:
        return {
            "status": "unavailable",
            "ready": False,
            "moonraker_url": moonraker_base_url(),
            "safe_y_mm": safe_y_mm(),
            "error": str(exc),
        }

    if not isinstance(printer_info, dict) or not isinstance(query_result, dict):
        raise RobotMotionUnavailable("Moonraker returned malformed printer status.")

    status = query_result.get("status") or {}
    toolhead = status.get("toolhead") or {}
    gcode_move = status.get("gcode_move") or {}
    print_stats = status.get("print_stats") or {}

    homed_axes = str(toolhead.get("homed_axes") or "")
    printer_state = str(printer_info.get("state") or "")
    print_state_raw = print_stats.get("state")
    print_state = str(print_state_raw) if print_state_raw is not None else None
    all_axes_homed = all(axis in homed_axes.lower() for axis in ("x", "y", "z"))
    print_idle = print_state is None or print_state.lower() not in {
        "printing",
        "paused",
    }
    ready = printer_state.lower() == "ready" and all_axes_homed and print_idle

    return {
        "status": "ok",
        "ready": ready,
        "moonraker_url": moonraker_base_url(),
        "printer_state": printer_state,
        "print_state": print_state,
        "homed_axes": homed_axes,
        "all_axes_homed": all_axes_homed,
        "print_idle": print_idle,
        "position": toolhead.get("position"),
        "gcode_position": gcode_move.get("position"),
        "absolute_coordinates": gcode_move.get("absolute_coordinates"),
        "axis_minimum": toolhead.get("axis_minimum"),
        "axis_maximum": toolhead.get("axis_maximum"),
        "safe_y_mm": safe_y_mm(),
        "coordinate_assumption": "Robot X/Y/Z are Klipper X/Y/Z in millimeters.",
        "error": None if ready else _not_ready_reason(
            printer_state=printer_state,
            homed_axes=homed_axes,
            print_state=print_state,
        ),
    }


def _not_ready_reason(
    *,
    printer_state: str,
    homed_axes: str,
    print_state: str | None,
) -> str:
    if printer_state.lower() != "ready":
        return f"Klipper is not ready (state={printer_state or 'unknown'})."
    missing = [axis.upper() for axis in ("x", "y", "z") if axis not in homed_axes.lower()]
    if missing:
        return "Home XYZ before a calibration test move; missing " + ", ".join(missing) + "."
    if print_state and print_state.lower() in {"printing", "paused"}:
        return f"Calibration test motion is blocked while print state is {print_state}."
    return "Robot motion is not ready."


def _axis_limits(status: dict[str, Any]) -> tuple[list[float], list[float]]:
    axis_minimum = status.get("axis_minimum")
    axis_maximum = status.get("axis_maximum")
    if (
        not isinstance(axis_minimum, list)
        or not isinstance(axis_maximum, list)
        or len(axis_minimum) < 3
        or len(axis_maximum) < 3
    ):
        raise RobotMotionError("Klipper did not report XYZ axis limits.")

    minimum = [_finite(axis_minimum[i], f"axis_minimum[{i}]") for i in range(3)]
    maximum = [_finite(axis_maximum[i], f"axis_maximum[{i}]") for i in range(3)]
    return minimum, maximum


def build_test_move_plan(
    *,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    axis_minimum: list[float],
    axis_maximum: list[float],
    safe_y: float | None = None,
    retract_feed_mm_min: float | None = None,
    travel_feed_mm_min: float | None = None,
    approach_feed_mm_min: float | None = None,
) -> dict[str, Any]:
    target = [
        _finite(x_mm, "Robot X"),
        _finite(y_mm, "Robot Y"),
        _finite(z_mm, "Robot Z"),
    ]
    if len(axis_minimum) < 3 or len(axis_maximum) < 3:
        raise RobotMotionError("XYZ axis limits are required.")
    minimum = [_finite(axis_minimum[i], f"axis_minimum[{i}]") for i in range(3)]
    maximum = [_finite(axis_maximum[i], f"axis_maximum[{i}]") for i in range(3)]

    labels = ("X", "Y", "Z")
    for index, value in enumerate(target):
        if value < minimum[index] or value > maximum[index]:
            raise RobotMotionError(
                f"Robot {labels[index]}={value:.3f} mm is outside Klipper limits "
                f"[{minimum[index]:.3f}, {maximum[index]:.3f}]."
            )

    safe = safe_y_mm() if safe_y is None else _finite(safe_y, "safe Y")
    if safe < minimum[1] or safe > maximum[1]:
        raise RobotMotionError(
            f"Safe Robot Y={safe:.3f} mm is outside Klipper Y limits "
            f"[{minimum[1]:.3f}, {maximum[1]:.3f}]."
        )
    if target[1] < safe:
        raise RobotMotionError(
            f"Target Robot Y={target[1]:.3f} mm is behind safe Y={safe:.3f} mm. "
            "Calibration test moves require +Y to approach the face."
        )

    retract_feed = (
        _env_float("ROBOT_TEST_RETRACT_FEED_MM_MIN", DEFAULT_RETRACT_FEED_MM_MIN)
        if retract_feed_mm_min is None
        else _finite(retract_feed_mm_min, "retract feed")
    )
    travel_feed = (
        _env_float("ROBOT_TEST_TRAVEL_FEED_MM_MIN", DEFAULT_TRAVEL_FEED_MM_MIN)
        if travel_feed_mm_min is None
        else _finite(travel_feed_mm_min, "travel feed")
    )
    approach_feed = (
        _env_float("ROBOT_TEST_APPROACH_FEED_MM_MIN", DEFAULT_APPROACH_FEED_MM_MIN)
        if approach_feed_mm_min is None
        else _finite(approach_feed_mm_min, "approach feed")
    )
    if min(retract_feed, travel_feed, approach_feed) <= 0:
        raise RobotMotionError("Calibration test feed rates must be positive.")

    commands = [
        "G90",
        f"G0 Y{safe:.3f} F{retract_feed:.0f}",
        f"G0 X{target[0]:.3f} Z{target[2]:.3f} F{travel_feed:.0f}",
        f"G0 Y{target[1]:.3f} F{approach_feed:.0f}",
    ]
    return {
        "target": {"x_mm": target[0], "y_mm": target[1], "z_mm": target[2]},
        "safe_y_mm": safe,
        "feed_mm_min": {
            "retract": retract_feed,
            "travel_xz": travel_feed,
            "approach_y": approach_feed,
        },
        "commands": commands,
        "script": "\n".join(commands),
    }


def test_move(
    *,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    execute: bool,
) -> dict[str, Any]:
    status = robot_status()
    if status.get("status") != "ok":
        raise RobotMotionUnavailable(status.get("error") or "Robot status is unavailable.")
    if not status.get("ready"):
        raise RobotMotionError(status.get("error") or "Robot is not ready for motion.")

    axis_minimum, axis_maximum = _axis_limits(status)
    plan = build_test_move_plan(
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        axis_minimum=axis_minimum,
        axis_maximum=axis_maximum,
    )

    if not execute:
        return {
            "status": "preview",
            "executed": False,
            "robot": status,
            "plan": plan,
        }

    response = _result(
        _request_json(
            "POST",
            "/printer/gcode/script",
            {"script": plan["script"]},
            timeout_seconds=20.0,
        )
    )
    return {
        "status": "ok",
        "executed": True,
        "robot": status,
        "plan": plan,
        "moonraker_result": response,
    }


def emergency_stop() -> dict[str, Any]:
    response = _result(
        _request_json(
            "POST",
            "/printer/emergency_stop",
            {},
            timeout_seconds=4.0,
        )
    )
    return {
        "status": "ok",
        "stopped": True,
        "moonraker_result": response,
    }
