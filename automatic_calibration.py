from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AutomaticCalibrationError(RuntimeError):
    pass


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AutomaticCalibrationError(f"{label} file not found: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AutomaticCalibrationError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AutomaticCalibrationError(f"{label} has an unexpected structure.")
    return payload


def read_automatic_calibration_bundle(
    *,
    calibration_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    calibration = _read_json(calibration_path, "Automatic ArUco calibration")
    validation = _read_json(validation_path, "Automatic ArUco validation")

    calibration_pass = bool(calibration.get("metrics", {}).get("quality_pass"))
    validation_pass = bool(validation.get("metrics", {}).get("quality_pass"))

    calibration_created_at = calibration.get("created_at")
    validation_source_created_at = validation.get("source_calibration_created_at")
    validation_matches_calibration = (
        calibration_created_at is not None
        and validation_source_created_at is not None
        and calibration_created_at == validation_source_created_at
    )

    calibration_serial = calibration.get("camera", {}).get("serial_number")
    validation_serial = validation.get("camera_serial_number")
    camera_matches = (
        not calibration_serial
        or not validation_serial
        or calibration_serial == validation_serial
    )

    try:
        rotation = calibration["transform"]["rotation_row_major"]
        translation = calibration["transform"]["translation_mm"]
    except (KeyError, TypeError) as exc:
        raise AutomaticCalibrationError(
            "Automatic ArUco calibration is missing its rigid transform."
        ) from exc

    transform_valid = (
        isinstance(rotation, list)
        and len(rotation) == 9
        and isinstance(translation, list)
        and len(translation) == 3
    )

    ready = (
        calibration_pass
        and validation_pass
        and validation_matches_calibration
        and camera_matches
        and transform_valid
    )

    reasons: list[str] = []
    if not calibration_pass:
        reasons.append("calibration quality gate did not pass")
    if not validation_pass:
        reasons.append("independent validation quality gate did not pass")
    if not validation_matches_calibration:
        reasons.append("validation does not correspond to the current calibration")
    if not camera_matches:
        reasons.append("validation camera serial does not match calibration camera")
    if not transform_valid:
        reasons.append("rigid transform dimensions are invalid")

    return {
        "status": "ok",
        "source": "automatic_aruco",
        "ready_for_targeting": ready,
        "not_ready_reasons": reasons,
        "calibration": calibration,
        "validation": validation,
    }
