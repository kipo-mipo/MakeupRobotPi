import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from calibration_depth import CalibrationDepthError, sample_capture_depth
from camera_geometry import CameraGeometryError, read_active_camera_geometry
from gemini_camera import (
    CAPTURE_DIR,
    CameraCaptureError,
    CameraUnavailableError,
    camera_status,
    capture_calibration,
)
from gemini_landmarks import (
    FACE_LANDMARKER,
    GeminiFaceNotFound,
    GeminiLandmarkError,
    GeminiLandmarkUnavailable,
)


app = FastAPI(
    title="MakeupRobot Pi API",
    version="0.5.0",
)

CONFIG_DIR = Path(__file__).resolve().parent / "config"
ACTIVE_CALIBRATION_PATH = CONFIG_DIR / "gemini_robot_calibration.json"


class TestMessage(BaseModel):
    command: str
    message: str


class DepthSamplePoint(BaseModel):
    id: str
    u_px: float
    v_px: float


class DepthSampleRequest(BaseModel):
    capture_id: str
    points: list[DepthSamplePoint]
    radius_px: int = 2


class LandmarkRequest(BaseModel):
    capture_id: str


class CalibrationProfileEnvelope(BaseModel):
    profile: dict[str, Any]


@app.get("/")
def root():
    return {
        "name": "MakeupRobot Pi API",
        "status": "running",
        "version": app.version,
    }


@app.get("/status")
def get_status():
    return {
        "status": "ok",
        "robot": "MakeupRobot",
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/camera/status")
def get_camera_status():
    return camera_status()


@app.get("/camera/geometry")
def get_camera_geometry():
    try:
        return {
            "status": "ok",
            "geometry": read_active_camera_geometry(),
        }
    except CameraGeometryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/calibration/landmarks/status")
def get_landmark_status():
    return {
        "status": "ok",
        "landmarker": FACE_LANDMARKER.status(),
    }


@app.post("/calibration/capture")
def create_calibration_capture():
    try:
        result = capture_calibration()
    except CameraUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CameraCaptureError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = result.as_dict()
    payload.update(
        {
            "status": "ok",
            "color_url": f"/captures/{result.color_filename}",
            "depth_url": f"/captures/{result.depth_filename}",
            "metadata_url": f"/captures/{result.metadata_filename}",
        }
    )
    return payload


@app.post("/calibration/landmarks")
def create_calibration_landmarks(data: LandmarkRequest):
    try:
        return FACE_LANDMARKER.detect_capture(data.capture_id)
    except GeminiLandmarkUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiFaceNotFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeminiLandmarkError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/calibration/depth-samples")
def create_depth_samples(data: DepthSampleRequest):
    try:
        return sample_capture_depth(
            data.capture_id,
            [
                {"id": point.id, "u_px": point.u_px, "v_px": point.v_px}
                for point in data.points
            ],
            radius_px=data.radius_px,
        )
    except CalibrationDepthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/calibration/profile")
def save_calibration_profile(data: CalibrationProfileEnvelope):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "profile": data.profile,
    }
    temporary_path = ACTIVE_CALIBRATION_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(ACTIVE_CALIBRATION_PATH)
    return {
        "status": "ok",
        "path": ACTIVE_CALIBRATION_PATH.name,
    }


@app.get("/calibration/profile")
def get_calibration_profile():
    if not ACTIVE_CALIBRATION_PATH.is_file():
        raise HTTPException(status_code=404, detail="No active Gemini-to-robot calibration is saved.")
    return json.loads(ACTIVE_CALIBRATION_PATH.read_text(encoding="utf-8"))


@app.get("/captures/{filename}")
def get_capture_file(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid capture filename.")

    file_path = CAPTURE_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Capture file not found.")

    return FileResponse(file_path)


@app.post("/test")
def test_connection(data: TestMessage):
    print(f"Received from app: {data}")

    return {
        "status": "ok",
        "received_command": data.command,
        "received_message": data.message,
        "reply": "hello iPhone",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
