from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gemini_camera import (
    CAPTURE_DIR,
    CameraCaptureError,
    CameraUnavailableError,
    camera_status,
    capture_calibration,
)


app = FastAPI(
    title="MakeupRobot Pi API",
    version="0.2.0",
)


class TestMessage(BaseModel):
    command: str
    message: str


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
