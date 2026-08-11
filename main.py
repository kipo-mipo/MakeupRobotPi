from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from landmarks import (
    FaceLandmarkCapture,
    FaceNotFound,
    LandmarkCaptureError,
    LandmarkCaptureUnavailable,
)


MANNEQUIN_CALIBRATION_PURPOSE = "mannequin_depth_calibration"

landmark_capture = FaceLandmarkCapture()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    landmark_capture.close()


app = FastAPI(
    title="MakeupRobot Pi API",
    version="0.3.0",
    lifespan=lifespan,
)


class TestMessage(BaseModel):
    command: str
    message: str


class PiLandmarkCaptureRequest(BaseModel):
    request_id: UUID
    purpose: str
    return_image: bool = True
    minimum_confidence: float = Field(default=0.50, ge=0.0, le=1.0)


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
        "api_version": app.version,
        "face_landmark_capture": landmark_capture.status(),
    }


@app.post("/test")
def test_connection(data: TestMessage):
    print(f"Received from app: {data}")

    return {
        "status": "ok",
        "received_command": data.command,
        "received_message": data.message,
        "reply": "hello iPhone",
    }


@app.post("/calibration/face-landmarks/capture")
def capture_calibration_face_landmarks(data: PiLandmarkCaptureRequest):
    if data.purpose != MANNEQUIN_CALIBRATION_PURPOSE:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported capture purpose. Expected "
                f"'{MANNEQUIN_CALIBRATION_PURPOSE}'."
            ),
        )

    try:
        return landmark_capture.capture_for_calibration(
            request_id=str(data.request_id),
            return_image=data.return_image,
            minimum_confidence=data.minimum_confidence,
        )
    except FaceNotFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LandmarkCaptureUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LandmarkCaptureError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
