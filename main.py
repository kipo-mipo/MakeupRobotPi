from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from landmarks import (
    FaceLandmarkCapture,
    FaceNotFound,
    LandmarkCaptureError,
    LandmarkCaptureUnavailable,
)


landmark_capture = FaceLandmarkCapture()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    landmark_capture.close()


app = FastAPI(
    title="MakeupRobot Pi API",
    version="0.2.0",
    lifespan=lifespan,
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


@app.post("/face/landmarks")
def capture_face_landmarks():
    try:
        return landmark_capture.capture()
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
