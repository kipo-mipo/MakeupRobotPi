from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="MakeupRobot Pi API",
    version="0.1.0",
)


class TestMessage(BaseModel):
    command: str
    message: str


@app.get("/")
def root():
    return {
        "name": "MakeupRobot Pi API",
        "status": "running",
    }


@app.get("/status")
def get_status():
    return {
        "status": "ok",
        "robot": "MakeupRobot",
        "server_time": datetime.now(timezone.utc).isoformat(),
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )