from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from landmarks import (
    CALIBRATION_LANDMARKS,
    DEFAULT_MODEL_PATH,
    MODEL_MINIMUM_CONFIDENCE,
)


RAW_WIDTH = 1280
RAW_HEIGHT = 720
DEFAULT_FPS = 5
DEFAULT_PORT = 8081
MJPEG_BOUNDARY = "frame"


class LatestFrame:
    """Thread-safe store that keeps only the newest annotated JPEG."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._stopped = False

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def wait_for_next(
        self,
        previous_sequence: int,
        timeout: float = 5.0,
    ) -> tuple[bytes | None, int, bool]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != previous_sequence or self._stopped,
                timeout=timeout,
            )
            return self._jpeg, self._sequence, self._stopped


def iter_mjpeg_frames(stream: BinaryIO):
    """Yield JPEG images from rpicam-vid's concatenated MJPEG stdout."""

    buffer = bytearray()
    soi = b"\xff\xd8"
    eoi = b"\xff\xd9"

    while True:
        read1 = getattr(stream, "read1", None)
        chunk = read1(65536) if read1 is not None else stream.read(65536)
        if not chunk:
            return

        buffer.extend(chunk)

        while True:
            start = buffer.find(soi)
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break

            end = buffer.find(eoi, start + 2)
            if end < 0:
                if start > 0:
                    del buffer[:start]
                break

            end += 2
            yield bytes(buffer[start:end])
            del buffer[:end]


def load_label_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), 18)
            except OSError:
                pass
    return ImageFont.load_default()


def encode_jpeg(image: Image.Image, *, quality: int = 88) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=False)
    return output.getvalue()


def make_status_frame(message: str) -> bytes:
    image = Image.new("RGB", (RAW_HEIGHT, RAW_WIDTH), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    font = load_label_font()
    draw.text(
        (30, 30),
        message,
        fill=(255, 220, 80),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return encode_jpeg(image)


class LandmarkStreamProducer(threading.Thread):
    """Capture continuous MJPEG, detect the face, and publish annotated frames."""

    def __init__(
        self,
        frame_store: LatestFrame,
        *,
        fps: int,
        model_path: Path,
    ) -> None:
        super().__init__(name="landmark-stream-producer", daemon=True)
        self.frame_store = frame_store
        self.fps = fps
        self.model_path = model_path
        self._stop_event = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._landmarker = None
        self._font = load_label_font()

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self.frame_store.stop()

    def run(self) -> None:
        try:
            self._run_stream()
        except Exception as exc:
            self.frame_store.publish(make_status_frame(f"Stream error: {exc}"))
            print(f"Landmark stream error: {exc}", flush=True)
        finally:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                except Exception:
                    pass
                self._landmarker = None

            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._process = None
            self.frame_store.stop()

    def _run_stream(self) -> None:
        command = shutil.which("rpicam-vid")
        if command is None:
            raise RuntimeError("rpicam-vid is not installed or not on PATH")
        if not self.model_path.is_file():
            raise RuntimeError(f"Face Landmarker model not found: {self.model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=MODEL_MINIMUM_CONFIDENCE,
            min_face_presence_confidence=MODEL_MINIMUM_CONFIDENCE,
            min_tracking_confidence=MODEL_MINIMUM_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        args = [
            command,
            "--timeout",
            "0",
            "--nopreview",
            "--width",
            str(RAW_WIDTH),
            "--height",
            str(RAW_HEIGHT),
            "--framerate",
            str(self.fps),
            "--codec",
            "mjpeg",
            "--quality",
            "85",
            "--autofocus-mode",
            "continuous",
            "--verbose",
            "0",
            "--output",
            "-",
        ]

        print(
            f"Starting camera at {RAW_WIDTH}x{RAW_HEIGHT}, target {self.fps} fps...",
            flush=True,
        )
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdout is None:
            raise RuntimeError("rpicam-vid stdout pipe was not created")

        processed = 0
        started = time.monotonic()

        for raw_jpeg in iter_mjpeg_frames(self._process.stdout):
            if self._stop_event.is_set():
                break

            annotated = self._annotate_frame(raw_jpeg)
            self.frame_store.publish(annotated)
            processed += 1

            if processed == 1:
                print("First annotated frame ready.", flush=True)
            elif processed % 25 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"Processed {processed} frames ({processed / elapsed:.1f} fps average)",
                    flush=True,
                )

        return_code = self._process.poll()
        if not self._stop_event.is_set() and return_code not in (None, 0):
            stderr = b""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"rpicam-vid exited with code {return_code}"
                + (f": {detail}" if detail else "")
            )

    def _annotate_frame(self, raw_jpeg: bytes) -> bytes:
        with Image.open(io.BytesIO(raw_jpeg)) as raw_image:
            raw_rgb = np.asarray(raw_image.convert("RGB"), dtype=np.uint8)

        if raw_rgb.shape != (RAW_HEIGHT, RAW_WIDTH, 3):
            raise RuntimeError(
                f"unexpected camera frame shape {raw_rgb.shape}; "
                f"expected {(RAW_HEIGHT, RAW_WIDTH, 3)}"
            )

        # Match the iOS calibration display: raw sideways camera frame -> 90 deg CCW.
        upright_rgb = np.ascontiguousarray(np.rot90(raw_rgb, k=1))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=upright_rgb)
        result = self._landmarker.detect(mp_image)

        upright = Image.fromarray(upright_rgb, mode="RGB")
        draw = ImageDraw.Draw(upright)

        if not result.face_landmarks:
            draw.text(
                (18, 18),
                "NO FACE DETECTED",
                fill=(255, 80, 80),
                font=self._font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            return encode_jpeg(upright)

        face = result.face_landmarks[0]
        width, height = upright.size

        for landmark_id, source_index in CALIBRATION_LANDMARKS:
            if source_index >= len(face):
                continue
            landmark = face[source_index]
            x = float(landmark.x) * (width - 1)
            y = float(landmark.y) * (height - 1)
            radius = 7

            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 70, 70),
                outline=(255, 255, 255),
                width=3,
            )
            draw.text(
                (x + 11, y - 11),
                landmark_id,
                fill=(255, 235, 70),
                font=self._font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        draw.text(
            (18, 18),
            "MakeupRobot semantic landmarks - upright preview",
            fill=(255, 255, 255),
            font=self._font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        return encode_jpeg(upright)


class LandmarkMJPEGHandler(BaseHTTPRequestHandler):
    frame_store: LatestFrame

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = (
                "MakeupRobot landmark debug stream\n"
                "Open /landmarks.mjpg in VLC or another MJPEG client.\n"
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path != "/landmarks.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        sequence = 0
        try:
            while True:
                jpeg, sequence, stopped = self.frame_store.wait_for_next(sequence)
                if jpeg is None:
                    if stopped:
                        return
                    continue

                self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                if stopped:
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args) -> None:
        print(f"VLC client: {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve an upright live mannequin landmark overlay as MJPEG for VLC."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="MediaPipe face_landmarker.task path.",
    )
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")

    frame_store = LatestFrame()
    LandmarkMJPEGHandler.frame_store = frame_store
    producer = LandmarkStreamProducer(
        frame_store,
        fps=args.fps,
        model_path=args.model,
    )
    server = ThreadingHTTPServer((args.host, args.port), LandmarkMJPEGHandler)

    print(
        f"Landmark stream URL: http://<PI-IP>:{args.port}/landmarks.mjpg",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)

    producer.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping landmark stream...", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        producer.stop()
        producer.join(timeout=3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
