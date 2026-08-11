# MakeupRobotPi

Raspberry Pi service for MakeupRobot camera capture and app communication.

## Current architecture

The Pi owns camera capture and facial landmark detection. The iOS app owns the
camera-to-robot calibration transform.

The coordinate contract between the two repos is intentionally simple:

1. The Pi camera captures a raw **1280 x 720** sideways frame.
2. The Pi rotates that frame **90 degrees counterclockwise** only for face
   detection so the detector sees an upright face.
3. Detected points are transformed back into the original raw image coordinate
   system before the Pi returns them.
4. The API returns `u_px` and `v_px` in raw Pi-image pixels.
5. FaceCaptureIOS applies its saved camera-to-robot homography to convert those
   raw `(U, V)` pixels into robot `(X, Z)` millimeters.

Do not add a second U/V-to-X/Z transform on the Pi. That would create two
sources of truth for calibration.

## API

### `GET /status`

Reports server health and whether the face-landmark model/camera have been
initialized.

### `POST /test`

Existing connection-test endpoint for the iOS app.

### `POST /face/landmarks`

Captures one Pi camera frame, detects one face, and returns all detected face
landmarks in raw Pi-image pixel coordinates.

Example response shape:

```json
{
  "status": "ok",
  "face_count": 1,
  "landmark_count": 478,
  "raw_image": {
    "width_px": 1280,
    "height_px": 720,
    "orientation": "sideways"
  },
  "landmarks": [
    {
      "index": 0,
      "u_px": 640.2,
      "v_px": 356.7,
      "normalized_z": -0.03
    }
  ]
}
```

`normalized_z` is MediaPipe-relative depth. It is **not** robot millimeters and
must not be used as a physical motion command.

The endpoint returns:

- HTTP 200 when a face and landmarks are captured.
- HTTP 422 when the camera frame contains no detected face.
- HTTP 503 when the camera, MediaPipe package, or model is unavailable.
- HTTP 500 for another landmark-capture failure.

## Raspberry Pi setup

Run from the repository root on the Pi:

```bash
bash setup_pi.sh
```

The script:

- installs Raspberry Pi OS Picamera2 support,
- creates `.venv` with access to system Picamera2 packages,
- installs the Python API/landmark dependencies, and
- downloads the Face Landmarker model into `models/face_landmarker.task`.

Then start the server:

```bash
source .venv/bin/activate
python main.py
```

The FastAPI server listens on port `8000` on all interfaces.

To trigger a capture from another machine on the same network:

```bash
curl -X POST http://<pi-ip>:8000/face/landmarks
```

## Tests

The coordinate-frame tests do not require camera hardware:

```bash
python -m unittest discover -s tests
```

These tests are important because the app calibration uses raw sideways-image
coordinates. A landmark detector can appear visually correct while still
returning coordinates in the wrong rotated frame.

## Scope of this stage

This branch performs **camera capture and landmark reporting only**. It does
not command Klipper, motors, the airbrush, the compressor, or the solenoid.
Robot motion should only be connected after the Pi landmarks and the iOS
homography agree on known calibration points.
