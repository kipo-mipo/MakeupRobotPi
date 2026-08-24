# MakeupRobotPi

Raspberry Pi API for the MakeupRobot prototype.

## One-stage Gemini 215 calibration

The active calibration workflow uses the Gemini RGB frame and its software-aligned depth frame directly. There is no flat-board calibration stage.

The iPhone detects facial landmarks in the Gemini RGB image, sends those exact raw U/V pixels back to the Pi for aligned depth sampling, then solves one Gemini `(U, V, depth)` → Robot `(X, Y, Z)` transform from measured robot ground truth.

### Install

Use a 64-bit Linux install and Python 3.9+ if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux, Orbbec requires a one-time device-permission setup. After installing `pyorbbecsdk2`, find its installed location and run its environment setup script:

```bash
python -c "import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))"
```

Then run `shared/setup_env.py` from that directory with the permissions it requests. Replug the camera afterward if needed.

### Check camera discovery

```bash
python scripts/check_camera.py
```

Or, once the API is running:

```bash
curl http://127.0.0.1:8000/camera/status
```

### Run the API

```bash
python main.py
```

### 1. Capture RGB + aligned depth

```bash
curl -X POST http://127.0.0.1:8000/calibration/capture
```

A successful capture returns URLs for:

- the RGB calibration image (`*_color.png`)
- the aligned 16-bit raw depth image (`*_depth_raw.png`)
- capture metadata (`*_metadata.json`) including the depth scale in millimeters per raw unit

Files are written under `captures/`.

### 2. Sample aligned depth at landmark pixels

The app sends the detected raw Gemini U/V coordinates back to the same capture:

```bash
curl -X POST http://127.0.0.1:8000/calibration/depth-samples \
  -H 'Content-Type: application/json' \
  -d '{
    "capture_id": "CAPTURE_ID",
    "radius_px": 2,
    "points": [
      {"id": "nose_tip", "u_px": 960, "v_px": 540}
    ]
  }'
```

Depth is sampled from a small neighborhood in the aligned 16-bit frame. Zero/invalid pixels are discarded and the median valid raw value is converted to millimeters using that capture's recorded depth scale.

### 3. Save the solved Gemini-to-robot profile

FaceCapture solves the 3D mapping and uploads the active profile:

```text
POST /calibration/profile
GET  /calibration/profile
```

The Pi stores the active profile in `config/gemini_robot_calibration.json`.

### Calibration model

For each correspondence the app forms a pinhole-compatible feature vector from raw Gemini pixels and optical depth:

```text
[(u_normalized - 0.5) * depth,
 (v_normalized - 0.5) * depth,
 depth,
 1]
```

A least-squares affine transform maps that camera-space feature vector to Robot X/Y/Z. This absorbs the camera intrinsics and fixed camera-to-robot pose without requiring a separate flat-board homography.

Use at least six measured facial landmarks; all eight provided by the app are preferred. The current prototype requires the camera mount to remain fixed after calibration.

### First Gemini 215 bring-up

1. Connect the Gemini directly to a USB 3 port.
2. Run `python scripts/check_camera.py` and confirm the reported device name/serial.
3. Start the API and check `/camera/status`.
4. Call `POST /calibration/capture`.
5. Open the returned color image and verify orientation/framing.
6. Inspect the metadata and confirm depth dimensions match the color dimensions.
7. Call `/calibration/depth-samples` on a known face pixel and confirm a plausible nonzero millimeter depth.
8. Complete the one-stage calibration in FaceCapture and confirm `GET /calibration/profile` returns the uploaded profile.
