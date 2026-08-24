# MakeupRobotPi

Raspberry Pi API for the MakeupRobot prototype.

## Gemini 215 calibration capture

The API is structured so the FastAPI server can still start when the Orbbec SDK or camera is missing. Camera readiness is reported separately.

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

### Capture calibration RGB + aligned depth

```bash
curl -X POST http://127.0.0.1:8000/calibration/capture
```

A successful capture returns URLs for:

- the RGB calibration image (`*_color.png`)
- the aligned 16-bit raw depth image (`*_depth_raw.png`)
- capture metadata (`*_metadata.json`) including the depth scale in millimeters per raw unit

Files are written under `captures/`.

### First Gemini 215 bring-up

1. Connect the Gemini directly to a USB 3 port.
2. Run `python scripts/check_camera.py` and confirm the reported device name/serial.
3. Start the API and check `/camera/status`.
4. Call `POST /calibration/capture`.
5. Open the returned color image and verify orientation/framing.
6. Inspect the metadata and confirm depth dimensions match the color dimensions.
7. If the default stream profiles are poor for the face-calibration geometry, lock explicit Gemini 215 resolution/FPS profiles after enumerating what the physical unit supports.
