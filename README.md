# MakeupRobotPi

Raspberry Pi API for the MakeupRobot prototype.

## One-stage Gemini 215 rigid calibration

The active calibration is a physical 3D camera-to-robot calibration. There is no flat-board stage and no free affine camera fit.

For each facial landmark:

1. FaceCapture detects raw Gemini RGB `U/V`.
2. The Pi samples the matching software-aligned depth frame.
3. The Pi reads the Gemini RGB intrinsics and distortion calibration from the active Orbbec stream profiles.
4. The raw pixel is undistorted and deprojected into Gemini optical camera coordinates in millimeters: `Camera X/Y/Z`.
5. FaceCapture pairs that camera-space point with measured `Robot X/Y/Z`.
6. FaceCapture solves one rigid transform only: a 3×3 rotation plus a 3D translation.

The solver is not allowed to stretch, shear, or independently scale axes. That prevents a low-residual affine fit from hiding an incorrect physical mapping.

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

### Run the API

```bash
python main.py
```

The rigid-calibration API reports version `0.4.0`.

### Check camera discovery

```bash
curl http://127.0.0.1:8000/camera/status
```

### Check Gemini intrinsics

```bash
curl http://127.0.0.1:8000/camera/geometry
```

A valid response must contain nonzero RGB focal lengths `fx/fy`, principal point `cx/cy`, RGB distortion coefficients, and the physical camera serial number. The Orbbec SDK calibration is read after the active color/depth streams have produced frames.

### 1. Capture RGB + aligned depth

```bash
curl -X POST http://127.0.0.1:8000/calibration/capture
```

A successful capture returns URLs for:

- the native Gemini RGB calibration image
- the software depth-to-color aligned 16-bit raw depth image
- capture metadata with depth scale and device identity

Files are written under `captures/`.

### 2. Sample depth and physical camera XYZ

The app sends landmark pixels back to the exact capture:

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

Each valid sample returns:

```text
depth_mm
camera_x_mm
camera_y_mm
camera_z_mm
```

Camera coordinates use the RGB optical frame:

```text
+X = image right
+Y = image down
+Z = forward optical depth
```

All units are millimeters.

The normal depth estimator uses a 5×5 median. If that window contains no depth, the existing face-consistent fallback may search slightly farther while rejecting values inconsistent with the other facial depths.

The first rigid depth-sample request also records the camera calibration in that capture's metadata so the camera model used for deprojection is auditable later.

### 3. Rigid camera-to-robot transform

FaceCapture solves:

```text
P_robot = R * P_camera + t
```

where:

- `R` is a proper 3D rotation matrix
- `t` is a three-element translation in millimeters
- scale is fixed to 1

The active FaceCapture workflow requires at least six included correspondences, requires the nose tip, and requires at least 15 mm of camera-depth spread. It reports training residuals plus leave-one-out validation. Entered points switched off from fitting are treated as independent validation holdouts.

### 4. Save the active profile

```text
POST /calibration/profile
GET  /calibration/profile
```

The Pi stores the active profile in:

```text
config/gemini_robot_calibration.json
```

Rigid profiles use `formatVersion = 2` and store the rotation matrix, translation, camera-space and robot-space calibration correspondences, and validation metrics.

### Validation before robot motion

1. Confirm `/camera/status` reports the Gemini ready.
2. Confirm `/camera/geometry` returns plausible nonzero intrinsics for the connected camera.
3. Capture the mannequin in FaceCapture.
4. Verify the RGB dots visually.
5. Verify most landmarks have plausible Camera XYZ values.
6. Include the nose and at least five other well-spread points.
7. Check training and leave-one-out errors rather than training RMS alone.
8. Save only when the rigid profile passes the selected tolerance.
9. Confirm `GET /calibration/profile` returns `formatVersion: 2`.
10. Perform a no-air robot positioning validation before enabling spraying.
