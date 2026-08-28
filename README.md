# MakeupRobotPi

Raspberry Pi API for the MakeupRobot prototype.

## One-stage Gemini 215 rigid calibration

The active calibration is a physical 3D camera-to-robot calibration. There is no flat-board stage and no free affine camera fit.

For each facial landmark:

1. The Pi captures native distorted Gemini RGB plus software depth-to-color aligned depth.
2. The Pi rotates only the display/detector copy when the camera mount is configured at 180°.
3. MediaPipe Face Landmarker runs on that upright RGB view and returns selected display `U/V` points.
4. The Pi maps each display point back to the native distorted RGB pixel.
5. Using the Gemini RGB intrinsics and distortion coefficients, the Pi undistorts that RGB pixel onto the undistorted color grid used by the aligned depth image.
6. The Pi samples aligned depth at that corrected pixel, then deprojects the same undistorted pixel plus depth into Gemini RGB optical `Camera X/Y/Z` in millimeters.
7. FaceCapture pairs that camera-space point with measured `Robot X/Y/Z`.
8. FaceCapture solves one rigid transform only: a 3×3 rotation plus a 3D translation.

The solver is not allowed to stretch, shear, or independently scale axes. That prevents a low-residual affine fit from hiding an incorrect physical mapping.

### Selected MediaPipe calibration landmarks

The Face Landmarker exposes hundreds of mesh points, but the calibration intentionally uses only points that are visually targetable and useful for rigid calibration:

- left/right outer eye corners
- left/right iris centers
- left/right inner eye corners
- nose bridge
- nose tip
- left/right mouth corners
- upper/lower lip centers
- chin

That is up to 13 landmarks. Iris centers are optional if the active MediaPipe model does not expose iris indices.

### Install

Use a 64-bit Linux install and Python 3.9+ if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
bash scripts/setup_face_landmarker.sh
```

The setup script downloads MediaPipe's `face_landmarker.task` into `models/`. The model file is intentionally not committed to Git.

On Linux, Orbbec requires a one-time device-permission setup. After installing `pyorbbecsdk2`, find its installed location and run its environment setup script:

```bash
python -c "import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))"
```

Then run `shared/setup_env.py` from that directory with the permissions it requests. Replug the camera afterward if needed.

### Run the API

```bash
python main.py
```

The MediaPipe rigid-calibration API reports version `0.7.0`.

### Check camera discovery

```bash
curl http://127.0.0.1:8000/camera/status
```

### Check Gemini intrinsics

```bash
curl http://127.0.0.1:8000/camera/geometry
```

A valid response must contain nonzero RGB focal lengths `fx/fy`, principal point `cx/cy`, RGB distortion coefficients, and the physical camera serial number. The Orbbec SDK calibration is read after the active color/depth streams have produced frames.

### Check MediaPipe readiness

```bash
curl http://127.0.0.1:8000/calibration/landmarks/status
```

A ready response should report:

```text
ready: true
mediapipe_installed: true
model_exists: true
landmark_count_requested: 13
```

### 1. Capture RGB + aligned depth

```bash
curl -X POST http://127.0.0.1:8000/calibration/capture
```

A successful capture returns URLs for:

- the native Gemini RGB calibration image
- the software depth-to-color aligned 16-bit raw depth image
- capture metadata with depth scale and device identity

Files are written under `captures/`.

### 2. Detect MediaPipe landmarks on that capture

```bash
curl -X POST http://127.0.0.1:8000/calibration/landmarks \
  -H 'Content-Type: application/json' \
  -d '{"capture_id":"CAPTURE_ID"}'
```

The response contains the selected MediaPipe point IDs, raw Gemini RGB `u_px/v_px`, display names, source mesh indices, and detector metadata.

### 3. Sample depth and physical camera XYZ

The app sends those exact MediaPipe display pixels back to the exact capture. The Pi maps them to native distorted RGB pixels, undistorts them onto the aligned-depth grid, and only then samples depth:

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

Each valid sample returns the user-facing display pixel plus diagnostic native/aligned pixels and physical camera coordinates:

```text
u_px / v_px                 # upright display / MediaPipe pixel
raw_u_px / raw_v_px         # native distorted Gemini RGB pixel
aligned_u_px / aligned_v_px # undistorted color-aligned depth pixel
distortion_shift_px
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

The normal depth estimator uses a 5×5 median centered on the undistorted color-aligned pixel. If that window contains no depth, the existing face-consistent fallback may search slightly farther while rejecting values inconsistent with the other facial depths.

The first rigid depth-sample request also records the camera calibration in that capture's metadata so the camera model used for deprojection is auditable later.

### 4. Rigid camera-to-robot transform

FaceCapture solves:

```text
P_robot = R * P_camera + t
```

where:

- `R` is a proper 3D rotation matrix
- `t` is a three-element translation in millimeters
- scale is fixed to 1

The active FaceCapture workflow requires at least eight included correspondences, requires the nose tip, and requires at least 15 mm of camera-depth spread. Ten to thirteen good points are recommended when depth is valid. It reports training residuals plus leave-one-out validation. Entered points switched off from fitting are treated as independent validation holdouts.

Every new Gemini capture starts with blank Robot XYZ ground truth. Old calibration measurements are not auto-imported into a new image because moving the mannequin or camera invalidates those coordinates.

### 5. Save the active profile

```text
POST /calibration/profile
GET  /calibration/profile
```

The Pi stores the active profile in:

```text
config/gemini_robot_calibration.json
```

Current MediaPipe rigid profiles use `formatVersion = 3` and store the rotation matrix, translation, camera-space and robot-space calibration correspondences, and validation metrics.

### Validation before robot motion

1. Confirm `/camera/status` reports the Gemini ready.
2. Confirm `/camera/geometry` returns plausible nonzero intrinsics for the connected camera.
3. Confirm `/calibration/landmarks/status` reports MediaPipe and the model ready.
4. Capture the mannequin in FaceCapture.
5. Verify the MediaPipe RGB dots visually, especially the nose tip.
6. Verify most selected landmarks have plausible Camera XYZ values.
7. Enter fresh Robot XYZ measurements for the current capture only.
8. Include the nose and at least seven other well-spread points; use more good points when practical.
9. Check training and leave-one-out errors rather than training RMS alone.
10. Save only when the rigid profile passes the selected tolerance.
11. Confirm `GET /calibration/profile` returns `formatVersion: 3`.
12. Perform a no-air robot positioning validation before enabling spraying.
