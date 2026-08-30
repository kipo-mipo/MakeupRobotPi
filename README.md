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

The MediaPipe rigid-calibration API reports version `0.8.0`.

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

### 6. Calibration point tester

The Pi exposes a guarded dry-run motion path through the local Moonraker/Klipper API. By default Moonraker is expected at:

```text
http://127.0.0.1:7125
```

Override it when needed:

```bash
export MOONRAKER_URL=http://127.0.0.1:7125
```

Check whether motion testing is allowed:

```bash
curl http://127.0.0.1:8000/robot/status
```

The Pi reports Klipper state, homed axes, current position, axis limits, and the configured safe Y plane. Test motion is blocked unless Klipper is ready, X/Y/Z are homed, no print is active or paused, and the target lies inside Klipper's reported axis limits.

Preview a target without moving:

```bash
curl -X POST http://127.0.0.1:8000/robot/test-move \
  -H 'Content-Type: application/json' \
  -d '{"x_mm":120,"y_mm":135,"z_mm":80,"execute":false}'
```

Execute the same target:

```bash
curl -X POST http://127.0.0.1:8000/robot/test-move \
  -H 'Content-Type: application/json' \
  -d '{"x_mm":120,"y_mm":135,"z_mm":80,"execute":true}'
```

The move sequence is intentionally conservative:

```text
wait for queued motion
G90
retract Y to the safe plane
move X/Z while retracted
advance Y to the test target
wait until the target move is physically complete
```

The default safe plane is Robot `Y=0 mm`, matching the calibration convention that +Y approaches the face. Override it with `ROBOT_TEST_SAFE_Y_MM` only if the physical robot uses a different retracted plane. Feed rates may be overridden with `ROBOT_TEST_RETRACT_FEED_MM_MIN`, `ROBOT_TEST_TRAVEL_FEED_MM_MIN`, and `ROBOT_TEST_APPROACH_FEED_MM_MIN`.

Retract the pointer back to the configured safe Y plane after inspecting a target:

```bash
curl -X POST http://127.0.0.1:8000/robot/retract
```

Emergency stop:

```bash
curl -X POST http://127.0.0.1:8000/robot/emergency-stop
```

The point tester assumes the Robot X/Y/Z values used during calibration are the same Klipper X/Y/Z coordinates in millimeters.

## 23° half-face serpentine spray test

For the angled-plane mannequin experiment, `scripts/spray_serpentine_test.py` runs a lawn-mower-style X/Z pattern:

- continuous vertical spray passes
- alternating up/down direction
- direct Raspberry Pi servo control on BCM GPIO18 by default
- spray servo released during each X reposition
- approximately 9 mm X step-over by default
- no Y-axis commands
- no solenoid commands

Klipper/Moonraker controls only X/Z motion. The airbrush trigger servo is driven directly from the Pi using 50 Hz PWM through `lgpio`.

Install/update dependencies in the active virtual environment after pulling:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The script deliberately does **not** guess servo positions. Supply the known-safe PWM pulse widths that correspond to the existing trigger-pressed and trigger-released positions.

Before combining servo and robot motion, test GPIO18 by itself:

```bash
python scripts/test_gpio18_servo.py \
  --spray-pulse-us YOUR_KNOWN_SPRAY_PULSE \
  --release-pulse-us YOUR_KNOWN_RELEASE_PULSE
```

That diagnostic sends no Klipper motion and no solenoid command. It commands release → spray for one second → release.

If your old servo code used 50 Hz duty cycle rather than microseconds, convert it with:

```text
pulse_us = duty_percent * 200
```

For example, a 7.5% duty cycle at 50 Hz is 1500 microseconds. Use only values already known to be safe for the installed linkage.

The default vertical spray speed is **15 mm/s (900 mm/min)**. With the measured roughly 17–20 mm spray diameter and 9 mm step-over, this is a useful first coat test with roughly 47–55% lateral overlap. If the coat is too light, try 10–12 mm/s. If it is too heavy/wet, try 20–25 mm/s.

Preview a pattern:

```bash
python scripts/spray_serpentine_test.py \
  --x-start 60 \
  --x-end 120 \
  --z-min 50 \
  --z-max 210 \
  --step-over 9 \
  --spray-speed 15 \
  --spray-pulse-us YOUR_KNOWN_SPRAY_PULSE \
  --release-pulse-us YOUR_KNOWN_RELEASE_PULSE
```

That is preview-only. It prints every pass and the direct-GPIO servo events without moving the robot or actuating the servo.

After checking the bounds, add `--execute`:

```bash
python scripts/spray_serpentine_test.py \
  --x-start 60 \
  --x-end 120 \
  --z-min 50 \
  --z-max 210 \
  --step-over 9 \
  --spray-speed 15 \
  --spray-pulse-us YOUR_KNOWN_SPRAY_PULSE \
  --release-pulse-us YOUR_KNOWN_RELEASE_PULSE \
  --execute
```

Use `--servo-gpio N` only if the servo signal is moved away from BCM GPIO18.

The script distributes X positions evenly so the final strip never becomes an unusually narrow high-overlap strip. For example, a requested maximum 9 mm step-over may become 8.6 mm when that divides the requested X span more evenly.

Run the planning tests with:

```bash
python -m unittest discover -s tests -p 'test_spray_serpentine.py'
```

For the first physical coat, use a paper-covered mannequin only. If possible, let the paper extend beyond the desired coated region in Z so acceleration/deceleration occurs outside the area being judged; otherwise endpoint regions can receive a little more material than the middle of each pass.

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
12. Use the calibration point tester with the pointer arm and air disabled.
13. Confirm multiple physical targets before enabling spraying.
