# MakeupRobot calibration handoff

_Last updated: 2026-08-10 PT / 2026-08-11 UTC_

This document records the validated Pi-side state of the one-time system-calibration work so future work can resume from the repository without reconstructing the development chat.

## Source-of-truth branches

- Pi repository: `kipo-mipo/MakeupRobotPi`
- Current Pi feature branch: `agent/pi-face-landmark-capture`
- iOS repository: `kipo-mipo/FaceCaptureIOS`
- Current iOS calibration branch: `system-calibration-workflow`

Do not merge the Pi feature branch into `main` until the live iOS mannequin-calibration flow has been validated end to end.

## Calibration architecture

The app is the calibration workflow, solver, verification, and approval authority.

The Pi is a camera/landmark measurement instrument during calibration. After the app approves a system calibration, the Pi will later receive and store the approved calibration package. Do not duplicate the app's raw-camera-U/V to robot-X/Z homography solver on the Pi.

The one-time system calibration has two stages:

1. Flat-board calibration: known raw Pi camera `(U,V)` correspondences to robot `(X,Z)` are used by the app to solve the planar mapping.
2. Mannequin depth calibration: Pi semantic face landmarks provide repeatable raw `(U,V)` identities; the app pairs those points with independently measured robot `(X,Y,Z)` coordinates.

The mannequin is only a known 3D calibration object. Its geometry is not assumed to match later human faces.

## Raw camera coordinate contract

Pi Camera Module 3 Standard is physically mounted sideways.

- Canonical raw capture size: **1280 x 720**
- Raw `U`: image horizontal, left to right
- Raw `V`: image vertical, top to bottom
- Detector input is rotated **90 degrees counterclockwise** only for MediaPipe.
- API returns the original raw sideways JPEG and raw U/V values.
- The app rotates for display.

Raw to upright display mapping:

```text
upright X = raw V
upright Y = raw width - 1 - raw U
```

Inverse detector-upright to raw mapping:

```text
raw U = raw width - 1 - upright Y
raw V = upright X
```

Do not "fix" orientation by rotating the API image or by returning upright coordinates; the raw convention is shared with flat-board calibration and is authoritative.

## App-facing endpoint contract

The iOS branch calls:

```text
POST /calibration/face-landmarks/capture
```

Request:

```json
{
  "request_id": "UUID",
  "purpose": "mannequin_depth_calibration",
  "return_image": true,
  "minimum_confidence": 0.5
}
```

Response shape:

```json
{
  "status": "ok",
  "request_id": "same UUID",
  "captured_at": "ISO-8601 UTC timestamp",
  "camera": {
    "model": "Pi Camera Module 3 Standard",
    "raw_width_px": 1280,
    "raw_height_px": 720,
    "rotation_degrees_ccw": 90
  },
  "image_jpeg_base64": "... or null",
  "landmarks": [
    {
      "id": "nose_tip",
      "u_px": 313.0,
      "v_px": 353.2,
      "confidence": 0.5,
      "source_index": 1,
      "confidence_source": "face_detector_acceptance_floor"
    }
  ],
  "detector": {
    "name": "MediaPipe Face Landmarker",
    "version": "0.10.18",
    "landmark_set": "makeuprobot_mannequin_v1",
    "left_right_convention": "anatomical_subject",
    "confidence_mode": "face detector acceptance floor; semantic landmarks require visual verification in the calibration app"
  }
}
```

The iOS decoder ignores the Pi-only traceability fields `source_index` and `confidence_source`.

## Stable semantic landmark set v1

Left/right means the mannequin/person's anatomical left/right, not viewer left/right.

| API ID | MediaPipe index | Validated meaning |
| --- | ---: | --- |
| `nose_tip` | 1 | physical nose tip on current mannequin/view |
| `left_inner_eye` | 362 | anatomical left inner eye corner |
| `right_inner_eye` | 133 | anatomical right inner eye corner |
| `left_mouth_corner` | 291 | anatomical left mouth corner |
| `right_mouth_corner` | 61 | anatomical right mouth corner |
| `chin` | 152 | bottom-center chin contour |

These semantic IDs are part of the app/Pi contract. If the underlying detector changes later, preserve the semantic IDs or version the mapping explicitly.

## Confidence semantics

MediaPipe Face Landmarker does not provide a useful calibrated per-vertex confidence for this mesh in the current Pi build. Earlier code attempted to use optional `presence`/`visibility` values and filtered out every selected landmark even though a face had been detected.

Current behavior is intentional:

- MediaPipe face detection/presence thresholds are configured at `0.50`.
- Once the overall face is accepted, the six semantic mesh points are returned with `confidence = 0.50`.
- `confidence_source = "face_detector_acceptance_floor"` documents that this is not a calibrated per-landmark probability.
- Semantic correctness is established visually, not by pretending the mesh has 99% per-point confidence.

The iOS request currently uses `minimum_confidence = 0.5`, which matches this behavior.

## Hardware/runtime state validated on the actual Pi

Pi OS environment encountered Python 3.13, which was incompatible with the ARM64 MediaPipe/NumPy wheel combination originally pinned. `setup_pi.sh` was changed to create a managed Python 3.12 environment.

Validated runtime:

```text
Python 3.12.13
MediaPipe 0.10.18
```

The camera backend uses Raspberry Pi `rpicam-*` commands so the MediaPipe Python environment does not need to import the OS Python 3.13 Picamera2 module.

Hardware-independent tests passed after this migration. Continue running:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

before live testing after future changes.

## Live mannequin validation completed

The real Pi endpoint successfully detected the mannequin and returned all six semantic landmarks.

The live upright annotated preview was inspected visually. All six points were confirmed to track the intended physical features, including the sharp nose tip under the current downward camera angle.

The current camera angle is noticeably downward, but there is no evidence that the selected semantic landmarks are incorrect. Do not change the camera mount solely because the view looks oblique; if camera geometry changes, the flat-board/mannequin calibration must ultimately correspond to the final fixed geometry.

## Repeatability testing

`repeatability.py` is a Pi-only diagnostic and is not part of the app API. It captures a stationary mannequin repeatedly and reports mean U/V, sample standard deviation, min/max, and spread.

### Initial 1280 x 720 framing

```text
landmark               n    mean U     sd U  spread U    mean V     sd V  spread V
nose_tip              10    172.84     1.74      5.35    372.60     0.74      2.14
left_inner_eye        10    345.12     0.44      1.37    432.53     0.68      1.69
right_inner_eye       10    345.48     0.67      2.09    304.79     0.80      2.45
left_mouth_corner     10    142.67     1.20      3.21    444.18     0.87      2.69
right_mouth_corner    10    141.88     0.86      2.85    305.51     1.19      4.02
chin                  10     67.16     1.37      4.55    379.66     0.76      2.02
```

### 2304 x 1296 experiment

Higher capture resolution did not improve the troublesome lower-face points after scaling jitter back to the 1280 x 720 reference coordinate system. It also changed camera framing/crop, so it is not the chosen canonical mode.

Keep the app/calibration capture at **1280 x 720**.

### Current preferred 1280 x 720 mannequin position

The mannequin was moved upward in the frame to give the chin more margin below it. This is the preferred current position.

```text
landmark               n    mean U     sd U  spread U    mean V     sd V  spread V
nose_tip              10    313.07     1.33      4.12    353.21     0.78      2.46
left_inner_eye        10    532.65     0.77      2.67    428.99     0.73      2.30
right_inner_eye       10    528.56     0.79      2.32    266.44     1.03      3.23
left_mouth_corner     10    276.23     1.13      3.67    449.52     0.88      2.77
right_mouth_corner    10    266.08     1.35      4.25    278.36     1.56      5.29
chin                  10    176.37     1.51      4.61    374.60     1.28      3.95
```

In the upright 720 x 1280 display, the current chin is around 86% of the way down the image, leaving substantially more useful margin below it than the earlier framing.

The observed ~1 to 1.5 px standard-deviation scale is acceptable for continuing calibration development. Do not convert that directly to robot millimeters without the solved camera/robot mapping.

## Pi-only live landmark stream

`landmark_stream.py` is a debug tool and does not change Shyla's app contract.

Start it with the main API server stopped so only one process owns the camera:

```bash
source .venv/bin/activate
python landmark_stream.py
```

Default VLC URL:

```text
http://<PI-IP>:8081/landmarks.mjpg
```

The stream is upright and overlays/labels the six semantic points. It is intended for camera-position and semantic-landmark verification, not as the app calibration transport.

## Next app/Pi integration steps

When resuming with the iOS calibration UI:

1. Keep Pi canonical capture at 1280 x 720 raw sideways orientation.
2. Start `main.py`, not `landmark_stream.py`, for app integration.
3. Have the app call `POST /calibration/face-landmarks/capture` with `return_image = true`.
4. Verify the app's rotated JPEG and overlay match the Pi VLC debug view.
5. Confirm all six semantic IDs remain correctly labeled.
6. Only then collect physical robot `(X,Y,Z)` for the mannequin points and let the app solve/validate mannequin depth calibration.
7. Multi-frame averaging may be considered later for the actual one-time measurement, but do not silently change the app-facing endpoint to average frames until the displayed-image/coordinate semantics are explicitly decided.

## Camera/chin-rest design note

The current mannequin position is usable, but the future human chin rest should be vertically adjustable rather than fixed to one exact mannequin height. The current image still has useful top clearance, so a modestly higher head position can be tested to reduce the effect of the downward camera view while retaining the full forehead and chin in frame. Final head-rest geometry should be chosen with the live annotated 1280 x 720 stream and then held fixed for system calibration.
