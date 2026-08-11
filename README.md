# MakeupRobotPi

Raspberry Pi service for MakeupRobot camera capture and app communication.

## Calibration architecture

The one-time system calibration has two stages:

1. **Flat Board Calibration** — the iOS app solves raw Pi-camera `(U, V)` pixels
   to robot `(X, Z)` and handles the fixed planar camera/nozzle relationship.
2. **Mannequin Depth Calibration** — the Pi identifies repeatable semantic
   facial landmarks; the app pairs their raw `(U, V)` positions with measured
   physical `(X, Y, Z)` coordinates and later solves the depth model.

The app remains the calibration/computation authority. The Pi acts as a camera
measurement instrument during calibration and later stores/uses the approved
system calibration package.

Do not add a second U/V-to-X/Z calibration solver on the Pi.

## Coordinate contract

The physical Pi Camera Module 3 is mounted sideways.

- Raw capture: **1280 x 720**
- Raw `U`: horizontal pixel coordinate, increasing left to right
- Raw `V`: vertical pixel coordinate, increasing top to bottom
- Detector input: temporary **90 degree counterclockwise** rotation
- API image: the original **unrotated raw JPEG**
- API landmark coordinates: transformed back to **raw U/V**

The iOS app rotates the returned JPEG and landmark overlay for display. The Pi
must not pre-rotate the returned image or return upright-display coordinates.

## Mannequin landmark endpoint

### `POST /calibration/face-landmarks/capture`

This endpoint is the contract used by the FaceCaptureIOS
`system-calibration-workflow` branch.

Example request:

```json
{
  "request_id": "19c5b823-1296-487d-85fd-27df422d5232",
  "purpose": "mannequin_depth_calibration",
  "return_image": true,
  "minimum_confidence": 0.5
}
```

Example response shape:

```json
{
  "status": "ok",
  "request_id": "19c5b823-1296-487d-85fd-27df422d5232",
  "captured_at": "2026-08-10T23:45:00+00:00",
  "camera": {
    "model": "Pi Camera Module 3 Standard (imx708)",
    "raw_width_px": 1280,
    "raw_height_px": 720,
    "rotation_degrees_ccw": 90
  },
  "image_jpeg_base64": "...",
  "landmarks": [
    {
      "id": "nose_tip",
      "u_px": 650.2,
      "v_px": 349.8,
      "confidence": 0.5,
      "source_index": 1,
      "confidence_source": "model_acceptance_floor"
    }
  ],
  "detector": {
    "name": "MediaPipe Face Landmarker",
    "version": "0.10.9",
    "landmark_set": "makeuprobot_mannequin_v1",
    "left_right_convention": "anatomical_subject"
  }
}
```

`source_index` and `confidence_source` are extra Pi-side traceability fields.
The current iOS decoder safely ignores them.

If `return_image` is false, `image_jpeg_base64` is null.

## Calibration landmark set v1

The first mannequin-calibration packet intentionally uses a small stable subset
instead of dumping hundreds of mesh vertices:

| API ID | MediaPipe index | Meaning |
| --- | ---: | --- |
| `nose_tip` | 1 | nose tip |
| `left_inner_eye` | 362 | mannequin/person anatomical left inner eye corner |
| `right_inner_eye` | 133 | mannequin/person anatomical right inner eye corner |
| `left_mouth_corner` | 291 | anatomical left mouth corner |
| `right_mouth_corner` | 61 | anatomical right mouth corner |
| `chin` | 152 | bottom-center face oval / chin |

Left/right always means the mannequin/person's own anatomical left/right, not
the viewer's left/right.

These IDs are an API contract. If the underlying detector changes, keep the
semantic IDs stable and version the mapping rather than silently changing what
an ID means.

## Confidence behavior

MediaPipe's landmark container can expose `presence` and `visibility`, but
models are allowed to leave those values unset. The Pi therefore does not
invent high per-point probabilities.

For each selected landmark:

1. If presence/visibility are supplied, use the more conservative supplied
   score.
2. If neither is supplied, report the Face Landmarker model acceptance floor
   (`0.50`) and label its traceability field
   `confidence_source = "model_acceptance_floor"`.

This means a displayed `50%` in the current app can mean "the model accepted
the face at the configured threshold; this detector did not supply a separate
per-landmark probability." It should not be interpreted as a calibrated
statistical probability.

## Other API endpoints

### `GET /status`

Reports server health and landmark subsystem readiness.

### `POST /test`

Simple app/Pi communication test endpoint.

## HTTP behavior

- `200` — capture succeeded
- `422` — unsupported purpose, no detected face, or no selected landmarks meet
  the requested confidence threshold
- `503` — camera, model, or runtime dependency unavailable
- `500` — another capture/detection failure

## Raspberry Pi setup

From the repository root on the Pi:

```bash
bash setup_pi.sh
source .venv/bin/activate
python -m unittest discover -s tests
python main.py
```

Then test the calibration endpoint locally:

```bash
curl -X POST http://localhost:8000/calibration/face-landmarks/capture \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "19c5b823-1296-487d-85fd-27df422d5232",
    "purpose": "mannequin_depth_calibration",
    "return_image": true,
    "minimum_confidence": 0.5
  }'
```

The response will be large when `return_image` is true because it contains a
base64 JPEG.

## Validation order

Before using mannequin measurements for a depth solver:

1. Confirm the returned JPEG is the same raw sideways orientation used by the
   flat-board calibration.
2. Confirm the app's rotated overlay places each landmark on the intended
   semantic feature.
3. Repeat captures on the stationary mannequin and measure U/V jitter.
4. Only keep landmarks that are stable enough across repeated captures.
5. Then enter independently measured physical X/Y/Z coordinates in the app.

The mannequin detector is calibration instrumentation only at this stage. This
branch does not command Klipper, motors, the airbrush, compressor, or solenoid.
