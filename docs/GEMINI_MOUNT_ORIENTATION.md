# Gemini mount orientation

The Gemini may be mounted either native/upright (`0` degrees) or physically upside down (`180` degrees).

The Pi keeps the native RGB image, aligned depth image, and factory camera intrinsics unchanged. For calibration it creates an upright display image for FaceCapture and MediaPipe. Landmark pixels from that upright display are mapped back into the native Gemini pixel frame before depth sampling and camera deprojection.

## Current default

The default is `180` degrees, for an upside-down Gemini mount.

Check the active setting:

```bash
curl http://127.0.0.1:8000/camera/mount-orientation
```

Set an upside-down mount:

```bash
curl -X POST http://127.0.0.1:8000/camera/mount-orientation \
  -H 'Content-Type: application/json' \
  -d '{"rotation_degrees":180}'
```

Return to a native/upright mount:

```bash
curl -X POST http://127.0.0.1:8000/camera/mount-orientation \
  -H 'Content-Type: application/json' \
  -d '{"rotation_degrees":0}'
```

The setting is persisted in `config/gemini_mount_orientation.json` and applies to new captures. Each capture records the rotation used in its own metadata, so changing the setting later does not reinterpret an older capture.

Do not physically move or rotate the camera after taking the capture used for robot calibration. A new camera pose requires a new rigid camera-to-robot calibration.
