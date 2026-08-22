from __future__ import annotations

import time

from pyorbbecsdk import Config, OBSensorType, Pipeline


def test_stream(name: str, sensor_types: list[OBSensorType], seconds: float = 5.0) -> None:
    print(f"\n=== {name} ===")
    pipeline = Pipeline()
    config = Config()

    for sensor_type in sensor_types:
        profiles = pipeline.get_stream_profile_list(sensor_type)
        profile = profiles.get_default_video_stream_profile()
        print(f"{sensor_type.name} default profile: {profile}")
        config.enable_stream(profile)

    started = False
    try:
        pipeline.start(config)
        started = True
        print("pipeline.start(): OK")

        deadline = time.monotonic() + seconds
        framesets = 0
        color_frames = 0
        depth_frames = 0

        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            framesets += 1
            if frames.get_color_frame() is not None:
                color_frames += 1
            if frames.get_depth_frame() is not None:
                depth_frames += 1

            if framesets >= 5:
                break

        print(
            f"result: framesets={framesets}, "
            f"color_frames={color_frames}, depth_frames={depth_frames}"
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception:
                pass
        time.sleep(1.0)


def main() -> None:
    test_stream("DEPTH ONLY", [OBSensorType.DEPTH_SENSOR])
    test_stream("COLOR ONLY", [OBSensorType.COLOR_SENSOR])
    test_stream(
        "COLOR + DEPTH (default profiles, no frame sync, no aggregate requirement)",
        [OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR],
        seconds=10.0,
    )


if __name__ == "__main__":
    main()
