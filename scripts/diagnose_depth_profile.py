from __future__ import annotations

import time

from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline


def main() -> None:
    pipeline = Pipeline()
    config = Config()

    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    print("Testing explicit depth profile: 640x400 Y16 @ 15 FPS")
    try:
        depth_profile = profiles.get_video_stream_profile(640, 400, OBFormat.Y16, 15)
    except Exception as exc:
        print(f"Could not obtain 640x400 Y16 @ 15 profile: {exc}")
        print("Default depth profile:", profiles.get_default_video_stream_profile())
        return

    print("Selected profile:", depth_profile)
    config.enable_stream(depth_profile)

    try:
        pipeline.start(config)
        print("pipeline.start(): OK")

        framesets = 0
        depth_frames = 0
        deadline = time.monotonic() + 10.0

        while time.monotonic() < deadline and depth_frames < 5:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            framesets += 1
            depth = frames.get_depth_frame()
            if depth is not None:
                depth_frames += 1
                print(
                    f"depth frame {depth_frames}: "
                    f"{depth.get_width()}x{depth.get_height()} "
                    f"format={depth.get_format()} "
                    f"scale={depth.get_depth_scale()}"
                )

        print(f"result: framesets={framesets}, depth_frames={depth_frames}")
    except Exception as exc:
        print(f"stream error: {type(exc).__name__}: {exc}")
    finally:
        try:
            pipeline.stop()
        except Exception as exc:
            print(f"Error stopping pipeline: {exc}")


if __name__ == "__main__":
    main()
