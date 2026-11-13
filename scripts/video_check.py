"""B0: is the tactical video the reference decode, aligned with the ATD, and does it have audio?

    python scripts/video_check.py              # full check (counts every packet, about a minute)
    python scripts/video_check.py --fast       # trust the container header for the frame count

Prints counts and fractions only. Exit code 1 if the frame count does not match the ATD.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.io import homography  # noqa: E402
from pompeu.vision import decode  # noqa: E402
import games  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="do not count packets")
    args = parser.parse_args()

    video_path = games.asset_paths()["video"]
    info = decode.probe(video_path, count_packets=not args.fast)
    tracking = load_tracking("atd")
    H, _, valid = homography.load()
    ball = decode.ball_in_image(tracking, H, valid, info["width"], info["height"])

    frames = info["frames_counted"] if info["frames_counted"] is not None else info["frames_header"]
    match = frames == tracking.n_frames
    print("video      {} {}x{} @ {:.3f} fps, {:.1f} s".format(
        info["codec"], info["width"], info["height"], info["fps"], info["duration_s"]))
    print("frames     header {}  counted {}  ATD {}  -> {}".format(
        info["frames_header"], info["frames_counted"], tracking.n_frames, "OK" if match else "MISMATCH"))
    print("homography {} rows, {} valid".format(len(H), int(valid.sum())))
    print("ball       {} graded frames with ball, {:.2%} project inside the image".format(
        ball["graded_with_ball"], ball["inside_fraction"]))
    print("audio      {}".format("{} {} Hz x{}".format(info["audio_codec"], info["audio_sample_rate"],
                                                     info["audio_channels"])
                                 if info["has_audio"] else "none"))
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
