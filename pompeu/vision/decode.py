"""B0: check the tactical video before anything in Line B reads a pixel from it.

    from pompeu.vision import decode
    decode.probe(path)                     # streams: frame count, size, fps, audio yes/no
    decode.ball_in_image(tracking, H, valid, width, height)

Three things have to hold before pseudo-labels (B1) or a detector (B2) can be trusted:

1. the decoded video has exactly as many frames as the ATD (199,319) -- the FAQ warns that a decode
   that disagrees is not the reference decode, and an off-by-one shifts every label;
2. the ATD ball projected through the homography lands inside the image on almost every frame
   that has a ball -- otherwise the frames, the tracking or the convention are misaligned;
3. whether there is an audio track, which decides if B7 (referee whistle) is possible.

Everything here returns counts and fractions, never pixels or frames: these are the only outputs
that may leave the machine the data is on. ffprobe (part of FFmpeg) must be on PATH.
"""

import json
import shutil
import subprocess

import numpy as np

from pompeu import paths  # noqa: F401  (makes games importable)
import games

# Params
FFPROBE = "ffprobe"


def _ffprobe(path, extra):
    if shutil.which(FFPROBE) is None:
        raise RuntimeError("ffprobe not found on PATH; install FFmpeg")
    out = subprocess.run([FFPROBE, "-v", "error", "-of", "json", *extra, str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def probe(path, count_packets=True):
    """Video and audio streams of `path`, as a dict of plain numbers.

    count_packets reads every packet (about a minute for the full match) to count frames instead of
    trusting the container header; for this H.264 file one packet is one frame.
    """
    extra = ["-show_entries",
             "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames,"
             "nb_read_packets,sample_rate,channels:format=duration"]
    if count_packets:
        extra = ["-count_packets", *extra]
    info = _ffprobe(path, extra)
    video = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not video:
        raise ValueError("{} has no video stream".format(path))
    v = video[0]
    num, den = (int(part) for part in v.get("r_frame_rate", "0/1").split("/"))
    return {
        "codec": v.get("codec_name"),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": num / den if den else float("nan"),
        "frames_header": int(v["nb_frames"]) if v.get("nb_frames", "N/A") != "N/A" else None,
        "frames_counted": int(v["nb_read_packets"]) if "nb_read_packets" in v else None,
        "duration_s": float(info.get("format", {}).get("duration", "nan")),
        "has_audio": bool(audio),
        "audio_codec": audio[0].get("codec_name") if audio else None,
        "audio_sample_rate": int(audio[0]["sample_rate"]) if audio and "sample_rate" in audio[0] else None,
        "audio_channels": int(audio[0]["channels"]) if audio and "channels" in audio[0] else None,
    }


def project_frames(H, x_m, y_m, game=games.GAME):
    """Pitch metres -> pixels with one 3x3 matrix per frame. H is (N, 3, 3); x_m, y_m are (N,)."""
    length, width = games.GAMES[game]["pitch"]
    x, y = np.asarray(x_m, dtype=np.float64) / length, np.asarray(y_m, dtype=np.float64) / width
    den = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    px = (H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]) / den
    py = (H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]) / den
    return px, py


def ball_in_image(tracking, H, valid, width, height, margin_px=0.0):
    """How often the tracked ball projects inside the image. Counts only, no coordinates.

    Row i of the Tracking and of H must be the same video frame (both start at frame 0 for the
    tactical recording). Frames outside the two halves are left out.
    """
    if len(H) != tracking.n_frames:
        raise ValueError("homography has {} rows, tracking {}".format(len(H), tracking.n_frames))
    graded = tracking.half() > 0
    has_ball = np.isfinite(tracking.ball_xy).all(axis=1) & valid & graded
    px, py = project_frames(H[has_ball], tracking.ball_xy[has_ball, 0], tracking.ball_xy[has_ball, 1],
                            tracking.meta.get("game", games.GAME))
    inside = ((px >= -margin_px) & (px < width + margin_px)
              & (py >= -margin_px) & (py < height + margin_px))
    n_ball = int(has_ball.sum())
    return {
        "graded_frames": int(graded.sum()),
        "graded_with_ball": n_ball,
        "ball_inside_image": int(inside.sum()),
        "inside_fraction": float(inside.mean()) if n_ball else float("nan"),
    }
