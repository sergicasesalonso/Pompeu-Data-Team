"""B0: video probe and ball-in-image check, on synthetic inputs only."""

import shutil
import subprocess

import numpy as np
import pytest

from pompeu.vision import decode
from synthetic import make_tracking, static_players

FIRST_GRADED = 11185            # BAR-ATH half 1 starts here (DATA.md)


def _tracking(n, ball_xy):
    """n frames starting at the first graded frame, two static players and the given ball."""
    players = static_players(n, [[30.0, 30.0], [70.0, 40.0]])
    return make_tracking(players, np.array([0, 1]), np.asarray(ball_xy, dtype=float),
                         frame0=FIRST_GRADED)


def _pitch_to_image(width, height):
    """A homography that maps the normalised pitch (0..1, 0..1) onto the full image."""
    return np.array([[width, 0.0, 0.0], [0.0, height, 0.0], [0.0, 0.0, 1.0]])


def test_project_frames_matches_starter_project():
    from starter import load_homography
    H = np.stack([_pitch_to_image(1920, 1080), _pitch_to_image(1280, 720)])
    x_m, y_m = np.array([10.0, 52.65]), np.array([5.0, 34.0])
    px, py = decode.project_frames(H, x_m, y_m)
    for i in range(2):
        ex, ey = load_homography.project(H[i], x_m[i] / 105.3, y_m[i] / 68.0)
        assert px[i] == pytest.approx(ex) and py[i] == pytest.approx(ey)


def test_ball_in_image_counts_only_graded_frames_with_a_ball():
    n = 10
    tr = _tracking(n, np.tile([52.0, 34.0], (n, 1)))
    tr.ball_xy[2] = np.nan                              # no ball on one frame
    H = np.repeat(_pitch_to_image(1920, 1080)[None], n, axis=0)
    H[3, 0, 2] = 5000.0                                 # one frame projects off the image
    valid = np.ones(n, dtype=bool)
    out = decode.ball_in_image(tr, H, valid, 1920, 1080)
    assert out["graded_with_ball"] == n - 1
    assert out["ball_inside_image"] == n - 2
    assert set(out) == {"graded_frames", "graded_with_ball", "ball_inside_image", "inside_fraction"}


def test_ball_in_image_rejects_misaligned_inputs():
    tr = _tracking(5, np.tile([52.0, 34.0], (5, 1)))
    with pytest.raises(ValueError):
        decode.ball_in_image(tr, np.zeros((4, 3, 3)), np.ones(4, bool), 1920, 1080)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_counts_frames_and_audio(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=3000:sample_rate=16000",
                    "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(path)], check=True)
    info = decode.probe(path)
    assert (info["width"], info["height"]) == (64, 48)
    assert info["fps"] == pytest.approx(25.0)
    assert info["frames_counted"] == 50
    assert info["has_audio"] and info["audio_sample_rate"] == 16000
