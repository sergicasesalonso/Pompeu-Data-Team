"""B3: ball tracker and fusion, on synthetic inputs only."""

import time

import numpy as np
import pandas as pd

from pompeu.io.contract import ARRAY_FIELDS, SRC_ATD, SRC_INTERP, SRC_VIDEO, Tracking
from pompeu.vision import fuse
from pompeu.vision.decode import project_frames
from synthetic import make_tracking, static_players

FIRST_GRADED = 11185            # BAR-ATH half 1 starts here (DATA.md)
N = 500
GAP = (200, 260)                # ATD loses the ball on rows [200, 260): 2.4 s


def _truth(n=N):
    """Ball rolling at 10 m/s along x."""
    return np.stack([20.0 + 0.4 * np.arange(n), np.full(n, 34.0)], axis=1)


def _scene(gap=GAP):
    ball = _truth()
    ball[gap[0]:gap[1]] = np.nan
    players = static_players(N, [[10.0, 10.0], [95.0, 60.0]])
    return make_tracking(players, np.array([0, 1]), ball, frame0=FIRST_GRADED)


def _cands(rows, conf=0.7, source="video", offset=(0.0, 0.0)):
    rows = np.asarray(rows)
    xy = _truth()[rows] + np.asarray(offset)
    # A deterministic wobble of a few centimetres, as a detector would have.
    xy = xy + 0.05 * np.stack([np.sin(rows), np.cos(rows)], axis=1)
    return pd.DataFrame({"frame": FIRST_GRADED + rows, "x_m": xy[:, 0], "y_m": xy[:, 1],
                         "conf": conf, "source": source})


def _point(row, x, y, conf=0.99, source="video"):
    return pd.DataFrame({"frame": [FIRST_GRADED + row], "x_m": [x], "y_m": [y], "conf": [conf],
                         "source": [source]})


def test_atd_kept_untouched():
    tr = _scene()
    out, _ = fuse.augment_ball(tr, [_cands(np.arange(*GAP)), _cands(np.arange(0, 200), 0.9)])
    atd = tr.ball_source == SRC_ATD
    assert np.array_equal(out.ball_xy[atd], tr.ball_xy[atd])
    assert (out.ball_source[atd] == SRC_ATD).all()
    assert (out.ball_conf[atd] == 1).all()
    assert out.players_xy is not None and np.array_equal(out.players_xy, tr.players_xy,
                                                          equal_nan=True)


def test_gap_filled_by_consistent_candidates():
    tr = _scene()
    out, report = fuse.augment_ball(tr, [_cands(np.arange(*GAP), conf=0.7)])
    gap = slice(*GAP)
    assert (out.ball_source[gap] == SRC_VIDEO).all()
    assert np.allclose(out.ball_xy[gap], _truth()[gap], atol=0.1)
    assert np.allclose(out.ball_conf[gap], 0.7 * fuse.PARAMS["conf_scale"])
    assert report["accepted"] == GAP[1] - GAP[0]
    assert report["accepted_per_source"] == {"video": GAP[1] - GAP[0]}
    assert report["frames_with_ball_after"] == N
    assert report["graded_with_ball_after"] - report["graded_with_ball_before"] == 60


def test_far_false_positives_rejected():
    tr = _scene()
    truth = _cands(np.arange(*GAP), conf=0.6)
    spike = _point(230, 90.0, 5.0, conf=0.99)                    # one frame, 60 m away
    spot = _cands(np.arange(205, 216), conf=0.95, source="broadcast")
    spot["x_m"], spot["y_m"] = 94.0, 34.0                        # a static penalty spot
    out, report = fuse.augment_ball(tr, [truth, spike, spot])
    gap = slice(*GAP)
    assert np.allclose(out.ball_xy[gap], _truth()[gap], atol=0.1)
    assert report["accepted_per_source"] == {"video": 60, "broadcast": 0}
    assert report["rejected_gate_atd"] + report["rejected_unsupported"] \
        + report["rejected_same_frame"] >= 12


def test_isolated_single_frame_candidate_rejected():
    ball = _truth()
    ball[300:] = np.nan                                          # ATD lost for the last 8 s
    tr = make_tracking(static_players(N, [[10.0, 10.0]]), np.array([0]), ball, frame0=FIRST_GRADED)
    out, report = fuse.augment_ball(tr, [_point(450, *_truth()[450], conf=0.9)])
    assert np.isnan(out.ball_xy[450]).all()
    assert report["accepted"] == 0 and report["rejected_unsupported"] == 1
    assert np.array_equal(out.ball_xy, tr.ball_xy, equal_nan=True)


def test_atd_wins_on_disagreement():
    tr = _scene()
    out, report = fuse.augment_ball(tr, [_cands([100], conf=1.0, offset=(5.0, 0.0)),
                                         _cands([101], conf=1.0, offset=(0.3, 0.0))])
    assert np.array_equal(out.ball_xy[100], tr.ball_xy[100])
    assert out.ball_source[100] == SRC_ATD
    assert report["on_atd_conflict"] == 1 and report["on_atd_agree"] == 1
    assert report["accepted"] == 0


def test_short_gap_interpolated_long_gap_not():
    tr = _scene()
    rows = np.concatenate((np.arange(200, 215), np.arange(245, 260)))   # 30-frame hole: 1.2 s
    out, report = fuse.augment_ball(tr, [_cands(rows)])
    assert (out.ball_source[215:245] == SRC_INTERP).all()
    assert np.allclose(out.ball_conf[215:245], fuse.PARAMS["interp_conf"])
    assert np.allclose(out.ball_xy[215:245], _truth()[215:245], atol=0.15)
    assert report["interpolated"] == 30

    rows = np.concatenate((np.arange(200, 210), np.arange(250, 260)))   # 40 frames: 1.6 s
    out, report = fuse.augment_ball(tr, [_cands(rows)])
    assert np.isnan(out.ball_xy[210:250]).all()
    assert report["interpolated"] == 0


def test_deterministic():
    tr = _scene()
    cands = [_cands(np.arange(*GAP), conf=0.6), _point(230, 90.0, 5.0),
             _cands(np.arange(200, 260, 2), conf=0.6, source="broadcast", offset=(0.4, 0.0))]
    a, ra = fuse.augment_ball(tr, cands)
    b, rb = fuse.augment_ball(tr, cands)
    for name in ARRAY_FIELDS:
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True)
    assert ra == rb


def test_output_honours_the_contract():
    tr = _scene()
    out, _ = fuse.augment_ball(tr, [_cands(np.arange(*GAP)), _point(230, 90.0, 5.0)])
    out.validate()
    again = Tracking(frame0=out.frame0, meta=dict(out.meta),
                     **{name: getattr(out, name) for name in ARRAY_FIELDS})
    assert again.n_frames == tr.n_frames and again.frame0 == tr.frame0
    assert np.isnan(out.ball_z[slice(*GAP)]).all()


def test_pixels_to_metres_round_trip():
    n = 20
    Hm = np.repeat(np.array([[1500.0, 220.0, 110.0],
                             [40.0, 880.0, 90.0],
                             [0.08, 0.25, 1.0]])[None], n, axis=0)
    Hm[:, 0, 2] += np.arange(n)                     # a slowly panning camera
    rows = np.repeat(np.arange(n), 3)               # top-3 per frame
    x_m = np.tile([10.0, 52.65, 100.0], n)
    y_m = np.tile([5.0, 34.0, 60.0], n)
    px, py = project_frames(Hm[rows], x_m, y_m)
    df = pd.DataFrame({"frame": rows, "x_px": px, "y_px": py, "conf": 0.5})
    out = fuse.video_candidates_to_metres(df, Hm)
    assert len(out) == len(df)
    assert np.allclose(out["x_m"], x_m, atol=1e-6) and np.allclose(out["y_m"], y_m, atol=1e-6)
    assert (out["source"] == "video").all()

    Hm[3] = np.nan                                  # an invalid frame drops its candidates
    assert len(fuse.video_candidates_to_metres(df, Hm)) == len(df) - 3


def test_pixels_to_metres_rounding_absorbs_last_bit_noise():
    n = 5
    Hm = np.repeat(np.array([[1500.0, 220.0, 110.0],
                             [40.0, 880.0, 90.0],
                             [0.08, 0.25, 1.0]])[None], n, axis=0)
    rows = np.arange(n)
    px, py = project_frames(Hm, np.full(n, 31.237), np.full(n, 17.401))
    df = pd.DataFrame({"frame": rows, "x_px": px, "y_px": py, "conf": 0.123456})
    noisy = df.assign(x_px=px + 1e-9, y_px=py - 1e-9, conf=0.1234561)   # another GPU's bits
    a, b = fuse.video_candidates_to_metres(df, Hm), fuse.video_candidates_to_metres(noisy, Hm)
    assert a.equals(b)
    assert np.allclose(a["x_m"], 31.24) and np.allclose(a["y_m"], 17.40)
    assert (a["conf"] == 0.123).all()
    raw = fuse.video_candidates_to_metres(df, Hm, params={"round_m": 0, "round_conf": None})
    assert np.allclose(raw["x_m"], 31.237, atol=1e-6)


def test_whole_match_size_is_fast():
    n = 199_319
    t = np.arange(n) / 25.0
    truth = np.stack([52.0 + 40.0 * np.sin(t / 20.0), 34.0 + 25.0 * np.sin(t / 13.0)], axis=1)
    ball = truth.copy()
    lost = (np.arange(n) // 250) % 5 < 2                         # 40 % lost, in 10 s blocks
    ball[lost] = np.nan
    players = static_players(n, [[10.0, 10.0], [95.0, 60.0]])
    tr = make_tracking(players, np.array([0, 1]), ball)
    rows = np.repeat(np.flatnonzero(lost), 3)
    xy = truth[rows] + np.tile([[0.0, 0.0], [30.0, 0.0], [0.0, 20.0]], (len(rows) // 3, 1))
    conf = np.tile([0.7, 0.5, 0.4], len(rows) // 3)
    cands = pd.DataFrame({"frame": rows, "x_m": xy[:, 0], "y_m": xy[:, 1], "conf": conf})
    start = time.perf_counter()
    out, report = fuse.augment_ball(tr, [cands])
    assert time.perf_counter() - start < 60
    assert report["accepted"] == int(lost.sum())
    assert np.allclose(out.ball_xy, truth)
