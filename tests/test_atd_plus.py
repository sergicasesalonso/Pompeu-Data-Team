"""B8 -> B3: broadcast ball candidates on the tactical timeline (synthetic)."""

import numpy as np

from pompeu.io import broadcast
from synthetic import PITCH, make_tracking


def _bcast(n=10):
    xy = np.full((n, 2, 2), np.nan)
    ball = np.column_stack((np.arange(n, dtype=float), np.full(n, 30.0)))
    ball[3] = np.nan                     # broadcast lost the ball here
    return make_tracking(xy, np.array([0, 1]), ball)


def _alignment(n_t, b_of_t, fit, flip=False, swap=False):
    return broadcast.Alignment(b_of_t=np.asarray(b_of_t, dtype=np.int64),
                               fit_of_t=np.asarray(fit, dtype=float),
                               orientation=(flip, swap), offsets=[])


def test_candidates_keep_good_fits_with_a_ball():
    bc = _bcast()
    # tactical rows 0..7: unaligned, aligned to b=2 (fit 0), to b=3 (no ball), to b=4 (fit 1),
    # to b=5 (fit 2.0, edge), to b=6 (fit 2.5, too bad), to b=7 (NaN fit), unaligned
    b_of_t = [-1, 2, 3, 4, 5, 6, 7, -1]
    fit = [np.nan, 0.0, 0.5, 1.0, 2.0, 2.5, np.nan, np.nan]
    df = broadcast.ball_candidates(_alignment(8, b_of_t, fit), bc, max_fit_m=2.0)
    assert list(df.columns) == ["frame", "x_m", "y_m", "conf", "source"]
    assert df["frame"].tolist() == [1, 3, 4]
    assert df["x_m"].tolist() == [2.0, 4.0, 5.0]
    assert np.allclose(df["y_m"], 30.0)
    assert np.allclose(df["conf"], [1.0, 0.75, 0.5])
    assert (df["source"] == "broadcast").all()


def test_candidates_follow_the_alignment_orientation_and_frame0():
    bc = _bcast()
    df = broadcast.ball_candidates(_alignment(2, [2, 4], [0.0, 0.0], flip=True, swap=True), bc,
                                   frame0=100)
    assert df["frame"].tolist() == [100, 101]
    assert np.allclose(df["x_m"], [PITCH[0] - 2.0, PITCH[0] - 4.0])
    assert np.allclose(df["y_m"], PITCH[1] - 30.0)


def test_candidates_empty_when_nothing_aligned():
    df = broadcast.ball_candidates(_alignment(3, [-1, -1, -1], [np.nan] * 3), _bcast())
    assert len(df) == 0
    assert list(df.columns) == ["frame", "x_m", "y_m", "conf", "source"]


def test_candidates_feed_augment_ball():
    from pompeu.vision import fuse
    n = 60
    tac_ball = np.column_stack((np.linspace(10, 40, n), np.full(n, 30.0)))
    tac_ball[20:30] = np.nan
    tac = make_tracking(np.full((n, 2, 2), np.nan), np.array([0, 1]), tac_ball)
    bball = np.column_stack((np.linspace(10, 40, n), np.full(n, 30.0)))
    bc = make_tracking(np.full((n, 2, 2), np.nan), np.array([0, 1]), bball)
    al = _alignment(n, np.arange(n), np.full(n, 0.5))
    df = broadcast.ball_candidates(al, bc)
    plus, report = fuse.augment_ball(tac, [df])
    assert report["accepted"] == 10
    assert report["accepted_per_source"] == {"broadcast": 10}
    assert plus.ball_present.all()


def _line(n, y=30.0, step=0.4):
    return np.column_stack((np.arange(n) * step, np.full(n, y)))


def test_default_filters_are_off():
    n = 50
    bc = make_tracking(np.full((n, 2, 2), np.nan), np.array([0, 1]), _line(n, step=2.0))
    al = _alignment(n, np.arange(n), np.zeros(n))
    plain = broadcast.ball_candidates(al, bc)
    df, st = broadcast.ball_candidates(al, bc, return_stats=True)
    assert df.equals(plain)
    assert st == {"candidates": n, "dropped_speed": 0, "dropped_reference": 0,
                  "reference_no_evidence": 0, "kept": n}


def test_speed_filter_drops_fast_broadcast_balls():
    n = 20
    ball = _line(n)                                  # 0.4 m/frame = 10 m/s
    ball[10:] += [15.0, 0.0]                         # a 15 m jump between rows 9 and 10
    bc = make_tracking(np.full((n, 2, 2), np.nan), np.array([0, 1]), ball)
    al = _alignment(n, np.arange(n), np.zeros(n))
    df, st = broadcast.ball_candidates(al, bc, max_speed_ms=25.0, return_stats=True)
    assert st["dropped_speed"] == 2                  # rows 9 and 10 straddle the jump
    assert 9 not in df["frame"].tolist() and 10 not in df["frame"].tolist()
    assert len(df) == n - 2


def test_reference_filter_drops_stretches_that_disagree_with_the_tactical_ball():
    # Two shots: rows 0..99 at offset 0, rows 100..199 at offset 50 (a new run).
    n = 200
    b_of_t = np.r_[np.arange(100), np.arange(50, 150)]
    bball = _line(200)
    bc = make_tracking(np.full((200, 2, 2), np.nan), np.array([0, 1]), bball)
    al = _alignment(n, b_of_t, np.zeros(n))
    ref = bball[b_of_t].copy()                       # the tactical ball agrees in run 1 ...
    ref[100:] += [0.0, 5.0]                          # ... and is 5 m off in run 2
    ref[40:60] = np.nan                              # one gap per run
    ref[140:160] = np.nan
    df, st = broadcast.ball_candidates(al, bc, reference_xy=ref, ref_window=30, ref_max_m=1.5,
                                       return_stats=True)
    frames = set(df["frame"].tolist())
    assert set(range(40, 60)) <= frames              # gap in the agreeing run: kept
    assert not (set(range(140, 160)) & frames)       # gap in the disagreeing run: dropped
    assert set(range(100, 140)) <= frames            # frames with a reference ball: never dropped
    assert st["dropped_reference"] == 20


def test_reference_filter_does_not_cross_runs():
    # The disagreeing run 2 must not condemn the gap at the end of the agreeing run 1.
    n = 120
    b_of_t = np.r_[np.arange(60), np.arange(100, 160)]
    bball = _line(200)
    bc = make_tracking(np.full((200, 2, 2), np.nan), np.array([0, 1]), bball)
    al = _alignment(n, b_of_t, np.zeros(n))
    ref = bball[b_of_t].copy()
    ref[60:] += [0.0, 5.0]
    ref[50:60] = np.nan
    df = broadcast.ball_candidates(al, bc, reference_xy=ref, ref_window=40)
    assert set(range(50, 60)) <= set(df["frame"].tolist())


def test_reference_filter_keeps_candidates_without_evidence():
    n = 30
    bc = make_tracking(np.full((n, 2, 2), np.nan), np.array([0, 1]), _line(n))
    al = _alignment(n, np.arange(n), np.zeros(n))
    ref = np.full((n, 2), np.nan)
    ref[:3] = bc.ball_xy[:3] + [0.0, 9.0]            # 3 reference frames < ref_min_frames
    df, st = broadcast.ball_candidates(al, bc, reference_xy=ref, return_stats=True)
    assert len(df) == n
    assert st["reference_no_evidence"] == n - 3


def test_aligned_runs_split_on_offset_change_and_gaps():
    run = broadcast._aligned_runs(np.array([-1, 0, 1, 2, 10, 11, -1, 12, 13]))
    assert run.tolist() == [-1, 0, 0, 0, 1, 1, -1, 2, 2]
