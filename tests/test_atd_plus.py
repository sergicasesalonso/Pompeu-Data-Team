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
