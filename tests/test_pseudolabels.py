"""B1: ball pseudo-labels, on synthetic inputs only."""

import numpy as np

from pompeu.io.contract import SRC_INTERP
from pompeu.vision import pseudolabels
from synthetic import make_tracking, static_players

FIRST_GRADED = 11185            # BAR-ATH half 1 starts here (DATA.md)
W, H_PX = 1920, 1080


def _homography(n):
    """Normalised pitch -> full image, the same for every frame."""
    return np.repeat(np.array([[W, 0.0, 0.0], [0.0, H_PX, 0.0], [0.0, 0.0, 1.0]])[None], n, axis=0)


def _scene(n=3000):
    """A ball rolling slowly across the pitch, two players far from it."""
    x = np.linspace(20.0, 80.0, n)
    ball = np.stack([x, np.full(n, 34.0)], axis=1)
    players = static_players(n, [[10.0, 10.0], [95.0, 60.0]])
    return make_tracking(players, np.array([0, 1]), ball, frame0=FIRST_GRADED)


def test_keeps_observed_balls_and_reports_counts():
    tr = _scene()
    n = tr.n_frames
    table, report = pseudolabels.build(tr, _homography(n), np.zeros(n), np.ones(n, bool), W, H_PX)
    assert report["kept"] == len(table) == n
    assert set(table.columns) == {"frame", "x_px", "y_px", "crowded", "split"}
    assert table["frame"].iloc[0] == FIRST_GRADED
    assert np.isclose(table["x_px"].iloc[0], 20.0 / 105.3 * W, atol=0.1)


def test_drops_interpolated_missing_and_off_image_balls():
    tr = _scene()
    n = tr.n_frames
    tr.ball_source[10] = SRC_INTERP                # filled by cleaning, not observed
    tr.ball_xy[20] = np.nan                        # no ball
    tr.ball_conf[20] = 0.0
    Hm = _homography(n)
    Hm[30, 0, 2] = 5000.0                          # projects off the image
    table, report = pseudolabels.build(tr, Hm, np.zeros(n), np.ones(n, bool), W, H_PX)
    kept = set(table["frame"])
    for i in (10, 20, 30):
        assert FIRST_GRADED + i not in kept
    assert report["dropped_outside_image"] == 1


def test_drops_untrustworthy_homography_and_flags_crowded_balls():
    tr = _scene()
    n = tr.n_frames
    err = np.zeros(n)
    err[:100] = 9.0                                # worst 100 of 3000 (3 %), above the 95 % cut
    tr.players_xy[500, 0] = tr.ball_xy[500] + [0.5, 0.0]   # a player 0.5 m from the ball
    table, report = pseudolabels.build(tr, _homography(n), err, np.ones(n, bool), W, H_PX)
    assert report["dropped_homography"] == 100
    assert bool(table.set_index("frame").loc[FIRST_GRADED + 500, "crowded"])


def test_split_is_temporal_blocks():
    tr = _scene(25 * 60 * 6)                       # six minutes
    n = tr.n_frames
    table, _ = pseudolabels.build(tr, _homography(n), np.zeros(n), np.ones(n, bool), W, H_PX)
    # within one 60 s block the split never changes
    block = table["frame"] // (25 * 60)
    assert (table.groupby(block)["split"].nunique() == 1).all()
    assert set(table["split"]) == {"train", "val"}
