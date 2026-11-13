"""Synthetic checks of the experimental TCN's input helpers (no data, no torch)."""

import numpy as np

from pompeu.experimental import tcn
from pompeu.io.contract import Tracking


def _tracking(n=20, k=4):
    rng = np.random.default_rng(0)
    xy = rng.uniform([0, 0], [105, 68], size=(n, k, 2))
    return Tracking(players_xy=xy, players_team=np.tile([0, 0, 1, 1], (n, 1)),
                    players_conf=np.ones((n, k)), ball_xy=rng.uniform([0, 0], [105, 68], (n, 2)),
                    ball_z=np.full(n, np.nan), ball_conf=np.ones(n), ball_source=np.ones(n),
                    meta={"pitch": (105.0, 68.0), "fps": 25})


def test_mirror_y_is_an_involution_and_keeps_x():
    tr = _tracking()
    once = tcn.mirror_y(tr)
    assert np.allclose(once.players_xy[..., 0], tr.players_xy[..., 0])
    assert np.allclose(once.players_xy[..., 1], 68.0 - tr.players_xy[..., 1])
    twice = tcn.mirror_y(once)
    assert np.allclose(twice.players_xy, tr.players_xy)
    assert np.allclose(twice.ball_xy, tr.ball_xy)


def test_normalise_adds_flags_and_reorders_columns():
    a = {"X": np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]], np.float32), "names": ["a", "b"]}
    norm = tcn.fit_normalisation([dict(a, X=np.repeat(a["X"], 5, axis=0))])
    assert norm["nan_cols"] == [1]
    Z = tcn.normalise(a["X"][:, ::-1].copy(), ["b", "a"], norm)
    assert Z.shape == (3, 3)
    assert np.all(np.isfinite(Z))
    assert Z[0, 2] == 1.0 and Z[1, 2] == 0.0                # missing flag of "b"
    assert Z[0, 0] < Z[2, 0]                                # "a" back in first place


def test_smooth_labels_majority():
    log_p = np.log(np.array([[0.9, 0.05, 0.05]] * 5 + [[0.05, 0.9, 0.05]] + [[0.9, 0.05, 0.05]] * 5))
    assert np.all(tcn.smooth_labels(log_p, 5) == 0)
    assert tcn.smooth_labels(log_p, 1)[5] == 1
