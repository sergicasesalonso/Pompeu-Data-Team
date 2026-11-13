"""scripts/experiments/gap_team.py: gap segmentation, carry, logistic fit, smoothing, buckets."""

import importlib.util
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                     "experiments", "gap_team.py")
_spec = importlib.util.spec_from_file_location("gap_team", _PATH)
gt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt_mod)

H, A, D, X = 0, 1, 2, 3


def test_gaps_never_cross_halves_and_skip_out_of_half():
    half = np.array([0, 1, 1, 1, 1, 2, 2, 2, 0], dtype=np.int8)
    ball = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    starts, stops = gt_mod.gaps(ball, half)
    assert starts.tolist() == [2, 5, 7] and stops.tolist() == [5, 6, 8]


def test_gap_table_carry_skips_dead_and_opening_gaps_have_none():
    half = np.ones(10, dtype=np.int8)
    ball = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    gt = np.array([H, H, A, A, D, A, H, H, H, H], dtype=np.int8)
    nearest = np.array([-1, -1, 1, 1, 0, -1, -1, -1, 0, 0])
    gid, age, length, carry, near = gt_mod.gap_table(ball, half, gt, nearest)
    assert gid.tolist() == [0, 0, -1, -1, -1, 1, 1, 1, -1, -1]
    assert carry[:2].tolist() == [-1, -1]               # gap opens the half
    assert carry[5:8].tolist() == [A, A, A]             # last H/A before the dead frame 4
    assert near[5:8].tolist() == [0, 0, 0]
    assert age[5:8].tolist() == [0, 1, 2] and length[5:8].tolist() == [3, 3, 3]
    changed, post = gt_mod.change_masks(gid, gt, carry)
    assert changed.tolist() == [False] * 5 + [True] * 3 + [False] * 2
    assert post.tolist() == [False] * 6 + [True] * 2 + [False] * 2


def test_logistic_recovers_sign_and_separates():
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(2000, 3))
    y = (Xr[:, 0] - 0.5 * Xr[:, 1] + 0.1 * rng.normal(size=2000) > 0).astype(float)
    Z, mean, std = gt_mod.standardise(np.column_stack([Xr, np.full(2000, np.nan)]))
    assert np.all(Z[:, 3] == 0)
    w, b = gt_mod.fit_logistic(Z, y)
    assert w[0] > 0 and w[1] < 0 and abs(w[2]) < 0.5 * abs(w[0])
    assert np.mean(((Z @ w + b) > 0) == y) > 0.95


def test_smoothing_stays_inside_each_gap():
    values = np.array([10.0, 1, 1, -1, 5, 5, 5])
    gid = np.array([-1, 0, 0, 0, 1, 1, 1])
    out = gt_mod.smooth_in_gaps(values, gid, 3)
    assert out[0] == 10.0
    np.testing.assert_allclose(out[1:4], [1.0, 1 / 3, 0.0])
    np.testing.assert_allclose(out[4:], [5, 5, 5])


def test_buckets_half_open():
    edges = np.array([0, 2, 5, np.inf])
    assert gt_mod.bucket(np.array([0.0, 1.99, 2.0, 4.9, 5.0, 100.0]), edges).tolist() == \
        [0, 0, 1, 1, 2, 2]
