"""scripts/experiments/self_train_gap.py pure logic, on synthetic arrays."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import self_train_gap as stg  # noqa: E402
from pompeu.experts import gap_team  # noqa: E402

H, A, D = stg.H, stg.A, 2


def test_long_runs_respects_length_and_halves():
    codes = np.array([H, H, H, A, A, D, H, H, H, H])
    half = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
    out = stg.long_runs(codes, half, 3)
    assert out.tolist() == [True] * 3 + [False] * 7       # H H | H H split by half: too short


def test_pseudo_labels_agreement_ball_exclusion_and_tagging():
    rules = np.array([H] * 6 + [A] * 6)
    hmm = rules.copy()
    hmm[0] = A                                             # disagreement breaks the first run
    ball = np.ones(12, bool)
    ball[7] = False
    half = np.ones(12, int)
    exclude = np.zeros(12, bool)
    exclude[11] = True
    lab, st = stg.pseudo_labels(rules, hmm, ball, half, 4, exclude=exclude)
    assert lab.tolist() == [-1, H, H, H, H, H, A, -1, A, A, A, -1]
    assert st["a"] == 9 and st["labelled"] == 9
    vote = np.zeros(12)
    vote[7] = -1                                           # fills the no-ball frame (a absent)
    vote[1] = -1                                           # conflicts with (a): dropped
    vote[2] = 1                                            # agrees
    lab2, st2 = stg.pseudo_labels(rules, hmm, ball, half, 4, vote, exclude)
    assert lab2[7] == A and lab2[1] == -1 and lab2[2] == H
    assert st2["overlap"] == 2 and st2["conflict"] == 1 and st2["ab_agreement"] == 0.5


def test_mass_weights_share_and_total():
    we, wm = stg.mass_weights(300, 100, 0.5)
    assert np.isclose(we.sum(), wm.sum()) and np.isclose(we.sum() + wm.sum(), 400)
    we, wm = stg.mass_weights(10, 0, 0.5)
    assert np.all(we == 1)


def test_weighted_fit_matches_unweighted_with_unit_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(float)
    Z, mean, std = gap_team.standardise(X)
    Zw, mw, sw = stg.weighted_standardise(X, np.ones(400))
    assert np.allclose(mean, mw) and np.allclose(std, sw)
    w0, b0 = gap_team.fit_logistic(Z, y)
    w1, b1 = stg.fit_weighted_logistic(Zw, y, np.ones(400))
    assert np.allclose(w0, w1, atol=1e-8) and np.isclose(b0, b1, atol=1e-8)
    # Doubling every weight equals duplicating every row.
    w2, _ = stg.fit_weighted_logistic(Zw, y, np.full(400, 2.0))
    w3, _ = gap_team.fit_logistic(np.vstack([Z, Z]), np.concatenate([y, y]))
    assert np.allclose(w2, w3, atol=1e-6)
