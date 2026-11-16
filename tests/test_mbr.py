"""A9: the fast spellF1 must be the challenge's spellF1, and MBR must never lower the expectation."""

import numpy as np
import pytest

from pompeu import paths  # noqa: F401
from pompeu.decode import mbr
from pompeu.decode import viterbi as sm
import possession_metrics as pm


def random_labels(rng, n=3000):
    runs = []
    while sum(len(r) for r in runs) < n:
        runs.append(np.full(int(rng.integers(1, 200)), rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])))
    return np.concatenate(runs)[:n].astype(np.int8)


@pytest.mark.parametrize("seed", range(30))
def test_fast_spell_f1_equals_possession_metrics(seed):
    rng = np.random.default_rng(seed)
    truth, pred = random_labels(rng), random_labels(rng)
    if seed % 3 == 0:  # also a prediction close to the truth: many high-IoU matches and ties
        pred = truth.copy()
        flips = rng.choice(len(pred), 40, replace=False)
        pred[flips] = rng.integers(0, 3, 40)
    runs = [(0, len(truth))]
    expected = pm.spell_scores(truth, pred, runs)["mean_f1"]
    assert mbr.spell_f1(truth, pred) == pytest.approx(expected, abs=1e-12)


def test_spells_of_labels():
    starts, stops, teams = mbr.spells(np.array([2, 0, 0, 1, 2, 2, 1, 1]))
    assert starts.tolist() == [1, 3, 6] and stops.tolist() == [3, 4, 8] and teams.tolist() == [0, 1, 1]


def test_mbr_never_lowers_the_expected_score():
    rng = np.random.default_rng(0)
    n = 1500
    emissions = rng.normal(0, 0.4, (n, 3))
    emissions[200:600, 0] += 0.8
    emissions[600:640, 1] += 0.6          # a short, uncertain away possession
    emissions[640:1100, 0] += 0.8
    emissions[1100:, 2] += 0.8
    d_max = 600
    log_dur = np.full((3, d_max + 1), -np.inf)
    log_dur[:, 1:] = -np.log(d_max)
    trans = np.full((n + 1, 3, 3), np.log(0.5))
    for s in range(3):
        trans[:, s, s] = -30.0
    start = np.log(np.full(3, 1 / 3))
    labels, before, after = mbr.decode_half(sm.cumulative(emissions), log_dur, trans, start,
                                            np.array([0, 1, 2], dtype=np.int8),
                                            {"n_samples": 100})
    assert after >= before
    assert labels.shape == (n,) and set(np.unique(labels)) <= {0, 1, 2}


# -- A10: the final label policy ---------------------------------------------------------------

def test_policy_fills_x_smooths_and_marks_ungraded():
    from pompeu.decode.policy import finalize
    from synthetic import make_tracking
    n = 40
    tr = make_tracking(np.zeros((n, 1, 2)), np.array([0]), np.zeros((n, 2)), frame0=11170)
    labels = np.array(list("D" * 15 + "XH" + "H" * 10 + "A" + "H" * 5 + "AA" + "D" * 5))
    out = finalize(labels, tr)
    assert set(out[:15]) == {"X"}                        # before the 1st half (11185)
    graded = out[15:]
    assert "X" not in graded                             # the X after kick-off is filled
    assert "A" not in graded[:18]                        # the 1-frame and 2-frame A runs are gone
    assert graded[0] == "H"                              # a half's leading X takes the next label
