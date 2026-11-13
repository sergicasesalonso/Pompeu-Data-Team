"""scripts/coverage_sweep.py: the nested ball-missing ladder, the summaries and the slopes."""

import importlib.util
import os

import numpy as np
import pytest

from pompeu.external import degrade
from synthetic import make_tracking

FPS = 25
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                     "coverage_sweep.py")
_spec = importlib.util.spec_from_file_location("coverage_sweep", _PATH)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def clean_match(n=6000):
    rng = np.random.default_rng(0)
    t = np.arange(n) / FPS
    start = rng.uniform(10, 95, (22, 2)) * (1.0, 0.6)
    players = start[None] + 3 * np.sin(t[:, None, None] / 3 + rng.uniform(0, 6, (1, 22, 2)))
    tr = make_tracking(players, np.array([0] * 11 + [1] * 11), players[:, 0] + 0.5)
    tr.meta.update(halves=[(1, 0, n // 2 - 1), (2, n // 2, n - 1)], source="test")
    labels = np.where((t % 60) < 20, 2, 0).astype(np.int8)
    return tr, labels


def test_nested_ladder_hits_each_rate_and_masks_nest():
    tr, labels = clean_match()
    lengths = np.array([25, 100, 250])
    base = degrade.degrade(tr, labels, 0, lengths, {"ball_missing": 0.2})
    ladder = sweep.nested_ladder(base, labels, [0.5, 0.2, 0.41, 0.3], lengths,
                                 np.random.default_rng(1))
    previous = None
    for rate in sorted(ladder):
        tracking = ladder[rate]
        assert sweep.missing_fraction(tracking) == pytest.approx(rate, abs=1e-3)
        np.testing.assert_array_equal(tracking.players_xy, base.players_xy)  # only the ball moves
        if previous is not None:
            assert not (tracking.ball_present & ~previous.ball_present).any()  # nothing comes back
            seen = tracking.ball_present
            np.testing.assert_array_equal(tracking.ball_xy[seen], previous.ball_xy[seen])
        previous = tracking


def test_synthetic_run_lengths_match_median_and_p99():
    lengths = sweep.synthetic_run_lengths(fps=FPS)
    assert np.median(lengths) / FPS == pytest.approx(sweep.GAP_MEDIAN_S, rel=0.03)
    assert np.percentile(lengths, 99) / FPS == pytest.approx(sweep.GAP_P99_S, rel=0.1)


def test_slope_is_per_point_of_coverage():
    scores = {0.41: 0.30, 0.37: 0.32, 0.30: 0.355}
    assert sweep.slope_per_point(scores, 0.41, 0.37) == pytest.approx(0.005)
    assert sweep.slope_per_point(scores, 0.41, 0.30) == pytest.approx(0.005)


def test_summarise():
    mean, lo, hi = sweep.summarise(np.arange(101))
    assert (mean, lo, hi) == (50.0, 5.0, 95.0)
