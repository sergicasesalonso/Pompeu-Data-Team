"""The published baseline, on a Tracking: nearest player to the ball, then a majority vote.

Same algorithm as starter/baseline.py, vectorised and reading the contract instead of the raw
file, so it runs on any Tracking -- the provided ATD, a stressed copy, Line B's ATD+. On the
provided ATD it reproduces the committed evaluation/results/baseline_*.csv byte for byte
(tests/test_baseline.py), which makes it the reference every later stage is compared against
(A1: "the cleaned tracking must match or beat the baseline").

The smoothing is starter.baseline.smooth itself, imported rather than copied.
"""

import numpy as np

from pompeu import paths  # noqa: F401
from starter.baseline import smooth

NOBODY = -1


def nearest_team(tracking):
    """(N,) int8: team of the nearest observed player to the ball, NOBODY without ball or players.

    Distances are computed in the same order and precision as starter/baseline.py
    (dx*dx + dy*dy in float64 metres) and ties go to the lowest slot, as there.
    """
    dx = tracking.players_xy[:, :, 0] - tracking.ball_xy[:, None, 0]
    dy = tracking.players_xy[:, :, 1] - tracking.ball_xy[:, None, 1]
    distance = dx * dx + dy * dy
    usable = ~np.isnan(distance)
    distance[~usable] = np.inf
    slot = np.argmin(distance, axis=1)

    rows = np.arange(tracking.n_frames)
    out = tracking.players_team[rows, slot].astype(np.int8)
    out[~usable.any(axis=1)] = NOBODY
    return out


def predict(tracking, window=50):
    """(N,) H/A/D labels. `window` is the vote half-width in frames; 0 disables smoothing."""
    return smooth(nearest_team(tracking), int(window))
