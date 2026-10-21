"""A3 check: the baseline's team vote, with D taken from the dead-ball detector instead of
"the ball is not visible".

The published baseline labels every frame without a ball D -- 41% of graded frames -- which is
both its D class and most of its possession splits. Here:

  1. A3 decides which stretches are dead (pompeu.events.deadball).
  2. Inside each LIVE stretch, the team is the majority vote of the nearest-player team over
     +/- `window` frames, counting only frames of that same stretch (a vote never reaches across a
     stoppage).
  3. Live frames with no vote in reach (the ball is missing for longer than the window) keep the
     last team of the stretch -- possession does not change while nobody sees the ball. The
     opening frames of a stretch, before the first vote, take the team the restart was given to,
     or the first vote if the restart has no team.

It is not rules_v0 (A4 adds touches and control); it isolates what A3 is worth.
"""

import numpy as np

from pompeu.events.deadball import detect
from pompeu.features.kinematics import runs
from pompeu.models.baseline import nearest_team

HOME, AWAY = 0, 1


def vote_in_stretch(codes, window):
    """(N,) team by majority vote within one live stretch; -1 where no vote is in reach."""
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    return np.where(h + a == 0, -1, np.where(h >= a, HOME, AWAY))


def predict(tracking, window=50, deadball=None):
    """(N,) H/A/D labels. `deadball` overrides pompeu.events.deadball.PARAMS."""
    result = detect(tracking, params=deadball)
    codes = nearest_team(tracking)
    restart_team = dict(zip((result.restarts["frame"] - tracking.frame0).tolist(),
                            result.restarts["team"].tolist()))
    teams = np.full(tracking.n_frames, -1, dtype=np.int64)
    starts, stops = runs(~result.dead)
    for start, stop in zip(starts, stops):
        voted = vote_in_stretch(codes[start:stop], int(window))
        known = np.flatnonzero(voted >= 0)
        if not len(known):
            opening = restart_team.get(int(start), -1)
            teams[start:stop] = opening
            continue
        # Forward fill inside the stretch; the head takes the restart team, else the first vote.
        carry = np.maximum.accumulate(np.where(voted >= 0, np.arange(len(voted)), -1))
        filled = np.where(carry >= 0, voted[np.maximum(carry, 0)], -1)
        opening = restart_team.get(int(start), -1)
        filled[:known[0]] = opening if opening >= 0 else voted[known[0]]
        teams[start:stop] = filled
    labels = np.where(teams == HOME, "H", np.where(teams == AWAY, "A", "D"))
    labels[result.dead] = "D"
    return labels
