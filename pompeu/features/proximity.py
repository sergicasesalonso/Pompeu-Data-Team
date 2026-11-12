"""A2 proximity: who is near the ball, by how much, and for how long.

All NaN on frames without a ball (the `ball_present` column says which). Distances in metres to
observed players only -- a slot with conf 0 is not there.

    d1_H d2_H d3_H, d1_A ...   distance to the 1st, 2nd, 3rd nearest player of each team
    d1_diff                    d1_H - d1_A: negative when home is closer
    nearest_team               0 home, 1 away
    near_streak_s_H/A          seconds the team has been nearest without interruption, up to now
    near_streak_fwd_s_H/A      seconds it will stay nearest from now (offline: looks ahead)
    n2_H n5_H n10_H, ...       players of the team within 2, 5 and 10 m
    near_speed_H/A             speed of that team's nearest player (Savitzky-Golay)
"""

import numpy as np

from pompeu.features.kinematics import runs

# Params
RANKS = (1, 2, 3)
RADII = (2, 5, 10)
TEAMS = (("H", 0), ("A", 1))


def ball_distances(tracking):
    """(N, K) distance of every observed player to the ball; NaN where either is missing."""
    delta = tracking.players_xy - tracking.ball_xy[:, None, :]
    distance = np.hypot(delta[..., 0], delta[..., 1])
    distance[tracking.players_conf <= 0] = np.nan
    return distance


def _streaks(flag, valid, fps):
    """(backward, forward) run length in seconds of True in `flag`, counted over `valid` frames.

    Frames that are not valid (no ball) neither extend nor break a streak; they get NaN.
    """
    backward = np.full(len(flag), np.nan)
    forward = np.full(len(flag), np.nan)
    index = np.flatnonzero(valid)
    if not len(index):
        return backward, forward
    compact = flag[index]
    starts, stops = runs(compact)
    back = np.zeros(len(compact))
    fwd = np.zeros(len(compact))
    for start, stop in zip(starts, stops):
        position = np.arange(stop - start)
        back[start:stop] = position + 1
        fwd[start:stop] = (stop - start) - position
    backward[index] = back / fps
    forward[index] = fwd / fps
    return backward, forward


def proximity_features(tracking, player_speed):
    """{column: (N,) array}. player_speed is (N, K) from kinematics.player_kinematics."""
    fps = tracking.meta.get("fps", 25)
    distance = ball_distances(tracking)
    team = tracking.players_team
    present = tracking.ball_present
    rows = np.arange(tracking.n_frames)
    out = {}
    nearest = {}

    for name, side in TEAMS:
        mine = np.where(team == side, distance, np.nan)
        filled = np.where(np.isnan(mine), np.inf, mine)
        order = np.argsort(filled, axis=1)
        ranked = np.take_along_axis(filled, order, axis=1)
        ranked[np.isinf(ranked)] = np.nan
        for rank in RANKS:
            out["d{}_{}".format(rank, name)] = ranked[:, rank - 1]
        for radius in RADII:
            with np.errstate(invalid="ignore"):
                count = np.sum(mine < radius, axis=1).astype(float)
            count[~present] = np.nan
            out["n{}_{}".format(radius, name)] = count
        speed = player_speed[rows, order[:, 0]]
        speed[np.isnan(ranked[:, 0])] = np.nan
        out["near_speed_" + name] = speed
        nearest[side] = ranked[:, 0]

    out["d1_diff"] = nearest[0] - nearest[1]
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(out["d1_diff"])
        nearest_team = np.where(out["d1_diff"] <= 0, 0.0, 1.0)
    nearest_team[~valid] = np.nan
    out["nearest_team"] = nearest_team
    for name, side in TEAMS:
        out["near_streak_s_" + name], out["near_streak_fwd_s_" + name] = _streaks(
            nearest_team == side, valid, fps)
    return out
