"""A2 collective features: team shape and motion. They need no ball, so they exist on every frame.

This is the group that has to carry the 41% of graded frames where the ball is missing: a team
pushing up and spreading out is attacking, a team dropping and narrowing is defending, and when
every player slows down at once the ball is probably dead.

Per team (suffix _H / _A), measured in that team's attacking frame (pompeu.clean.normalize: it
always attacks towards +x) so the same number means the same thing for both teams and both halves:

    n_vis            observed players
    cx, cy           centroid; cx is how far up the pitch the team is
    cvx, cvy         centroid velocity (mean of player velocities); cvx > 0 is moving forward
    width, depth     spread across (y) and along (x) the pitch, outfield only (deepest player out)
    def_line         x of the last outfield defender (second-deepest player)
    front            x of the most advanced player
    compact          mean distance to the team centroid
    n_recover        players running back towards their own goal faster than RUN_SPEED
    n_press          players running towards the opponent's goal faster than RUN_SPEED
    speed_mean       mean player speed

Global: field_speed_mean / field_speed_median of all players (the dead-ball fingerprint),
stopped_frac (share under STOP_SPEED), centroid_gap (home minus away centroid, in the home frame),
line_gap (home def_line minus the most advanced away player, in the home frame).
"""

import warnings

import numpy as np

from pompeu.clean.normalize import attack_flip

# Params
RUN_SPEED = 2.0   # m/s along x to count as recovering or pressing
STOP_SPEED = 1.0  # m/s


def _nan(function, values, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(values, axis=axis)


def _kth_smallest(values, k):
    """k-th smallest (1-based) along axis 1, ignoring NaN; NaN if fewer than k values."""
    ordered = np.sort(np.where(np.isnan(values), np.inf, values), axis=1)
    if ordered.shape[1] < k:
        return np.full(len(values), np.nan)
    out = ordered[:, k - 1]
    return np.where(np.isinf(out), np.nan, out)


def collective_features(tracking, player_vel, player_speed):
    """{column: (N,)}. player_vel (N, K, 2) and player_speed (N, K) from player_kinematics."""
    length, width = tracking.meta["pitch"]
    observed = tracking.players_conf > 0
    xy = np.where(observed[..., None], tracking.players_xy, np.nan)
    vel = np.where(observed[..., None], player_vel, np.nan)
    speed = np.where(observed, player_speed, np.nan)
    out = {}
    centroid_home = {}

    for name, side in (("H", 0), ("A", 1)):
        mine = tracking.players_team == side
        flip = attack_flip(tracking, side)[:, None]
        x = np.where(mine, np.where(flip, length - xy[..., 0], xy[..., 0]), np.nan)
        y = np.where(mine, np.where(flip, width - xy[..., 1], xy[..., 1]), np.nan)
        vx = np.where(mine, np.where(flip, -vel[..., 0], vel[..., 0]), np.nan)
        vy = np.where(mine, np.where(flip, -vel[..., 1], vel[..., 1]), np.nan)

        cx, cy = _nan(np.nanmean, x), _nan(np.nanmean, y)
        out["n_vis_" + name] = np.sum(~np.isnan(x), axis=1).astype(float)
        out["cx_" + name], out["cy_" + name] = cx, cy
        out["cvx_" + name], out["cvy_" + name] = _nan(np.nanmean, vx), _nan(np.nanmean, vy)

        deepest = _kth_smallest(x, 1)
        outfield = np.where(x == deepest[:, None], np.nan, x)
        outfield_y = np.where(np.isnan(outfield), np.nan, y)
        out["width_" + name] = _nan(np.nanmax, outfield_y) - _nan(np.nanmin, outfield_y)
        out["depth_" + name] = _nan(np.nanmax, outfield) - _nan(np.nanmin, outfield)
        out["def_line_" + name] = _kth_smallest(x, 2)
        out["front_" + name] = _nan(np.nanmax, x)
        out["compact_" + name] = _nan(np.nanmean, np.hypot(x - cx[:, None], y - cy[:, None]))
        with np.errstate(invalid="ignore"):
            out["n_recover_" + name] = np.sum(vx < -RUN_SPEED, axis=1).astype(float)
            out["n_press_" + name] = np.sum(vx > RUN_SPEED, axis=1).astype(float)
        out["speed_mean_" + name] = _nan(np.nanmean, np.where(mine, speed, np.nan))

        # Centroid in the HOME frame, for the between-team gap below. Away's frame is the home
        # frame turned round, so its x converts as length - x.
        centroid_home[side] = cx if side == 0 else length - cx

    out["field_speed_mean"] = _nan(np.nanmean, speed)
    out["field_speed_median"] = _nan(np.nanmedian, speed)
    with np.errstate(invalid="ignore"):
        stopped = np.sum(speed < STOP_SPEED, axis=1) / np.maximum(np.sum(~np.isnan(speed), axis=1), 1)
    stopped[np.all(np.isnan(speed), axis=1)] = np.nan
    out["stopped_frac"] = stopped
    out["centroid_gap"] = centroid_home[0] - centroid_home[1]
    # Home back line minus the most advanced away player, both in the home frame (away's front is
    # length - front_A there). Near zero or positive: away has players level with or behind it.
    out["line_gap"] = out["def_line_H"] - (length - out["front_A"])
    return out
