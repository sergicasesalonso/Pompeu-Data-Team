"""Velocity and acceleration from positions, with Savitzky-Golay inside each observed run.

The ATD's own speed column is empty on most frames (and the ball's on ~91%), so speeds are
recomputed here from positions. A Savitzky-Golay filter (window 7, order 2) differentiates a local
quadratic fit, which smooths the tracker's jitter without the lag of a moving average.

The filter runs inside each maximal run of observed frames and never across a gap: differentiating
across a hole would invent a velocity from two unrelated observations. Runs shorter than the window
use the largest odd window that fits; a run of two frames gets a plain difference; a single frame
gets NaN.
"""

import warnings

import numpy as np
from scipy.signal import savgol_filter

from pompeu.io.contract import SRC_INTERP

# Params
WINDOW = 7
ORDER = 2


def runs(mask):
    """(starts, stops) of the maximal True runs of a 1-D boolean array, stops exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def savgol_kinematics(xy, fps, window=WINDOW, order=ORDER):
    """(velocity, acceleration), each (N, 2), m/s and m/s^2. NaN wherever xy is NaN."""
    velocity = np.full(xy.shape, np.nan)
    acceleration = np.full(xy.shape, np.nan)
    starts, stops = runs(~np.isnan(xy[:, 0]))
    delta = 1.0 / fps
    for start, stop in zip(starts, stops):
        segment = xy[start:stop]
        length = stop - start
        if length == 1:
            continue
        if length == 2:
            velocity[start:stop] = (segment[1] - segment[0]) * fps
            continue
        w = min(window, length if length % 2 else length - 1)
        p = min(order, w - 1)
        velocity[start:stop] = savgol_filter(segment, w, p, deriv=1, delta=delta, axis=0,
                                             mode="interp")
        if p >= 2:
            acceleration[start:stop] = savgol_filter(segment, w, p, deriv=2, delta=delta,
                                                     axis=0, mode="interp")
    return velocity, acceleration


def _describe(velocity, acceleration):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "vel": velocity,
            "speed": np.hypot(velocity[..., 0], velocity[..., 1]),
            "acc": acceleration,
            "acc_norm": np.hypot(acceleration[..., 0], acceleration[..., 1]),
            # Direction of motion in radians, atan2(vy, vx); NaN where there is no velocity.
            "heading": np.arctan2(velocity[..., 1], velocity[..., 0]),
        }


def ball_kinematics(tracking, window=WINDOW, order=ORDER):
    """{'vel', 'speed', 'acc', 'acc_norm', 'heading'} for the ball, one row per frame."""
    fps = tracking.meta.get("fps", 25)
    return _describe(*savgol_kinematics(tracking.ball_xy, fps, window, order))


def player_kinematics(tracking, window=WINDOW, order=ORDER):
    """Same keys as ball_kinematics, shaped (N, K, ...) -- per slot, per observed run."""
    fps = tracking.meta.get("fps", 25)
    velocity = np.full(tracking.players_xy.shape, np.nan)
    acceleration = np.full(tracking.players_xy.shape, np.nan)
    for slot in range(tracking.n_slots):
        velocity[:, slot], acceleration[:, slot] = savgol_kinematics(
            tracking.players_xy[:, slot], fps, window, order)
    return _describe(velocity, acceleration)


# -- A2: ball state ----------------------------------------------------------------------------

STILL_SPEED = 0.5   # m/s
STILL_MIN_S = 1.0
OUT_MARGIN = 0.5    # metres beyond a line before the ball counts as out
TURN_LAG = 2        # frames either side for the change of direction
TURN_MIN_SPEED = 1.0
PENALTY_SPOT = 11.0


def _gap_times(present, fps):
    """(age, left, length) in seconds of the no-ball gap each frame is in; 0 where present."""
    n = len(present)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(present, index, -1))
    following = np.minimum.accumulate(np.where(present, index, n)[::-1])[::-1]
    age = np.where(last >= 0, index - last, np.nan).astype(float)
    left = np.where(following < n, following - index, np.nan).astype(float)
    return age / fps, left / fps, (age + left) / fps


def ball_features(tracking, kin, flip_home):
    """{column: (N,)}: kinematics, stillness, geometry and gap timing of the ball.

    `kin` is ball_kinematics(tracking); flip_home is normalize.attack_flip(tracking, 0), used to
    express the ball's x as progress towards the goal the home side attacks.
    """
    fps = tracking.meta.get("fps", 25)
    length, width = tracking.meta["pitch"]
    xy = tracking.ball_xy
    x, y = xy[:, 0], xy[:, 1]
    present = tracking.ball_present
    out = {
        "ball_present": present.astype(float),
        "ball_interp": (tracking.ball_source == SRC_INTERP).astype(float),
        "ball_speed": kin["speed"],
        "ball_acc": kin["acc_norm"],
    }

    vel = kin["vel"]
    before = np.full(vel.shape, np.nan)
    after = np.full(vel.shape, np.nan)
    before[TURN_LAG:] = vel[:-TURN_LAG]
    after[:-TURN_LAG] = vel[TURN_LAG:]
    with np.errstate(invalid="ignore"):
        cosine = np.sum(before * after, axis=1) / (np.hypot(*before.T) * np.hypot(*after.T))
        turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        turn[(np.hypot(*before.T) < TURN_MIN_SPEED) | (np.hypot(*after.T) < TURN_MIN_SPEED)] = np.nan
    out["ball_turn_deg"] = turn

    with np.errstate(invalid="ignore"):
        slow = kin["speed"] < STILL_SPEED
    still_for = np.full(len(xy), np.nan)
    still = np.zeros(len(xy))
    still_for[present] = 0.0
    starts, stops = runs(slow)
    for start, stop in zip(starts, stops):
        still_for[start:stop] = (np.arange(stop - start) + 1) / fps
        if stop - start >= STILL_MIN_S * fps:
            still[start:stop] = 1.0
    still[~present] = np.nan
    out["ball_still"] = still
    out["ball_still_s"] = still_for

    edges = np.column_stack((x, length - x, y, width - y))
    with np.errstate(invalid="ignore"):
        out["ball_out"] = np.where(present, (edges.min(axis=1) < -OUT_MARGIN).astype(float), np.nan)
    out["ball_to_goal_line_min"] = np.minimum(edges[:, 0], edges[:, 1])
    out["ball_to_touchline_min"] = np.minimum(edges[:, 2], edges[:, 3])
    out["ball_edge_dist"] = edges.min(axis=1)
    out["ball_dist_centre"] = np.hypot(x - length / 2, y - width / 2)
    corners = [np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)]
    out["ball_dist_corner"] = np.min(corners, axis=0)
    out["ball_dist_penalty"] = np.minimum(np.hypot(x - PENALTY_SPOT, y - width / 2),
                                          np.hypot(x - (length - PENALTY_SPOT), y - width / 2))

    # Position in the frame where the home side attacks +x (pompeu.clean.normalize), and the last
    # known x (works without a ball: where it was last seen is still evidence of who attacks).
    x_home = np.where(flip_home, length - x, x)
    out["ball_x_home"] = x_home
    out["ball_y_home"] = np.where(flip_home, width - y, y)
    last = np.maximum.accumulate(np.where(present, np.arange(len(xy)), -1))
    held = np.where(last >= 0, x_home[np.maximum(last, 0)], np.nan)
    out["ball_x_home_last"] = held
    out["ball_gap_age_s"], out["ball_gap_left_s"], out["ball_gap_len_s"] = _gap_times(present, fps)
    return out
