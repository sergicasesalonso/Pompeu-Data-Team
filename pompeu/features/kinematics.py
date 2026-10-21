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
