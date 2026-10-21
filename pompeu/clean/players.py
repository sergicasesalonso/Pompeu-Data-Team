"""A1 player cleaning: frozen positions and duplicated tracks.

FROZEN, NOT STATIC OBJECTS. The plan assumed slots 24 and 46 were fixed objects to delete. They
are not: they are the two goalkeepers (slot 24 sits at x ~ 11 m in the 1st half and ~ 94 m in the
2nd, slot 46 the mirror image -- exactly Barcelona and Athletic swapping ends), present on 90% of
frames. What does happen is that the tracker *holds* a position when it loses someone, mostly a
goalkeeper outside the tactical camera's view: the slot stops dead (< 0.2 m in 10 s, often with
exactly zero steps) while every other player keeps moving. No other slot does this more than twice
in the match.

So a frozen stretch is a stale observation of a real player, not noise to throw away. By default it
is kept with a low confidence (`action: flag`) so later stages can discount it; `action: drop`
removes it instead. The rule is general -- any slot, any time -- not a hard-coded slot list.

DUPLICATES. Two slots of the same team closer than 0.5 m for more than 1 s are the tracker seeing
one player twice. For each such stretch the slot whose tracklet is shorter loses those frames.
"""

import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from pompeu.features.kinematics import runs

# Params
FROZEN_WINDOW_S = 10.0
FROZEN_MAX_DISP_M = 0.2
FIELD_MIN_SPEED = 1.0  # m/s, median player speed above which "the rest are moving"
FROZEN_CONF = 0.3
DUP_MAX_DIST_M = 0.5
DUP_MIN_S = 1.0


def field_speed(xy, fps):
    """(N,) median over observed slots of the frame-to-frame speed. NaN where nobody is seen."""
    step = np.full(xy.shape[:2], np.nan)
    step[1:] = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1)) * fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(step, axis=1)


def frozen_mask(xy, fps, window_s=FROZEN_WINDOW_S, max_disp=FROZEN_MAX_DISP_M,
                field_min_speed=FIELD_MIN_SPEED):
    """Boolean (N, K): frames inside a window where the slot moved < max_disp while others ran.

    The window range is computed with running max/min filters (O(N) per slot). A NaN inside a
    window makes its range infinite, so only fully observed windows can be frozen.
    """
    n, k = xy.shape[:2]
    width = int(round(window_s * fps))
    speed = field_speed(xy, fps)
    seen = ~np.isnan(speed)
    # Mean of the field speed over each window, ignoring frames where nobody is seen.
    total = uniform_filter1d(np.where(seen, speed, 0.0), width, mode="constant")
    count = uniform_filter1d(seen.astype(float), width, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        moving = (total / count) > field_min_speed

    out = np.zeros((n, k), dtype=bool)
    for slot in range(k):
        x, y = xy[:, slot, 0], xy[:, slot, 1]
        if np.isnan(x).all():
            continue
        ranges = []
        for values in (x, y):
            high = maximum_filter1d(np.where(np.isnan(values), np.inf, values), width,
                                    mode="constant", cval=np.inf)
            low = minimum_filter1d(np.where(np.isnan(values), -np.inf, values), width,
                                   mode="constant", cval=-np.inf)
            ranges.append(high - low)
        with np.errstate(invalid="ignore"):
            centre = (np.hypot(*ranges) < max_disp) & moving
        # A window centred on i covers [i - width//2, i + width - width//2): spread the flag to it.
        out[:, slot] = maximum_filter1d(centre.astype(np.int8), width, mode="constant") > 0
        out[:, slot] &= ~np.isnan(x)
    return out


def tracklet_length(present):
    """(N, K) int: length in frames of the observed run each frame belongs to, 0 where absent."""
    out = np.zeros(present.shape, dtype=np.int64)
    for slot in range(present.shape[1]):
        starts, stops = runs(present[:, slot])
        for start, stop in zip(starts, stops):
            out[start:stop, slot] = stop - start
    return out


def duplicate_mask(xy, team, fps, max_dist=DUP_MAX_DIST_M, min_s=DUP_MIN_S):
    """(boolean (N, K) of frames to drop, number of duplicate stretches found).

    `team` is (K,): one team per slot, which the ATD guarantees.
    """
    n, k = xy.shape[:2]
    present = ~np.isnan(xy[:, :, 0])
    length = tracklet_length(present)
    min_frames = int(round(min_s * fps))
    drop = np.zeros((n, k), dtype=bool)
    stretches = 0
    for a in range(k):
        if not present[:, a].any():
            continue
        for b in range(a + 1, k):
            if team[a] != team[b] or team[a] < 0 or not present[:, b].any():
                continue
            with np.errstate(invalid="ignore"):
                close = np.hypot(*(xy[:, a] - xy[:, b]).T) < max_dist
            starts, stops = runs(close)
            for start, stop in zip(starts, stops):
                if stop - start < min_frames:
                    continue
                stretches += 1
                shorter = a if length[start:stop, a].mean() < length[start:stop, b].mean() else b
                drop[start:stop, shorter] = True
    return drop, stretches
