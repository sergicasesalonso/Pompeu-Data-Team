"""A1 ball cleaning: drop impossible points, drop isolated fragments, bridge short gaps.

Three steps, in this order, each returning a mask so the report can say what it did:

1. Spikes. A point that implies more than `vmax` (40 m/s; the hardest shots are ~35 m/s) both
   from its previous and to its next observed neighbour is a false detection -- a head, a boot, a
   penalty spot -- and is removed. Repeated until stable, so a spike next to a spike also goes.
2. Short fragments. The track is split at gaps and at remaining jumps (> vmax between consecutive
   observations); fragments shorter than `min_run` frames are removed. That covers "fewer than 3
   frames surrounded by gaps" and the 2-frame bursts that step 1 cannot see.
3. Short gaps. Gaps up to `interp_max_s` (0.5 s) with observations on both sides are filled with a
   cubic spline through up to `context` points each side -- linear if there are too few -- unless
   bridging the gap would itself need more than vmax. Filled frames are marked ball_source=interp
   with confidence `interp_conf`. Longer gaps are left alone: a guess there is worse than a NaN.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from pompeu.features.kinematics import runs

# Params
VMAX = 40.0
MIN_RUN = 3
INTERP_MAX_S = 0.5
INTERP_CONF = 0.5
CONTEXT = 4  # observed points used each side of a gap for the spline
MAX_PASSES = 10


def _neighbour_speeds(xy, index, fps):
    """Speed from each observed point to the previous and the next one (NaN at the ends)."""
    steps = np.hypot(*(xy[index[1:]] - xy[index[:-1]]).T) * fps / np.diff(index)
    return np.concatenate(([np.nan], steps)), np.concatenate((steps, [np.nan]))


def remove_spikes(xy, fps, vmax=VMAX):
    """Boolean (N,) of points removed as spikes. xy is not modified."""
    removed = np.zeros(len(xy), dtype=bool)
    for _ in range(MAX_PASSES):
        index = np.flatnonzero(~np.isnan(xy[:, 0]) & ~removed)
        if len(index) < 2:
            break
        before, after = _neighbour_speeds(xy, index, fps)
        # With one neighbour only (the ends of the track), that one decides.
        fast_before = np.where(np.isnan(before), after > vmax, before > vmax)
        fast_after = np.where(np.isnan(after), before > vmax, after > vmax)
        spikes = index[fast_before & fast_after]
        if not len(spikes):
            break
        removed[spikes] = True
    return removed


def remove_short_fragments(xy, fps, vmax=VMAX, min_run=MIN_RUN):
    """Boolean (N,) of points in fragments shorter than min_run, fragments split at gaps and jumps."""
    removed = np.zeros(len(xy), dtype=bool)
    index = np.flatnonzero(~np.isnan(xy[:, 0]))
    if not len(index):
        return removed
    _, after = _neighbour_speeds(xy, index, fps)
    # A fragment ends where the next observation is not the next frame, or is too far to reach.
    breaks = (np.diff(index) > 1) | (after[:-1] > vmax)
    bounds = np.concatenate(([0], np.flatnonzero(breaks) + 1, [len(index)]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < min_run:
            removed[index[lo:hi]] = True
    return removed


def interpolate_gaps(xy, fps, max_gap_s=INTERP_MAX_S, vmax=VMAX, context=CONTEXT):
    """(filled xy, boolean (N,) of interpolated frames)."""
    out = xy.copy()
    filled = np.zeros(len(xy), dtype=bool)
    present = ~np.isnan(xy[:, 0])
    starts, stops = runs(~present)
    max_gap = int(np.floor(max_gap_s * fps + 1e-9))
    for start, stop in zip(starts, stops):
        if start == 0 or stop == len(xy) or stop - start > max_gap:
            continue
        reach = np.hypot(*(xy[stop] - xy[start - 1])) * fps / (stop - start + 1)
        if reach > vmax:
            continue
        # Only the contiguous observed points touching the gap: never reach across another one.
        lo = max(0, start - context)
        missing = np.flatnonzero(~present[lo:start])
        before = np.arange(lo + (missing[-1] + 1 if len(missing) else 0), start)
        hi = min(len(xy), stop + context)
        missing = np.flatnonzero(~present[stop:hi])
        after = np.arange(stop, stop + missing[0] if len(missing) else hi)
        known = np.concatenate((before, after))
        gap = np.arange(start, stop)
        if len(known) >= 4:
            out[gap] = CubicSpline(known, xy[known], axis=0)(gap)
        else:
            for axis in (0, 1):
                out[gap, axis] = np.interp(gap, known, xy[known, axis])
        filled[gap] = True
    return out, filled


def clean_ball(xy, fps, vmax=VMAX, min_run=MIN_RUN, interp_max_s=INTERP_MAX_S):
    """(clean xy, masks dict) with masks 'spike', 'fragment', 'interp'."""
    spikes = remove_spikes(xy, fps, vmax)
    work = xy.copy()
    work[spikes] = np.nan
    fragments = remove_short_fragments(work, fps, vmax, min_run)
    work[fragments] = np.nan
    work, interp = interpolate_gaps(work, fps, interp_max_s, vmax)
    return work, {"spike": spikes, "fragment": fragments, "interp": interp}
