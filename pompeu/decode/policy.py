"""A10: the final label policy, applied to whatever a method produced, last of all.

    1. Zero X on graded frames. Writing our best label is never worse than X under either metric
       (EVALUATION.md: macroF1 counts X as a miss; spellF1 fills it with the previous label, which
       is our label anyway). Any X a method left inside a half is filled from the previous label
       of that half (the next one at a half's start).
    2. X outside the halves (before kick-off, half-time, after the final whistle). Ungraded, so it
       changes no score, but the file says what we actually claim.
    3. No possession or stoppage shorter than min_run_frames inside a half: one- or two-frame runs
       are artefacts of rounding between decisions, never football. They take the previous run's
       label (the next one's at a half's start).

Configured in a pipeline config as `policy: {}` (min_run_frames 3), which pipeline.predict
applies after the method.
"""

import numpy as np

# Params
PARAMS = {"min_run_frames": 3}


def finalize(labels, tracking, params=None):
    """(N,) H/A/D/X labels after the three rules above."""
    p = dict(PARAMS, **(params or {}))
    out = np.asarray(labels).astype("<U1").copy()
    frames = tracking.frames
    graded = np.zeros(len(out), dtype=bool)
    for _, first, last in tracking.half_blocks():
        rows = np.flatnonzero((frames >= first) & (frames <= last))
        if not len(rows):
            continue
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        graded[lo:hi] = True
        out[lo:hi] = _fill_and_smooth(out[lo:hi], p["min_run_frames"])
    out[~graded] = "X"
    return out


def _fill_and_smooth(labels, min_run):
    labels = labels.copy()
    known = np.flatnonzero(labels != "X")
    if not len(known):
        return np.full(len(labels), "D")
    carry = np.maximum.accumulate(np.where(labels != "X", np.arange(len(labels)), -1))
    carry[carry < 0] = known[0]
    labels = labels[carry]
    while True:
        change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [len(labels)]))
        short = [(a, b) for a, b in zip(starts, stops) if b - a < min_run]
        if not short or len(starts) == 1:
            return labels
        a, b = short[0]
        labels[a:b] = labels[a - 1] if a > 0 else labels[b]
