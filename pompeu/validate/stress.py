"""public_stress: the public window with the ball erased the way the rest of the match loses it.

The public 5 minutes are unusually clean -- the ball is missing on ~12% of them against ~41% of
all graded frames -- so a method scored there is scored on easy data. This bench erases the ball
inside the window, in runs whose lengths are drawn from the gaps the ATD actually has (measured,
not assumed: median ~10.7 s, p99 ~105 s), until a target fraction of the window has no ball.

Gaps are placed uniformly. Placing them where the real ATD loses the ball (dead balls, crowded
boxes) is the degradation simulator's job (A6, external/degrade.py); this bench only asks "does
the method survive the missing ball", and keeps the answer comparable across configs because the
masks depend on the seed alone.
"""

import numpy as np

from pompeu import paths  # noqa: F401
import games

# Params
DEFAULT_LEVELS = (0.3, 0.5, 0.7)
DEFAULT_SEEDS = 20
# Cap on a sampled gap. The measured tail reaches ~105 s at p99; the few longer runs are kickoff
# and restart waits that would erase a third of a 5-minute window in one draw.
DEFAULT_MAX_GAP_S = 105.0


def missing_run_lengths(tracking, game=games.GAME):
    """Lengths in frames of every no-ball run strictly inside a half.

    Runs touching a half's first or last frame are dropped: they are cut by the half boundary, not
    by the tracker, so their length says nothing about how the ball goes missing.
    """
    present = tracking.ball_present
    lengths = []
    for _, first, last in games.half_video_blocks(game):
        lo, hi = first - tracking.frame0, last + 1 - tracking.frame0
        if lo < 0 or hi > tracking.n_frames:
            continue
        missing = ~present[lo:hi]
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        inside = (starts > 0) & (stops < len(missing))
        lengths.extend((stops - starts)[inside].tolist())
    return np.array(lengths, dtype=np.int64)


def stress_mask(present, target, run_lengths, rng):
    """Boolean mask over `present` of frames to erase so that a `target` fraction has no ball.

    Runs are drawn from run_lengths and dropped at uniform positions (a run may hang over either
    edge and is clipped). The last run is trimmed so the fraction lands on the target exactly.
    Frames already missing count towards the target; if there are already more, nothing is erased.
    """
    n = len(present)
    missing = ~np.asarray(present, dtype=bool)
    erase = np.zeros(n, dtype=bool)
    goal = int(round(target * n))
    while missing.sum() < goal:
        length = int(rng.choice(run_lengths))
        start = int(rng.integers(-length + 1, n))
        span = np.arange(max(start, 0), min(start + length, n))
        new = span[~missing[span]][:goal - int(missing.sum())]
        missing[new] = True
        erase[new] = True
    return erase


def stressed_trackings(tracking, window, levels=DEFAULT_LEVELS, seeds=DEFAULT_SEEDS,
                       max_gap_s=DEFAULT_MAX_GAP_S, run_lengths=None):
    """Yield (level, seed, achieved_missing_fraction, stressed Tracking) for every combination.

    `tracking` is whatever the pipeline will see (normally the window plus context); only frames
    inside `window` = (start, stop), absolute and half-open, are ever erased. run_lengths should
    be measured on the full match -- the default measures it on `tracking`, which only works if
    that is the full match.
    """
    if run_lengths is None:
        run_lengths = missing_run_lengths(tracking)
    fps = tracking.meta.get("fps", 25)
    run_lengths = run_lengths[run_lengths <= max_gap_s * fps]
    if not len(run_lengths):
        raise ValueError("no ball gaps to sample from")

    lo, hi = window[0] - tracking.frame0, window[1] - tracking.frame0
    present = tracking.ball_present[lo:hi]
    for level in levels:
        for seed in range(seeds):
            # One generator per (level, seed), so adding a level never changes another's masks.
            rng = np.random.default_rng([seed, int(round(level * 1000))])
            erase = np.zeros(tracking.n_frames, dtype=bool)
            erase[lo:hi] = stress_mask(present, level, run_lengths, rng)
            stressed = tracking.without_ball(erase)
            yield level, seed, float((~stressed.ball_present[lo:hi]).mean()), stressed
