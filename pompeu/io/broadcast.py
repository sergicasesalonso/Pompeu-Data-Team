"""B8: put the broadcast recording on the tactical timeline, from player positions alone.

    from pompeu.io import broadcast
    al = broadcast.align(tactical, bcast, bcast_valid)
    al.b_of_t      # (N_tactical,) broadcast frame shown at each tactical frame, -1 if none
    al.fit_of_t    # (N_tactical,) median metres between the broadcast players and the tactical ones

Both ATDs are in pitch metres, so the same instant looks the same in both: the players the
broadcast camera sees sit where the tactical tracker has players of the same team. The ball is
deliberately NOT used to align, so ball agreement stays an independent check of the alignment.

How it works:

1. Segments. The broadcast is cut into continuous shots: frames with a valid homography and at
   least `min_players` visible, split wherever the slots stop continuing (a camera cut resets the
   tracker) or jump (median slot displacement above `cut_jump_m` in one frame).
2. Global offsets. A TV feed is real time, so while it shows live play the offset t - b is
   constant. Sampled broadcast frames vote for the tactical frame they match best (coarse search
   over the whole match, one candidate per second); the modes of t - b are the live offsets.
3. Per segment. Each segment is fitted at every live offset with a fine search (+/- `fine_window`
   frames). A good fit makes it live. A segment that fits no live offset is searched freely; if it
   matches a moment already covered by a live segment it is a replay and is discarded.
4. Per frame. Every frame of an accepted segment is scored at the chosen offset (fit_of_t); the
   caller decides which fit is good enough.

The match score between one broadcast frame and one tactical frame is a one-sided chamfer
distance: for every visible broadcast player, the distance to the nearest tactical player of the
same team; the median over the broadcast players. One-sided, because the broadcast sees a subset.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

# Params
PARAMS = {
    "min_players": 6,        # visible broadcast players for a frame to be usable
    "min_segment": 25,       # frames; shorter shots are ignored
    "cut_continue": 0.5,     # fraction of slots that must carry over to the next frame
    "cut_jump_m": 2.0,       # median one-frame slot displacement that marks a cut
    "vote_every": 100,       # broadcast frames between two voting samples
    "coarse_stride": 25,     # tactical frames between two candidates in the coarse search
    "vote_max_m": 2.5,       # a vote counts only if its best match is this close
    "offset_bin": 25,        # frames per histogram bin for the offset modes
    "offset_min_share": 0.03,  # a mode needs this share of the votes
    "fine_window": 40,       # +/- frames around an offset in the fine search
    "segment_samples": 12,   # frames sampled per segment for the fine search
    "segment_max_m": 2.0,    # median fit for a segment to be accepted
    "replay_overlap": 0.5,   # share of a freely matched segment already covered -> replay
}


@dataclass
class Alignment:
    b_of_t: np.ndarray                  # (N_t,) int64, -1 where no broadcast frame is aligned
    fit_of_t: np.ndarray                # (N_t,) float64 metres, NaN where not aligned
    orientation: tuple                  # (flip_xy, swap_teams) chosen automatically
    offsets: list                       # live offsets t - b found by the vote
    stats: dict = field(default_factory=dict)   # counts only


def _slot_team(tracking):
    return np.asarray(tracking.players_team[0], dtype=np.int8)


def _chamfer(bpts, bteam, T, tteam):
    """Median over broadcast players of the distance to the nearest same-team tactical player.

    bpts (nb, 2), bteam (nb,), T (M, K, 2) candidate tactical frames, tteam (K,). Returns (M,).
    """
    M = T.shape[0]
    out = np.full((M, len(bpts)), np.inf)
    for team in (0, 1):
        bi = bteam == team
        ti = tteam == team
        if not bi.any() or not ti.any():
            continue
        diff = T[:, ti, None, :] - bpts[bi][None, None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        d = np.where(np.isnan(d), np.inf, d)
        out[:, bi] = d.min(axis=1)
    return np.median(out, axis=1)


def _chamfer_pairs(B, bteam, T, tteam):
    """Per-pair score for aligned frame pairs: B (L, Kb, 2) against T (L, Kt, 2). Returns (L,)."""
    L = B.shape[0]
    out = np.full((L, B.shape[1]), np.nan)
    for team in (0, 1):
        bi = bteam == team
        ti = tteam == team
        if not bi.any() or not ti.any():
            continue
        diff = T[:, ti, None, :] - B[:, None, bi, :]
        d = np.sqrt((diff ** 2).sum(-1))
        d = np.where(np.isnan(d), np.inf, d)
        m = d.min(axis=1)
        m[np.isnan(B[:, bi, 0])] = np.nan
        out[:, bi] = m
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = np.nanmedian(np.where(np.isinf(out), 1e6, out), axis=1)
    return res


def _orient(xy, flip, pitch):
    if not flip:
        return xy
    out = xy.copy()
    out[..., 0] = pitch[0] - out[..., 0]
    out[..., 1] = pitch[1] - out[..., 1]
    return out


def segments(bxy, valid, p):
    """[(start, stop)] continuous shots of the broadcast, stop exclusive."""
    visible = np.isfinite(bxy[:, :, 0])
    ok = valid & (visible.sum(axis=1) >= p["min_players"])
    both = visible[1:] & visible[:-1]
    carried = both.sum(axis=1) / np.maximum(visible[1:].sum(axis=1), 1)
    disp = np.linalg.norm(bxy[1:] - bxy[:-1], axis=2)
    disp[~both] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        jump = np.nanmedian(disp, axis=1)
    cut = np.zeros(len(bxy), dtype=bool)
    cut[1:] = (carried < p["cut_continue"]) | (np.nan_to_num(jump, nan=0.0) > p["cut_jump_m"])
    out, start = [], None
    for b in range(len(bxy)):
        if not ok[b] or cut[b]:
            if start is not None and b - start >= p["min_segment"]:
                out.append((start, b))
            start = b if ok[b] else None
        elif start is None:
            start = b
    if start is not None and len(bxy) - start >= p["min_segment"]:
        out.append((start, len(bxy)))
    return out


def _coarse_best(bpts, bteam, Tc, tteam, cand):
    scores = _chamfer(bpts, bteam, Tc, tteam)
    i = int(np.argmin(scores))
    return int(cand[i]), float(scores[i])


def _frame_points(bxy, bteam, b):
    vis = np.isfinite(bxy[b, :, 0])
    return bxy[b, vis], bteam[vis]


def _votes(bxy, bteam, txy, tteam, segs, p, every=None):
    every = every or p["vote_every"]
    cand = np.arange(0, len(txy), p["coarse_stride"])
    Tc = txy[cand]
    samples = [b for s, e in segs for b in range(s + (e - s) // 2 % every, e, every)]
    votes, fits = [], []
    for b in samples:
        pts, tm = _frame_points(bxy, bteam, b)
        t, f = _coarse_best(pts, tm, Tc, tteam, cand)
        fits.append(f)
        if f <= p["vote_max_m"]:
            votes.append(t - b)
    return np.asarray(votes), np.asarray(fits)


def _fit_segment(bxy, bteam, txy, tteam, s, e, offset, p):
    """(best offset, median fit) for segment [s, e) searched at offset +/- fine_window."""
    n_t = len(txy)
    k = min(p["segment_samples"], e - s)
    bs = np.unique(np.linspace(s, e - 1, k).astype(int))
    deltas = np.arange(-p["fine_window"], p["fine_window"] + 1)
    C = np.full((len(bs), len(deltas)), np.inf)
    for i, b in enumerate(bs):
        t = b + offset + deltas
        inside = (t >= 0) & (t < n_t)
        if not inside.any():
            continue
        pts, tm = _frame_points(bxy, bteam, b)
        C[i, inside] = _chamfer(pts, tm, txy[t[inside]], tteam)
    med = np.median(C, axis=0)
    j = int(np.argmin(med))
    return int(offset + deltas[j]), float(med[j])


def align(tactical, bcast, bcast_valid, params=None):
    """Align `bcast` (a Tracking of the broadcast ATD) to `tactical`. Returns an Alignment."""
    p = dict(PARAMS, **(params or {}))
    pitch = tactical.meta.get("pitch", (105.3, 68.0))
    txy = np.asarray(tactical.players_xy, dtype=np.float64)
    tteam = _slot_team(tactical)
    bxy_raw = np.asarray(bcast.players_xy, dtype=np.float64)
    bteam_raw = _slot_team(bcast)
    valid = np.asarray(bcast_valid, dtype=bool)

    segs = segments(bxy_raw, valid, p)

    # Orientation: the convention that matches best on a small vote wins.
    best = None
    for flip in (False, True):
        for swap in (False, True):
            bxy = _orient(bxy_raw, flip, pitch)
            bteam = np.where(bteam_raw >= 0, (1 - bteam_raw) if swap else bteam_raw, bteam_raw)
            _, fits = _votes(bxy, bteam, txy, tteam, segs, p, every=p["vote_every"] * 10)
            score = float(np.median(fits)) if len(fits) else np.inf
            if best is None or score < best[0]:
                best = (score, flip, swap)
    _, flip, swap = best
    bxy = _orient(bxy_raw, flip, pitch)
    bteam = np.where(bteam_raw >= 0, (1 - bteam_raw) if swap else bteam_raw, bteam_raw)

    votes, _ = _votes(bxy, bteam, txy, tteam, segs, p)
    offsets = []
    if len(votes):
        bins = np.round(votes / p["offset_bin"]).astype(int)
        values, counts = np.unique(bins, return_counts=True)
        # Stable: ties in the vote count keep the np.unique order (ascending bin), on any numpy.
        order = np.argsort(-counts, kind="stable")
        for v, c in zip(values[order], counts[order]):
            if c < p["offset_min_share"] * len(votes):
                break
            # Merge adjacent bins into the mode already kept.
            if any(abs(v * p["offset_bin"] - o) <= p["offset_bin"] for o in offsets):
                continue
            members = votes[np.abs(bins - v) <= 1]
            offsets.append(int(np.median(members)))

    n_t = len(txy)
    b_of_t = np.full(n_t, -1, dtype=np.int64)
    covered = np.zeros(n_t, dtype=bool)
    seg_fit = {}
    live, pending = [], []
    for s, e in segs:
        fits = [_fit_segment(bxy, bteam, txy, tteam, s, e, o, p) for o in offsets]
        if fits:
            o, f = min(fits, key=lambda x: x[1])
            if f <= p["segment_max_m"]:
                live.append((s, e, o, f))
                continue
        pending.append((s, e))

    n_unmatched = 0
    cand = np.arange(0, n_t, p["coarse_stride"])
    Tc = txy[cand]
    for s, e in pending:
        mid = (s + e) // 2
        pts, tm = _frame_points(bxy, bteam, mid)
        t0, f0 = _coarse_best(pts, tm, Tc, tteam, cand)
        if f0 > p["vote_max_m"]:
            n_unmatched += 1
            continue
        o, f = _fit_segment(bxy, bteam, txy, tteam, s, e, t0 - mid, p)
        if f > p["segment_max_m"]:
            n_unmatched += 1
            continue
        live.append((s, e, o, f))

    # A replay shows a moment again, later in the broadcast: same tactical frames, smaller offset.
    # So segments are placed largest offset first, and one that lands mostly on frames already
    # covered is a replay. A replay of a moment the live feed never showed is kept: its
    # positions are those of that tactical frame all the same.
    n_replays = 0
    for s, e, o, f in sorted(live, key=lambda x: (-x[2], x[3])):
        b = np.arange(s, e)
        t = b + o
        inside = (t >= 0) & (t < n_t)
        b, t = b[inside], t[inside]
        if len(t) and covered[t].mean() >= p["replay_overlap"]:
            n_replays += 1
            continue
        fresh = ~covered[t]
        b_of_t[t[fresh]] = b[fresh]
        covered[t] = True
        seg_fit[(s, e)] = f

    fit_of_t = np.full(n_t, np.nan)
    t_idx = np.flatnonzero(b_of_t >= 0)
    for chunk in np.array_split(t_idx, max(1, len(t_idx) // 5000)):
        if len(chunk):
            fit_of_t[chunk] = _chamfer_pairs(bxy[b_of_t[chunk]], bteam, txy[chunk], tteam)

    stats = {
        "segments": len(segs),
        "segment_frames": int(sum(e - s for s, e in segs)),
        "live_segments": len(seg_fit),
        "replays_discarded": n_replays,
        "unmatched_segments": n_unmatched,
        "offsets": len(offsets),
        "aligned_tactical_frames": int(len(t_idx)),
        "median_segment_fit_m": float(np.median(list(seg_fit.values()))) if seg_fit else float("nan"),
    }
    return Alignment(b_of_t=b_of_t, fit_of_t=fit_of_t, orientation=(flip, swap),
                     offsets=offsets, stats=stats)


def _aligned_runs(b_of_t):
    """(N_t,) run id of each aligned tactical row: consecutive rows with one offset t - b; -1 off."""
    n = len(b_of_t)
    on = b_of_t >= 0
    off = np.where(on, np.arange(n) - b_of_t, np.iinfo(np.int64).min)
    new = np.ones(n, dtype=bool)
    new[1:] = (off[1:] != off[:-1]) | ~on[:-1]
    new &= on
    return np.where(on, np.cumsum(new) - 1, -1)


def _ball_speed(ball_xy, fps):
    """(N,) central-difference speed in m/s on the recording's own timeline, NaN at the edges."""
    v = np.full(len(ball_xy), np.nan)
    if len(ball_xy) > 2:
        v[1:-1] = np.linalg.norm(ball_xy[2:] - ball_xy[:-2], axis=1) * fps / 2
    return v


def _local_disagreement(t, xy, run, reference_xy, window, min_frames):
    """(len(t),) median distance to the reference ball over the reference frames of the same
    aligned run within +/- `window` rows of each candidate; NaN with fewer than `min_frames`."""
    out = np.full(len(t), np.nan)
    ref = np.isfinite(reference_xy[t]).all(axis=1)
    d = np.linalg.norm(xy - np.where(ref[:, None], reference_xy[t], np.nan), axis=1)
    r = run[t]
    for s in np.unique(r):
        idx = np.flatnonzero(r == s)                  # candidates of this run, t ascending
        rt, rd = t[idx[ref[idx]]], d[idx[ref[idx]]]
        if len(rt) < min_frames:
            continue
        lo = np.searchsorted(rt, t[idx] - window)
        hi = np.searchsorted(rt, t[idx] + window, side="right")
        for i, a, b in zip(idx, lo, hi):
            if b - a >= min_frames:
                out[i] = np.median(rd[a:b])
    return out


def ball_candidates(alignment, bcast_tracking, max_fit_m=2.0, frame0=0, max_speed_ms=None,
                    reference_xy=None, ref_window=125, ref_max_m=1.5, ref_min_frames=10,
                    return_stats=False):
    """B3 input from B8: the broadcast ball on the tactical timeline, one row per usable frame.

    DataFrame with frame (tactical video frame = tactical row + `frame0`), x_m, y_m, conf and
    source="broadcast". A tactical frame contributes when it has an aligned broadcast frame, the
    per-frame player fit is <= `max_fit_m` and the broadcast ATD has a ball there. The ball gets
    the orientation the alignment chose (flip_xy; swap_teams does not move the ball). conf falls
    linearly from 1 at a perfect fit to 0.5 at `max_fit_m`.

    Optional filters (off by default, so the defaults reproduce the plain candidates):

    - `max_speed_ms`: drop a broadcast ball whose own speed (central difference on the broadcast
      timeline) exceeds this. Fast balls are often in the air, and a ground-plane homography puts
      them metres off along the camera's depth axis.
    - `reference_xy`: (N_t, 2) the tactical ATD ball on tactical rows (our input data, NaN where
      absent). On a frame WITHOUT a reference ball, the candidate is dropped when, on the reference
      frames of the same aligned run within +/- `ref_window` rows, the median broadcast-to-
      reference distance exceeds `ref_max_m`: the broadcast ball tracker is off the ball in that
      stretch. With fewer than `ref_min_frames` such frames there is no evidence and the candidate
      stays. Candidates on frames that have a reference ball are never dropped by this filter, so
      the agree/conflict counts of fuse.augment_ball stay an independent check.

    With `return_stats=True` returns (DataFrame, dict of counts).
    """
    import pandas as pd

    b_of_t = np.asarray(alignment.b_of_t, dtype=np.int64)
    fit = np.asarray(alignment.fit_of_t, dtype=np.float64)
    pitch = bcast_tracking.meta.get("pitch", (105.3, 68.0))
    bball = _orient(np.asarray(bcast_tracking.ball_xy, dtype=np.float64),
                    bool(alignment.orientation[0]), pitch)
    t = np.flatnonzero((b_of_t >= 0) & np.isfinite(fit) & (fit <= max_fit_m))
    xy = bball[b_of_t[t]]
    has = np.isfinite(xy).all(axis=1)
    t, xy = t[has], xy[has]
    stats = {"candidates": int(len(t)), "dropped_speed": 0, "dropped_reference": 0,
             "reference_no_evidence": 0}
    if max_speed_ms is not None and len(t):
        v = _ball_speed(bball, float(bcast_tracking.meta.get("fps", 25)))[b_of_t[t]]
        fast = np.isfinite(v) & (v > max_speed_ms)
        stats["dropped_speed"] = int(fast.sum())
        t, xy = t[~fast], xy[~fast]
    if reference_xy is not None and len(t):
        ref = np.asarray(reference_xy, dtype=np.float64)
        if ref.shape != (len(b_of_t), 2):
            raise ValueError("reference_xy must be (N_tactical, 2), got {}".format(ref.shape))
        med = _local_disagreement(t, xy, _aligned_runs(b_of_t), ref, ref_window, ref_min_frames)
        open_ = ~np.isfinite(ref[t]).all(axis=1)
        bad = open_ & np.isfinite(med) & (med > ref_max_m)
        stats["dropped_reference"] = int(bad.sum())
        stats["reference_no_evidence"] = int((open_ & ~np.isfinite(med)).sum())
        t, xy = t[~bad], xy[~bad]
    stats["kept"] = int(len(t))
    f = fit[t]
    if max_fit_m > 0:
        conf = 1.0 - np.clip(f, 0.0, max_fit_m) / max_fit_m * 0.5
    else:
        conf = np.ones(len(t))
    df = pd.DataFrame({"frame": (t + int(frame0)).astype(np.int64), "x_m": xy[:, 0],
                       "y_m": xy[:, 1], "conf": conf, "source": "broadcast"})
    return (df, stats) if return_stats else df
