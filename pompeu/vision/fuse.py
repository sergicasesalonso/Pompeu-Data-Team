"""B3: ball tracker and fusion -- more ball frames in the same Tracking, so Line A runs unchanged.

    from pompeu.vision import fuse
    video = fuse.video_candidates_to_metres(detections_px, H)   # B2 top-k, pixels -> metres
    plus, report = fuse.augment_ball(tracking, [video, broadcast])
    plus.save(os.path.join(paths.CACHE_DIR, "atd_plus.npz"))    # config: tracking: cache/atd_plus.npz

The ATD loses the ball on ~43 % of the frames. The ball detector (B2) and the broadcast ATD aligned
to the tactical clock (B8) see some of those frames, but each candidate is a guess: a detector also
fires on heads, boots and the penalty spot. So a candidate only enters the Tracking when the
physics around it says it is the ball, and never where the ATD already has one.

1. Pool. Every candidate table (frame, x_m, y_m, conf, source) goes into one list; at most
   `max_per_frame` per frame are kept, highest conf first.
2. ATD wins. A frame the ATD observed keeps the ATD ball untouched. Candidates there are only
   counted: agreeing (within `agree_m`) or conflicting.
3. Gate to the ATD. A candidate must be reachable from the previous and from the next ATD ball at
   `max_speed_ms` (35 m/s, the hardest shots) plus `tol_m`. Far from the ATD in time the bound is
   loose, near it it is strict -- the reach grows with dt, as it should.
4. Support. Candidates (and the ATD balls next to them) are linked when they are at most
   `link_frames` apart and within reach of each other; the connected components are tracklets.
   A tracklet that spans fewer than `min_frames` distinct frames is noise: this is what rejects an
   isolated single-frame detection with nothing around it.
5. Choice. Per frame, the candidate of the strongest tracklet (sum of conf, ATD balls counting 1)
   wins; ties go to higher conf, then to the earlier source in the input list. Deterministic.
6. Consistency. On the merged track (ATD + chosen), a new point that is out of reach of both its
   neighbours is a spike and goes, until stable; then new-only fragments shorter than
   `min_frames` (split at gaps and jumps) go too. ATD points are never removed.
7. Gaps. A gap of at most `interp_max_s` (1.5 s) with an accepted ball on both sides, at least one
   of them new, is filled in a straight line if it can be crossed at `max_speed_ms`. Gaps between
   two ATD balls are left as A1 decided (0.5 s spline), unless `interp_new_only` is False.

Codes: any accepted candidate -- detector or broadcast -- is SRC_VIDEO with conf * `conf_scale`;
gap fills are SRC_INTERP with `interp_conf`. The report keeps the per-source counts. The contract
has no "broadcast" code yet; see the B3 section in ARQUITECTURA.md.

Everything is vectorised numpy and the same inputs give identical arrays. The report holds counts
only; the Tracking it returns is per-frame data and lives in cache/, never in git.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pompeu import paths  # noqa: F401  (makes games importable)
import games
from pompeu.features.kinematics import runs
from pompeu.io.contract import SRC_INTERP, SRC_NONE, SRC_VIDEO, Tracking

# Params
PARAMS = {
    "max_speed_ms": 35.0,     # physical bound on ball speed
    "tol_m": 1.0,             # position noise added to every reach test
    "agree_m": 1.0,           # a candidate this close to the ATD ball agrees with it
    "link_frames": 3,         # tracklet links span at most this many frames
    "min_frames": 3,          # a tracklet (or new-only fragment) needs this many distinct frames
    "max_per_frame": 8,       # candidates kept per frame, highest conf first
    "min_conf": 0.0,          # candidates at or below this conf are ignored
    "conf_scale": 0.8,        # ball_conf of an accepted candidate = conf * this
    "interp_max_s": 1.5,      # longest gap filled in a straight line
    "interp_conf": 0.5,       # ball_conf of a gap fill
    "interp_new_only": True,  # only fill gaps that touch a new observation
    "max_passes": 10,         # spike removal iterations
    "pitch_margin_m": 5.0,    # video_candidates_to_metres: keep balls this far outside the lines
}
CANDIDATE_COLUMNS = ("frame", "x_m", "y_m", "conf")
DEFAULT_SOURCE = "video"


# -- pixels -> metres ---------------------------------------------------------------------------

def video_candidates_to_metres(df_px, H, game=games.GAME, h_frame0=0, params=None):
    """Detector output (frame, x_px, y_px, conf; top-k per frame) -> candidates in metres.

    H is (N, 3, 3), normalised pitch -> pixels (starter/load_homography.py), row i = video frame
    h_frame0 + i. Rows whose matrix is missing or singular, or whose point lands more than
    `pitch_margin_m` outside the pitch, are dropped.
    """
    p = {**PARAMS, **(params or {})}
    length, width = games.GAMES[game]["pitch"]
    frame = np.asarray(df_px["frame"], dtype=np.int64)
    px = np.asarray(df_px["x_px"], dtype=np.float64)
    py = np.asarray(df_px["y_px"], dtype=np.float64)
    conf = np.asarray(df_px["conf"], dtype=np.float64)
    row = frame - int(h_frame0)
    ok = (row >= 0) & (row < len(H)) & np.isfinite(px) & np.isfinite(py) & np.isfinite(conf)

    x_m = np.full(len(frame), np.nan)
    y_m = np.full(len(frame), np.nan)
    urow, inverse = np.unique(row[ok], return_inverse=True)
    if len(urow):
        mats = np.asarray(H, dtype=np.float64)[urow]
        good = np.isfinite(mats).all(axis=(1, 2))
        good[good] = np.abs(np.linalg.det(mats[good])) > 1e-12
        inv = np.full_like(mats, np.nan)
        inv[good] = np.linalg.inv(mats[good])
        m = inv[inverse]
        a, b = px[ok], py[ok]
        den = m[:, 2, 0] * a + m[:, 2, 1] * b + m[:, 2, 2]
        x_m[ok] = (m[:, 0, 0] * a + m[:, 0, 1] * b + m[:, 0, 2]) / den * length
        y_m[ok] = (m[:, 1, 0] * a + m[:, 1, 1] * b + m[:, 1, 2]) / den * width

    margin = p["pitch_margin_m"]
    keep = (np.isfinite(x_m) & np.isfinite(y_m)
            & (x_m >= -margin) & (x_m <= length + margin)
            & (y_m >= -margin) & (y_m <= width + margin))
    out = pd.DataFrame({"frame": frame[keep], "x_m": x_m[keep], "y_m": y_m[keep],
                        "conf": conf[keep], "source": DEFAULT_SOURCE})
    out.attrs["source"] = DEFAULT_SOURCE
    return out


# -- helpers ------------------------------------------------------------------------------------

def _pool(candidates, frame0, n, p):
    """All candidates as arrays (row, x, y, conf, src), sorted by row then conf; source names."""
    names, parts = [], []
    for df in candidates:
        if df is None or not len(df):
            continue
        default = df.attrs.get("source", DEFAULT_SOURCE) if hasattr(df, "attrs") else DEFAULT_SOURCE
        src = (df["source"].astype(str).to_numpy() if "source" in df.columns
               else np.full(len(df), default, dtype=object))
        for name in pd.unique(src):
            if name not in names:
                names.append(name)
        code = pd.Series(src).map({name: i for i, name in enumerate(names)}).to_numpy(np.int64)
        parts.append((np.asarray(df["frame"], dtype=np.int64) - frame0,
                      np.asarray(df["x_m"], dtype=np.float64),
                      np.asarray(df["y_m"], dtype=np.float64),
                      np.asarray(df["conf"], dtype=np.float64), code))
    if not parts:
        empty = np.zeros(0)
        return (empty.astype(np.int64), empty, empty, empty, empty.astype(np.int64)), names
    row, x, y, conf, src = (np.concatenate(cols) for cols in zip(*parts))
    ok = ((row >= 0) & (row < n) & np.isfinite(x) & np.isfinite(y) & np.isfinite(conf)
          & (conf > p["min_conf"]))
    row, x, y, src = row[ok], x[ok], y[ok], src[ok]
    conf = np.clip(conf[ok], 0.0, 1.0)
    # Stable order: frame, then higher conf, then source order, then input order.
    order = np.lexsort((np.arange(len(row)), src, -conf, row))
    row, x, y, conf, src = row[order], x[order], y[order], conf[order], src[order]
    if len(row):
        _, first, counts = np.unique(row, return_index=True, return_counts=True)
        rank = np.arange(len(row)) - np.repeat(first, counts)
        keep = rank < p["max_per_frame"]
        row, x, y, conf, src = row[keep], x[keep], y[keep], conf[keep], src[keep]
    return (row, x, y, conf, src), names


def _reach(d, dt, fps, p):
    """True where a distance d (m) can be covered in dt frames."""
    return d <= p["max_speed_ms"] * dt / fps + p["tol_m"]


def _neighbour_rows(mask):
    """(previous, next) True row for every row; -1 / len(mask) where there is none."""
    n = len(mask)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(mask, idx, -1))
    nxt = np.minimum.accumulate(np.where(mask, idx, n)[::-1])[::-1]
    return prev, nxt


def _gate_to_anchor(row, x, y, anchor, ball_xy, fps, p):
    """Candidates reachable from the previous and the next anchor ball (strictly before/after)."""
    prev, nxt = _neighbour_rows(anchor)
    ok = np.ones(len(row), dtype=bool)
    before = prev[row]
    has = before >= 0
    d = np.hypot(x[has] - ball_xy[before[has], 0], y[has] - ball_xy[before[has], 1])
    ok[has] &= _reach(d, row[has] - before[has], fps, p)
    after = nxt[row]
    has = after < len(anchor)
    d = np.hypot(x[has] - ball_xy[after[has], 0], y[has] - ball_xy[after[has], 1])
    ok[has] &= _reach(d, after[has] - row[has], fps, p)
    return ok


def _tracklets(nrow, nx, ny, fps, p):
    """Component label of each node (nodes sorted by row); links within link_frames and reach."""
    m = len(nrow)
    lo = np.searchsorted(nrow, nrow + 1, side="left")
    hi = np.searchsorted(nrow, nrow + p["link_frames"], side="right")
    width = int((hi - lo).max()) if m else 0
    src, dst = [], []
    for j in range(width):
        a = np.flatnonzero(lo + j < hi)
        b = lo[a] + j
        d = np.hypot(nx[b] - nx[a], ny[b] - ny[a])
        linked = _reach(d, nrow[b] - nrow[a], fps, p)
        src.append(a[linked])
        dst.append(b[linked])
    src = np.concatenate(src) if src else np.zeros(0, dtype=np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(m, m)).tocsr()
    return connected_components(graph, directed=False)[1]


def _remove_spikes(xy, present, protected, fps, p):
    """Boolean (N,) of unprotected points out of reach of both observed neighbours."""
    removed = np.zeros(len(xy), dtype=bool)
    for _ in range(p["max_passes"]):
        index = np.flatnonzero(present & ~removed)
        if len(index) < 2:
            break
        step = np.hypot(*(xy[index[1:]] - xy[index[:-1]]).T)
        fast = ~_reach(step, np.diff(index), fps, p)
        before = np.concatenate(([False], fast))
        after = np.concatenate((fast, [False]))
        # At the ends of the track the one neighbour decides.
        before[0], after[-1] = after[0], before[-1]
        spikes = index[before & after & ~protected[index]]
        if not len(spikes):
            break
        removed[spikes] = True
    return removed


def _short_new_fragments(xy, present, new, fps, p):
    """Boolean (N,) of new points in new-only fragments shorter than min_frames."""
    removed = np.zeros(len(xy), dtype=bool)
    index = np.flatnonzero(present)
    if not len(index):
        return removed
    step = np.hypot(*(xy[index[1:]] - xy[index[:-1]]).T)
    breaks = (np.diff(index) > p["link_frames"]) | ~_reach(step, np.diff(index), fps, p)
    frag = np.concatenate(([0], np.cumsum(breaks)))
    size = np.bincount(frag)
    has_old = np.bincount(frag, weights=~new[index]) > 0
    bad = (size < p["min_frames"]) & ~has_old
    removed[index[bad[frag]]] = True
    return removed


def _expand(starts, stops):
    """(rows, run id) of every row in the half-open runs [starts, stops)."""
    sizes = stops - starts
    run = np.repeat(np.arange(len(starts)), sizes)
    offset = np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return (np.repeat(starts, sizes) + offset).astype(np.int64), run


def _coverage(present, graded):
    n = int(graded.sum())
    k = int((present & graded).sum())
    return k, (round(k / n, 4) if n else float("nan"))


# -- the fusion ---------------------------------------------------------------------------------

def augment_ball(tracking, candidates, H=None, params=None):
    """(Tracking with more ball frames, report of counts).

    `tracking` is the cleaned ATD (or any Tracking). `candidates` is a list of DataFrames with
    frame, x_m, y_m, conf and a source name (a `source` column, else df.attrs["source"], else
    "video"). `H` is accepted for symmetry with the pixel helper and not needed here: candidates
    arrive in metres.
    """
    del H
    p = {**PARAMS, **(params or {})}
    n = tracking.n_frames
    fps = float(tracking.meta.get("fps", 25))
    src_in = tracking.ball_source
    ball_xy = tracking.ball_xy
    anchor = (src_in != SRC_NONE) & (src_in != SRC_INTERP)     # observed, never touched

    (row, x, y, conf, src), names = _pool(candidates, tracking.frame0, n, p)
    n_in = len(row)

    # ATD wins on its own frames.
    on_anchor = anchor[row]
    d = np.hypot(x[on_anchor] - ball_xy[row[on_anchor], 0], y[on_anchor] - ball_xy[row[on_anchor], 1])
    n_agree, n_conflict = int((d <= p["agree_m"]).sum()), int((d > p["agree_m"]).sum())
    keep = ~on_anchor
    row, x, y, conf, src = row[keep], x[keep], y[keep], conf[keep], src[keep]

    # Gate to the ATD.
    gated = _gate_to_anchor(row, x, y, anchor, ball_xy, fps, p)
    n_gate = int((~gated).sum())
    row, x, y, conf, src = row[gated], x[gated], y[gated], conf[gated], src[gated]

    # Tracklets over the candidates plus the anchor balls next to them.
    near = np.zeros(n, dtype=bool)
    for shift in range(-p["link_frames"], p["link_frames"] + 1):
        near[np.clip(row + shift, 0, n - 1)] = True
    arow = np.flatnonzero(anchor & near)
    nrow = np.concatenate((row, arow))
    nx = np.concatenate((x, ball_xy[arow, 0]))
    ny = np.concatenate((y, ball_xy[arow, 1]))
    nconf = np.concatenate((conf, np.ones(len(arow))))
    is_cand = np.concatenate((np.ones(len(row), bool), np.zeros(len(arow), bool)))
    order = np.lexsort((np.arange(len(nrow)), nrow))
    nrow, nx, ny, nconf, is_cand = nrow[order], nx[order], ny[order], nconf[order], is_cand[order]
    cand_of_node = np.full(len(nrow), -1)
    cand_of_node[is_cand] = order[is_cand]          # node -> index into the candidate arrays

    if len(nrow):
        label = _tracklets(nrow, nx, ny, fps, p)
        pairs = np.unique(label.astype(np.int64) * (n + 1) + nrow)
        span = np.bincount(pairs // (n + 1), minlength=label.max() + 1)
        score = np.bincount(label, weights=nconf)
        supported = is_cand & (span[label] >= p["min_frames"])
    else:
        label, score = np.zeros(0, dtype=np.int64), np.zeros(0)
        supported = np.zeros(0, dtype=bool)
    n_unsupported = int((is_cand & ~supported).sum())

    # One candidate per frame: strongest tracklet, then conf, then source order.
    s = np.flatnonzero(supported)
    ci = cand_of_node[s]
    pick = np.lexsort((ci, src[ci], -conf[ci], -score[label[s]], row[ci]))
    s_ci = ci[pick]
    first = np.concatenate(([True], row[s_ci][1:] != row[s_ci][:-1])) if len(s_ci) else \
        np.zeros(0, dtype=bool)
    chosen = s_ci[first]
    n_lost = int(len(s_ci) - len(chosen))

    work = np.where(anchor[:, None], ball_xy, np.nan)
    new = np.zeros(n, dtype=bool)
    work[row[chosen]] = np.stack((x[chosen], y[chosen]), axis=1)
    new[row[chosen]] = True
    new_conf = np.zeros(n)
    new_conf[row[chosen]] = conf[chosen]
    new_src = np.full(n, -1)
    new_src[row[chosen]] = src[chosen]

    # Consistency on the merged track.
    spikes = _remove_spikes(work, anchor | new, anchor, fps, p)
    work[spikes], new[spikes] = np.nan, False
    frags = _short_new_fragments(work, anchor | new, new, fps, p)
    work[frags], new[frags] = np.nan, False

    # Output, starting from the input.
    out_xy = ball_xy.copy()
    out_z = tracking.ball_z.copy()
    out_conf = tracking.ball_conf.copy()
    out_src = src_in.copy()
    out_xy[new] = work[new]
    out_z[new] = np.nan
    out_conf[new] = np.clip(new_conf[new] * p["conf_scale"], 0.0, 1.0)
    out_src[new] = SRC_VIDEO

    # Gaps between accepted balls.
    present = anchor | new
    starts, stops = runs(~present)
    inner = (starts > 0) & (stops < n)
    starts, stops = starts[inner], stops[inner]
    touches = new[starts - 1] | new[stops]
    length = stops - starts
    a, b = work[starts - 1], work[stops]
    fillable = (length <= int(np.floor(p["interp_max_s"] * fps + 1e-9))) & \
        _reach(np.hypot(*(b - a).T), length + 1, fps, p)
    if p["interp_new_only"]:
        fillable &= touches
    # A1's old fills inside a gap that now touches a new ball no longer bridge the right points.
    stale = touches & ~fillable
    rows_stale, _ = _expand(starts[stale], stops[stale])
    rows_stale = rows_stale[src_in[rows_stale] == SRC_INTERP]
    out_xy[rows_stale], out_z[rows_stale] = np.nan, np.nan
    out_conf[rows_stale], out_src[rows_stale] = 0.0, SRC_NONE
    fs, fe = starts[fillable], stops[fillable]
    rows_fill, gap = _expand(fs, fe)
    n_interp = len(rows_fill)
    if n_interp:
        t = (rows_fill - (fs[gap] - 1)) / (fe[gap] - fs[gap] + 1)
        lo_xy, hi_xy = work[fs - 1][gap], work[fe][gap]
        out_xy[rows_fill] = lo_xy + t[:, None] * (hi_xy - lo_xy)
        out_z[rows_fill] = np.nan
        out_conf[rows_fill] = p["interp_conf"]
        out_src[rows_fill] = SRC_INTERP

    out = Tracking(players_xy=tracking.players_xy, players_team=tracking.players_team,
                   players_conf=tracking.players_conf, ball_xy=out_xy, ball_z=out_z,
                   ball_conf=out_conf, ball_source=out_src, frame0=tracking.frame0,
                   meta={**tracking.meta, "source": str(tracking.meta.get("source", "")) + "+b3"})

    graded = tracking.half() > 0
    before_present = tracking.ball_present
    after_present = out.ball_present
    cov_before, frac_before = _coverage(before_present, graded)
    cov_after, frac_after = _coverage(after_present, graded)
    accepted = {name: int((new_src[new] == i).sum()) for i, name in enumerate(names)}
    report = {
        "frames": n,
        "candidates_in": n_in,
        "frames_with_ball_before": int(before_present.sum()),
        "frames_with_ball_after": int(after_present.sum()),
        "accepted": int(new.sum()),
        "accepted_per_source": accepted,
        "on_atd_agree": n_agree,
        "on_atd_conflict": n_conflict,
        "rejected_gate_atd": n_gate,
        "rejected_unsupported": n_unsupported,
        "rejected_same_frame": n_lost,
        "rejected_spike": int(spikes.sum()),
        "rejected_fragment": int(frags.sum()),
        "interpolated": n_interp,
        "old_interp_cleared": int(len(rows_stale)),
        "graded_frames": int(graded.sum()),
        "graded_with_ball_before": cov_before,
        "graded_with_ball_after": cov_after,
        "graded_coverage_before": frac_before,
        "graded_coverage_after": frac_after,
    }
    return out, report
