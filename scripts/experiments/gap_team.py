"""Experiment: can ball-free collective features tell who has the ball inside ball gaps?

    python scripts/experiments/gap_team.py                  # fit on train, report on val, seeds 0-2
    python scripts/experiments/gap_team.py --seeds 0 --stride 10

QUESTION. On ~41% of graded frames the ball is missing; our models then mostly keep the last
possessor, so changes of possession inside a gap are missed. Before anyone builds an HSMM "team
evidence" expert, this measures on external labelled data whether team shape and motion (A2's
collective group: centroids, centroid velocity, lines, width/depth, pressing/recovering counts,
speeds -- nothing that needs the ball) say more about the possessor than carrying the last team.

HOW. External `train` matches fit, `val` matches report; `test` is never touched. Each match is
degraded exactly like the bench (degrade.PARAMS, seed 1000*seed + index, coverage_sweep's
log-normal gap lengths) and cleaned with A1 defaults as the pipeline does. Ground truth: the
source labels inside the halves through external.align_labels(min_possession_s=5), as the bench.

    gap        a maximal run of in-half frames with no ball after cleaning. Gaps that open a half
               have no "last visible" frame and are left out (counted in the report).
    target     gap frames whose ground truth is H or A.
    carry      the true H/A label at the last visible frame before the gap (the last H/A label at
               or before it): an UPPER bound for what our models do, they must infer it.
    nearest    the team of the player nearest the ball at the last visible frame (oracle-free).
    collective logistic regression on the collective columns plus their 3 s centred means.
    coll+carry the same plus the carried team (+/-1) and its interaction with log(1 + gap age).
    coll+near  the same with the nearest-team carry instead (what a real model would have).
    *_s1/_s2   the logit smoothed with a centred 1 s / 2 s mean inside the gap.

The logistic regression is a ridge-penalised Newton fit in numpy on standardised features (NaN ->
0 after standardising, i.e. the train mean). Everything is seeded; nothing is printed per frame.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import degrade, dfl, metrica  # noqa: E402
from pompeu.features import collective, kinematics  # noqa: E402

# Params
H, A, D, X = 0, 1, 2, 3
MIN_POSSESSION_S = 5.0
WINDOW_S = 3.0                      # centred mean added to every collective column
RIDGE = 1.0                         # L2 on standardised coefficients (not on the intercept)
GAP_BUCKETS_S = (0, 2, 5, 10, 30, np.inf)
AGE_BUCKETS_S = (0, 2, 5, 10, 30, np.inf)
SMOOTH_S = (1.0, 2.0)


# -- pure logic (tests/test_gap_team.py) -------------------------------------------------------

def gaps(ball_present, half):
    """(starts, stops) of maximal runs of in-half frames without ball, never crossing halves."""
    starts, stops = [], []
    for h in np.unique(half[half > 0]):
        s, e = kinematics.runs((half == h) & ~ball_present)
        starts.append(s)
        stops.append(e)
    if not starts:
        return np.zeros(0, int), np.zeros(0, int)
    starts, stops = np.concatenate(starts), np.concatenate(stops)
    order = np.argsort(starts)
    return starts[order], stops[order]


def last_team(gt, half, frame):
    """Last H/A label at or before `frame` within the same half, or -1."""
    h = half[frame]
    for t in range(frame, -1, -1):
        if half[t] != h:
            return -1
        if gt[t] in (H, A):
            return int(gt[t])
    return -1


def gap_table(ball_present, half, gt, nearest):
    """Per-frame arrays over the whole match: gap id (-1 outside), age (frames since gap start),
    gap length (frames), carry (true), near (nearest-team carry). Gaps opening a half: carry -1."""
    n = len(gt)
    gid = np.full(n, -1)
    age = np.zeros(n)
    length = np.zeros(n)
    carry = np.full(n, -1)
    near = np.full(n, -1)
    starts, stops = gaps(ball_present, half)
    for k, (s, e) in enumerate(zip(starts, stops)):
        gid[s:e] = k
        age[s:e] = np.arange(e - s)
        length[s:e] = e - s
        if s > 0 and half[s - 1] == half[s] and ball_present[s - 1]:
            carry[s:e] = last_team(gt, half, s - 1)
            near[s:e] = nearest[s - 1]
    return gid, age, length, carry, near


def standardise(X, mean=None, std=None):
    """(Z, mean, std); NaN -> 0 after standardising, constant columns -> std 1."""
    if mean is None:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where(~(std > 1e-9), 1.0, std)
    Z = (X - mean) / std
    return np.where(np.isnan(Z), 0.0, Z), mean, std


def fit_logistic(Z, y, ridge=RIDGE, iters=50, tol=1e-8):
    """(w, b) of a ridge-penalised logistic regression by Newton's method. y in {0, 1}."""
    n, d = Z.shape
    Xb = np.column_stack([Z, np.ones(n)])
    beta = np.zeros(d + 1)
    penalty = np.full(d + 1, ridge)
    penalty[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        grad = Xb.T @ (p - y) + penalty * beta
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[:-1], beta[-1]


def smooth_in_gaps(values, gid, window):
    """Centred rolling mean of `values` inside each gap (min 1 value), window in frames."""
    out = values.astype(float).copy()
    inside = gid >= 0
    if window <= 1 or not inside.any():
        return out
    s = pd.Series(values[inside])
    out[inside] = s.groupby(gid[inside]).transform(
        lambda v: v.rolling(int(window), center=True, min_periods=1).mean()).to_numpy()
    return out


def bucket(values, edges):
    """Index of the half-open bucket [edges[i], edges[i+1]) for each value."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def change_masks(gid, gt, carry):
    """(changed_gap per frame, post_change per frame): the gap contains an H/A frame whose team is
    not the carried one; post_change = such frames themselves."""
    target = (gid >= 0) & np.isin(gt, (H, A)) & (carry >= 0)
    other = target & (gt != carry)
    changed_ids = np.unique(gid[other])
    changed = target & np.isin(gid, changed_ids)
    return changed, other


# -- data --------------------------------------------------------------------------------------

def present(name):
    if external.available(name):
        return True
    module, key = external.parse(name)
    path = dfl._cache_path(key) if module is dfl else metrica._cache_path(key)
    return os.path.isfile(path)


def nearest_team(tracking):
    """(N,) team (0/1) of the observed player nearest the ball, -1 without ball or players."""
    xy = np.where((tracking.players_conf > 0)[..., None], tracking.players_xy, np.nan)
    d = np.hypot(xy[..., 0] - tracking.ball_xy[:, None, 0], xy[..., 1] - tracking.ball_xy[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    team = np.take_along_axis(tracking.players_team, k[:, None], axis=1)[:, 0].astype(int)
    return np.where(np.isfinite(d.min(axis=1)), team, -1)


def match_rows(name, index, seed, run_lengths):
    """Per-frame arrays for the gap frames of one degraded match (features, labels, gap info)."""
    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    gt = external.align_labels(np.where(half > 0, labels, X).astype(np.int8), MIN_POSSESSION_S, fps)
    degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
    cleaned, _ = clean_tracking(degraded, {})
    players = kinematics.player_kinematics(cleaned)
    cols = collective.collective_features(cleaned, players["vel"], players["speed"])
    frame = pd.DataFrame(cols)
    rolled = frame.rolling(int(WINDOW_S * fps), center=True, min_periods=1).mean()
    rolled.columns = [c + "_m3s" for c in frame.columns]
    frame = pd.concat([frame, rolled], axis=1)
    gid, age, length, carry, near = gap_table(cleaned.ball_present, half, gt, nearest_team(cleaned))
    keep = gid >= 0
    n_opening = int(np.sum(keep & (carry < 0) & np.isin(gt, (H, A))))
    keep &= np.isin(gt, (H, A)) & (carry >= 0)
    changed, post = change_masks(gid, gt, carry)
    uid = np.where(gid >= 0, gid + 1_000_000 * (1000 * seed + index), -1)
    return dict(
        X=frame.to_numpy(np.float64)[keep], columns=list(frame.columns),
        y=(gt[keep] == A).astype(float), gid=uid[keep], age=age[keep] / fps,
        length=length[keep] / fps, carry=carry[keep], near=near[keep],
        changed=changed[keep], post=post[keep],
        source=np.full(int(keep.sum()), name.split(":")[0]), match=np.full(int(keep.sum()), name),
        n_opening=n_opening, n_graded_live=int(np.sum(np.isin(gt, (H, A)))),
        n_gap_live=int(np.sum((gid >= 0) & np.isin(gt, (H, A)))))


def collect(names, seeds, run_lengths, verbose=True):
    parts = []
    for index, name in enumerate(names):
        for seed in seeds:
            started = time.time()
            parts.append(match_rows(name, index, seed, run_lengths))
            if verbose:
                p = parts[-1]
                print("  {} seed {}: live {}, live in gaps {} ({:.0%}), usable {}, opening-half "
                      "{}  ({:.0f} s)".format(name, seed, p["n_graded_live"], p["n_gap_live"],
                                              p["n_gap_live"] / max(p["n_graded_live"], 1),
                                              len(p["y"]), p["n_opening"], time.time() - started),
                      flush=True)
    out = {k: np.concatenate([p[k] for p in parts]) for k in
           ("X", "y", "gid", "age", "length", "carry", "near", "changed", "post", "source", "match")}
    out["columns"] = parts[0]["columns"]
    return out


def design(data, kind):
    """Feature matrix for a model kind: collective, coll+carry, coll+near."""
    if kind == "collective":
        return data["X"], list(data["columns"])
    team = data["carry"] if kind == "coll+carry" else data["near"]
    sign = np.where(team == A, 1.0, np.where(team == H, -1.0, 0.0))
    extra = np.column_stack([sign, sign * np.log1p(data["age"])])
    return np.column_stack([data["X"], extra]), list(data["columns"]) + ["carry", "carry_x_logage"]


# -- report ------------------------------------------------------------------------------------

def accuracy(pred, y, mask):
    n = int(mask.sum())
    return (float(np.mean(pred[mask] == y[mask])) if n else float("nan")), n


def table(title, preds, y, groups, labels):
    lines = [title, "  {:<16}".format("") + "".join("{:>12}".format(l) for l in labels)]
    lines.append("  {:<16}".format("frames") + "".join(
        "{:>12}".format(int(g.sum())) for g in groups))
    for name, pred in preds.items():
        lines.append("  {:<16}".format(name) + "".join(
            "{:>12.3f}".format(accuracy(pred, y, g)[0]) for g in groups))
    return "\n".join(lines)


def edge_label(edges, i):
    hi = edges[i + 1]
    return "<{:g}s".format(hi) if i == 0 else (">{:g}s".format(edges[i]) if np.isinf(hi)
                                               else "{:g}-{:g}s".format(edges[i], hi))


def per_gap_recovery(pred, data):
    """Share of changed gaps where the method gives the post-change team on > half of its
    post-change frames."""
    ids = np.unique(data["gid"][data["post"]])
    hits = 0
    for g in ids:
        m = data["post"] & (data["gid"] == g)
        hits += np.mean(pred[m] == data["y"][m]) > 0.5
    return hits / max(len(ids), 1), len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stride", type=int, default=5, help="train on every n-th gap frame")
    args = parser.parse_args()
    started = time.time()

    import coverage_sweep
    run_lengths = coverage_sweep.synthetic_run_lengths()
    splits = {}
    for split in ("train", "val"):                                   # never "test"
        names = [external.canonical(m) for m in external.SPLITS[split]]
        splits[split] = [m for m in names if present(m)]
        for m in names:
            if m not in splits[split]:
                print("skipped {} ({}): not present locally".format(m, split))
    print("train {}\nval   {}\nseeds {}".format(splits["train"], splits["val"], args.seeds))
    print("collecting train")
    train = collect(splits["train"], args.seeds, run_lengths)
    print("collecting val")
    val = collect(splits["val"], args.seeds, run_lengths)

    y = val["y"]
    preds = {"carry": (val["carry"] == A).astype(float),
             "nearest": (val["near"] == A).astype(float)}
    coefs = {}
    gid_val = val["gid"]
    for kind in ("collective", "coll+carry", "coll+near"):
        Xtr, cols = design(train, kind)
        Ztr, mean, std = standardise(Xtr[::args.stride])
        w, b = fit_logistic(Ztr, train["y"][::args.stride])
        coefs[kind] = sorted(zip(cols, w), key=lambda t: -abs(t[1]))
        Zva, _, _ = standardise(design(val, kind)[0], mean, std)
        logit = Zva @ w + b
        preds[kind] = (logit > 0).astype(float)
        for s in SMOOTH_S:
            preds["{}_s{:g}".format(kind, s)] = (
                smooth_in_gaps(logit, gid_val, s * 25) > 0).astype(float)
        train_acc = np.mean(((Ztr @ w + b) > 0) == train["y"][::args.stride])
        print("fitted {:<11} on {} frames, {} features, train acc {:.3f}".format(
            kind, len(Ztr), Ztr.shape[1], train_acc))

    every = np.ones(len(y), bool)
    print()
    print(table("VAL accuracy on gap frames (live, H/A ground truth, carry defined)",
                preds, y, [every, val["source"] == "dfl", val["source"] == "metrica"],
                ["all", "dfl", "metrica"]))
    print()
    gb = bucket(val["length"], np.array(GAP_BUCKETS_S, float))
    print(table("by gap length", preds, y, [gb == i for i in range(len(GAP_BUCKETS_S) - 1)],
                [edge_label(GAP_BUCKETS_S, i) for i in range(len(GAP_BUCKETS_S) - 1)]))
    print()
    ab = bucket(val["age"], np.array(AGE_BUCKETS_S, float))
    print(table("by time since gap start", preds, y, [ab == i for i in range(len(AGE_BUCKETS_S) - 1)],
                [edge_label(AGE_BUCKETS_S, i) for i in range(len(AGE_BUCKETS_S) - 1)]))
    print()
    changed, post = val["changed"], val["post"]
    print(table("possession changes inside the gap", preds, y,
                [~changed, changed, post, changed & ~post, post & (val["source"] == "dfl")],
                ["no-chg gap", "chg gap", "post-chg", "chg pre", "post dfl"]))
    print()
    print("changed gaps where the method names the post-change team on > half of its "
          "post-change frames")
    for name, pred in preds.items():
        share, n = per_gap_recovery(pred, val)
        print("  {:<16}{:>8.3f}   (of {} changed gaps)".format(name, share, n))
    print()
    print("share of target frames in changed gaps {:.3f}; post-change {:.3f}; nearest == carry "
          "{:.3f}".format(changed.mean(), post.mean(), np.mean(val["near"] == val["carry"])))
    print()
    for kind, ranked in coefs.items():
        print("top standardised coefficients, {} (+ = away has it)".format(kind))
        for c, v in ranked[:12]:
            print("  {:<26}{:+.3f}".format(c, v))
    print("\nruntime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
