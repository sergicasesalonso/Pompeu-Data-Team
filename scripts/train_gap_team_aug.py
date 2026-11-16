"""Fit the gap-team expert on external TRAIN with augmentation (configs/gap_team_v2.json).

    python scripts/train_gap_team_aug.py                 # writes configs/gap_team_v2.json
    python scripts/train_gap_team_aug.py --workers 1     # same JSON, serial

Same model, features and data pipeline as scripts/train_gap_team.py (v1): the DFL matches of
external `train` (val and test never touched), degraded like the bench, cleaned with A1 defaults,
build_features' collective group, gap_team.design_matrix, ridge logistic on standardised features.
Only the training set grows:

a) GEOMETRY. The collective features are measured in each team's own attacking frame
   (pompeu.features.collective via clean.normalize.attack_flip), so the valid symmetries are:

       identity
       mirror_y       y -> width - y (and vy -> -vy) for every player. Attack directions do not
                      change. Per team: cy -> width - cy, cvy -> -cvy; every other column is
                      invariant (widths, depths, lines, counts and speeds do not see the sign of y).
       swap           home <-> away: the players keep their positions and their attack direction
                      (home_attack_sign is negated with the labels), labels H <-> A. Per-team
                      columns swap _H <-> _A; field_speed_*, stopped_frac and centroid_gap
                      (= cx_H + cx_A - length) are invariant; line_gap = def_line_H - length +
                      front_A becomes def_line_A - length + front_H.
       swap_mirror_y  both.

   Mirroring x alone is NOT used: with the attack metadata kept, each team would attack its own
   goal (not a plausible match); with the attack metadata flipped as well, every attacking-frame
   feature is unchanged (x -> length - x twice), so it adds nothing. The 180-degree rotation
   (x and y) with flipped attack metadata reduces to mirror_y. The transforms are exact on the
   per-frame columns and are applied before the centred rolling means (which are linear), so
   design_matrix sees exactly the features of the transformed match (tests/test_gap_team_aug.py
   checks this against collective_features on a transformed synthetic Tracking).

b) SEEDS. Degradation seeds 0..9 (v1: 0..2); seed 1000*seed + index as the bench.

c) LABEL CONVENTIONS. Ground truth aligned at min_possession_s 2, 5 and 8 s (v1: 5 only). The
   three copies share the same features, so they are merged exactly: a frame enters if any
   convention says H or A there; its target is the mean of the conventions' away indicators and
   its weight the number of conventions that say H or A (the weighted log-likelihood equals the
   sum over the three copies).

Deterministic (fixed seeds; worker results are joined in submission order). Only aggregates are
printed, and the sha256 of the JSON. Derived from public external data only, never from BAR-ATH.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths  # noqa: E402
from pompeu.experts import gap_team  # noqa: E402

# Params
MIN_POSSESSION_S = (2.0, 5.0, 8.0)
SEEDS = tuple(range(10))
STRIDE = 5
VARIANTS = ("identity", "mirror_y", "swap", "swap_mirror_y")
OUT = os.path.join(paths.CONFIGS_DIR, "gap_team_v2.json")

TEAM_NAMES = ("n_vis", "cx", "cy", "cvx", "cvy", "width", "depth", "def_line", "front",
              "compact", "n_recover", "n_press", "speed_mean")


# -- augmentation (pure, tests/test_gap_team_aug.py) -------------------------------------------

def mirror_y(F, width):
    """Copy of the collective columns of F for the match mirrored across the long axis."""
    out = F.astype(np.float64, copy=True)
    for team in ("H", "A"):
        if "cy_" + team in out:
            out["cy_" + team] = width - out["cy_" + team]
        if "cvy_" + team in out:
            out["cvy_" + team] = -out["cvy_" + team]
    return out


def swap_teams(F):
    """Copy of the collective columns of F with home and away exchanged (labels H <-> A)."""
    out = F.astype(np.float64, copy=True)
    for name in TEAM_NAMES:
        h, a = name + "_H", name + "_A"
        if h in F and a in F:
            out[h], out[a] = F[a].astype(np.float64), F[h].astype(np.float64)
    need = ("line_gap", "def_line_H", "def_line_A", "front_H", "front_A")
    if all(c in F for c in need):
        f = F.astype(np.float64)
        # line_gap = def_line_H - length + front_A  ->  def_line_A - length + front_H
        out["line_gap"] = f["line_gap"] - f["def_line_H"] - f["front_A"] + f["def_line_A"] + f["front_H"]
    return out


def augment(F, width, variant):
    """(F', swapped) for one of VARIANTS."""
    if variant == "identity":
        return F.astype(np.float64, copy=True), False
    if variant == "mirror_y":
        return mirror_y(F, width), False
    if variant == "swap":
        return swap_teams(F), True
    if variant == "swap_mirror_y":
        return swap_teams(mirror_y(F, width)), True
    raise ValueError(variant)


def merge_conventions(gts, H, A):
    """(keep, ybar, weight) over label conventions: frames where any says H/A, the mean away
    indicator over the conventions that do, and how many do."""
    gts = np.asarray(gts)
    live = np.isin(gts, (H, A))
    count = live.sum(axis=0)
    away = ((gts == A) & live).sum(axis=0)
    keep = count > 0
    ybar = np.where(keep, away / np.maximum(count, 1), 0.0)
    return keep, ybar, count.astype(np.float64)


def fit_logistic_weighted(Z, y, weight, ridge=gap_team.RIDGE, iters=50, tol=1e-8):
    """gap_team.fit_logistic with per-row weights and fractional targets y in [0, 1]."""
    n, d = Z.shape
    Xb = np.column_stack([Z, np.ones(n)])
    beta = np.zeros(d + 1)
    penalty = np.full(d + 1, ridge)
    penalty[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        grad = Xb.T @ (weight * (p - y)) + penalty * beta
        hess = (Xb * (weight * p * (1 - p))[:, None]).T @ Xb + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[:-1], beta[-1]


# -- data --------------------------------------------------------------------------------------

def match_rows(job):
    """(X, y, weight, variant id, n_live) for one degraded match, every variant stacked."""
    name, index, seed, run_lengths, stride = job
    from pompeu.clean import clean_tracking
    from pompeu.external import degrade
    from pompeu.features.build import build_features
    import possession_metrics as pm

    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    width = float(tracking.meta["pitch"][1])
    half = tracking.half()
    graded = np.where(half > 0, labels, pm.X).astype(np.int8)
    gts = [external.align_labels(graded, v, fps) for v in MIN_POSSESSION_S]
    degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
    cleaned, _ = clean_tracking(degraded, {})
    F = build_features(cleaned, tagging=None, with_windows=False, groups=("collective",))
    live, ybar, weight = merge_conventions(gts, pm.H, pm.A)
    keep = (half > 0) & ~cleaned.ball_present & live
    rows = np.flatnonzero(keep)[::stride]
    Xs, ys, ws, vs = [], [], [], []
    for k, variant in enumerate(VARIANTS):
        Fv, swapped = augment(F, width, variant)
        Xs.append(gap_team.design_matrix(Fv, fps)[rows])
        ys.append(1.0 - ybar[rows] if swapped else ybar[rows])
        ws.append(weight[rows])
        vs.append(np.full(len(rows), k, np.int8))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(vs),
            int(keep.sum()))


def present(name):
    from pompeu.external import dfl, metrica
    if external.available(name):
        return True
    module, key = external.parse(name)
    path = dfl._cache_path(key) if module is dfl else metrica._cache_path(key)
    return os.path.isfile(path)


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    started = time.time()

    import coverage_sweep
    run_lengths = coverage_sweep.synthetic_run_lengths()
    names = [external.canonical(m) for m in external.SPLITS["train"]]       # never val/test
    names = [m for m in names if m.startswith("dfl:")]
    matches = [m for m in names if present(m)]
    missing = sorted(set(names) - set(matches))
    if missing:
        sys.exit("train matches not present locally: {}".format(missing))

    jobs = [(name, index, seed, run_lengths, args.stride)
            for index, name in enumerate(matches) for seed in args.seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(match_rows, jobs))                    # submission order
    else:
        results = [match_rows(job) for job in jobs]
    for job, res in zip(jobs, results):
        print("  {} seed {}: live gap frames {}, rows {} ({} variants)".format(
            job[0], job[2], res[4], len(res[1]), len(VARIANTS)), flush=True)

    X = np.concatenate([r[0] for r in results])
    y = np.concatenate([r[1] for r in results])
    weight = np.concatenate([r[2] for r in results])
    variant = np.concatenate([r[3] for r in results])
    del results
    Z, mean, std = gap_team.standardise(X)
    del X
    w, b = fit_logistic_weighted(Z, y, weight)
    logit = Z @ w
    hard = y >= 0.5
    acc = float(np.average(((logit + b) > 0) == hard, weights=weight))
    acc_noint = float(np.average((logit > 0) == hard, weights=weight))

    model = {
        "name": "gap_team_v2",
        "kind": "logistic",
        "target": "P(away has the ball) on in-half frames without ball",
        "base_columns": list(gap_team.BASE_COLUMNS),
        "window_s": gap_team.WINDOW_S,
        "features": gap_team.feature_names(),
        "mean": [round(float(v), 8) for v in mean],
        "std": [round(float(v), 8) for v in std],
        "coef": [round(float(v), 8) for v in w],
        "intercept": round(float(b), 8),
        "ridge": gap_team.RIDGE,
        "training": {
            "matches": matches, "split": "train", "source": "dfl (public, IDSSE/DFL figshare)",
            "seeds": args.seeds, "degrade": "degrade.PARAMS default, seed 1000*seed + index",
            "run_lengths": "coverage_sweep.synthetic_run_lengths()",
            "min_possession_s": list(MIN_POSSESSION_S), "stride": args.stride,
            "n_frames": int(len(y)), "n_frames_per_variant": int(len(y) // len(VARIANTS)),
            "weight_sum": round(float(weight.sum()), 3),
            "away_share": round(float(np.average(y, weights=weight)), 6),
            "train_accuracy": round(acc, 6), "git": git_hash(),
            "augmentation": {
                "variants": list(VARIANTS),
                "geometry": "per-team attacking-frame features: mirror_y (cy -> width - cy, "
                            "cvy -> -cvy), swap home/away (team columns swapped, line_gap "
                            "recomputed, labels H<->A); x mirroring is degenerate (identity with "
                            "attack meta flipped) or implausible (without), so not used",
                "label_conventions": "min_possession_s 2/5/8 merged: target = mean away "
                                     "indicator, weight = number of conventions with H/A",
                "fit": "weighted ridge logistic (train_gap_team_aug.fit_logistic_weighted)",
            },
        },
    }
    blob = (json.dumps(model, indent=1) + "\n").encode("utf-8")
    with open(args.out, "wb") as handle:
        handle.write(blob)
    print("fitted on {} rows ({} per variant), {} features, weight sum {:.0f}: train acc {:.3f} "
          "(without intercept {:.3f}), intercept {:+.3f}".format(
              len(y), len(y) // len(VARIANTS), Z.shape[1], weight.sum(), acc, acc_noint, b))
    for k, name in enumerate(VARIANTS):
        m = variant == k
        print("  {:<14} train acc without intercept {:.3f}".format(
            name, np.average((logit[m] > 0) == hard[m], weights=weight[m])))
    print("wrote {}  sha256 {}".format(os.path.relpath(args.out, ROOT),
                                       hashlib.sha256(blob).hexdigest()))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
