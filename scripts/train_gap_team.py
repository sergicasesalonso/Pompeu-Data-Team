"""Fit the gap-team expert (pompeu.experts.gap_team) on external TRAIN, DFL only.

    python scripts/train_gap_team.py                    # writes configs/gap_team_v1.json

Data. The DFL matches of external `train` (val and test are never touched). Metrica is left out:
games 1/2 rebuild their ball from the events, so their own ball gaps hold no possession change
and would teach the model nothing about gaps. Each match is degraded exactly like the external
bench (degrade.PARAMS default, seed 1000*seed + index, coverage_sweep's log-normal gap lengths --
no BAR-ATH data is read), cleaned with A1 defaults, and its features built with the same
build_features call hsmm.prepare makes (collective group, float32), then
gap_team.design_matrix -- so training and inference compute the same numbers.

Target. In-half frames with no ball after cleaning whose ground truth (source labels through
external.align_labels at min_possession_s = 5, as the bench) is H or A; y = 1 for away. Every
`stride`-th such frame is used (consecutive frames are near copies).

Model. Ridge logistic regression by Newton's method on standardised features (NaN -> train mean).
Deterministic: fixed seeds, no other randomness. Only aggregates are printed, and the sha256 of
the JSON. The JSON is derived from public external data only, never from BAR-ATH.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.experts import gap_team  # noqa: E402
from pompeu.external import degrade, dfl, metrica  # noqa: E402
from pompeu.features.build import build_features  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
MIN_POSSESSION_S = 5.0
OUT = os.path.join(paths.CONFIGS_DIR, "gap_team_v1.json")


def present(name):
    """Loadable without downloading: raw files or the converted cache (POMPEU_CACHE_DIR)."""
    if external.available(name):
        return True
    module, key = external.parse(name)
    path = dfl._cache_path(key) if module is dfl else metrica._cache_path(key)
    return os.path.isfile(path)


def match_rows(name, index, seed, run_lengths, stride):
    """(X, y, n_gap_live) for one degraded match: gap frames with H/A ground truth."""
    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    gt = external.align_labels(np.where(half > 0, labels, pm.X).astype(np.int8),
                               MIN_POSSESSION_S, fps)
    degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
    cleaned, _ = clean_tracking(degraded, {})
    F = build_features(cleaned, tagging=None, with_windows=False, groups=("collective",))
    X = gap_team.design_matrix(F, fps)
    keep = (half > 0) & ~cleaned.ball_present & np.isin(gt, (pm.H, pm.A))
    rows = np.flatnonzero(keep)[::stride]
    return X[rows], (gt[rows] == pm.A).astype(float), int(keep.sum())


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stride", type=int, default=5)
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

    parts_X, parts_y = [], []
    for index, name in enumerate(matches):
        for seed in args.seeds:
            t0 = time.time()
            X, y, n_gap = match_rows(name, index, seed, run_lengths, args.stride)
            parts_X.append(X)
            parts_y.append(y)
            print("  {} seed {}: live gap frames {}, used {}, away share {:.3f} ({:.0f} s)".format(
                name, seed, n_gap, len(y), y.mean(), time.time() - t0), flush=True)
    X, y = np.concatenate(parts_X), np.concatenate(parts_y)
    Z, mean, std = gap_team.standardise(X)
    w, b = gap_team.fit_logistic(Z, y)
    logit = Z @ w + b
    acc = float(np.mean((logit > 0) == y))
    acc_noint = float(np.mean((Z @ w > 0) == y))

    model = {
        "name": "gap_team_v1",
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
            "min_possession_s": MIN_POSSESSION_S, "stride": args.stride,
            "n_frames": int(len(y)), "away_share": round(float(y.mean()), 6),
            "train_accuracy": round(acc, 6), "git": git_hash(),
        },
    }
    blob = (json.dumps(model, indent=1) + "\n").encode("utf-8")
    with open(args.out, "wb") as handle:
        handle.write(blob)
    print("fitted on {} frames, {} features: train acc {:.3f} (without intercept {:.3f}), "
          "intercept {:+.3f}".format(len(y), Z.shape[1], acc, acc_noint, b))
    print("wrote {}  sha256 {}".format(os.path.relpath(args.out, ROOT),
                                       hashlib.sha256(blob).hexdigest()))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
