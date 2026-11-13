"""Compare fitted gap-team models (JSON) on external VAL gap frames, with gap_team.py's metrics.

    python scripts/experiments/gap_team_compare.py                       # v1 vs v2, seeds 0-4
    python scripts/experiments/gap_team_compare.py --models configs/gap_team_v1.json --seeds 0

Only `val` matches (never `test`). Each (match, seed) is degraded and cleaned exactly like
scripts/experiments/gap_team.py and the trainers (seed 1000*seed + index, coverage_sweep's
log-normal gap lengths, A1 defaults); the features come from build_features' collective group
and gap_team.away_logit, i.e. what the HSMM expert sees. Prediction = logit > 0 WITHOUT the
intercept (as gap_team.team_evidence); the "+b" rows add the intercept for reference.

Ground truth aligned at min_possession_s 2, 5 and 8 s (gaps, carry and changes are recomputed
for each). Reported per source, since metrica:2's gaps hold no possession change. Only
aggregates are printed.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.experts import gap_team as expert  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.features.build import build_features  # noqa: E402
import gap_team as gt_exp  # noqa: E402  (scripts/experiments/gap_team.py)

# Params
VALUES = (2.0, 5.0, 8.0)
MODELS = ("configs/gap_team_v1.json", "configs/gap_team_v2.json")


def collect(matches, seeds, models, run_lengths):
    """{value: data dict as gap_team.py's, with one prediction per model}."""
    parts = {v: [] for v in VALUES}
    for index, name in enumerate(matches):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        half = tracking.half()
        graded = np.where(half > 0, labels, gt_exp.X).astype(np.int8)
        truths = {v: external.align_labels(graded, v, fps) for v in VALUES}
        for seed in seeds:
            started = time.time()
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            cleaned, _ = clean_tracking(degraded, {})
            F = build_features(cleaned, tagging=None, with_windows=False, groups=("collective",))
            logits = {m: expert.away_logit(F, fps, expert.load_model(m)) for m in models}
            near_all = gt_exp.nearest_team(cleaned)
            for v, gt in truths.items():
                gid, age, length, carry, near = gt_exp.gap_table(cleaned.ball_present, half, gt,
                                                                  near_all)
                keep = (gid >= 0) & np.isin(gt, (gt_exp.H, gt_exp.A)) & (carry >= 0)
                changed, post = gt_exp.change_masks(gid, gt, carry)
                uid = np.where(gid >= 0, gid + 1_000_000 * (1000 * seed + index), -1)
                p = dict(y=(gt[keep] == gt_exp.A).astype(float), gid=uid[keep],
                         length=length[keep] / fps, carry=carry[keep], near=near[keep],
                         changed=changed[keep], post=post[keep],
                         source=np.full(int(keep.sum()), name.split(":")[0]))
                for m in models:
                    model = expert.load_model(m)
                    lg = np.nan_to_num(logits[m][keep])
                    p["pred:" + m] = (lg > 0).astype(float)
                    p["pred+b:" + m] = ((lg + model["intercept"]) > 0).astype(float)
                parts[v].append(p)
            print("  {} seed {} ({:.0f} s)".format(name, seed, time.time() - started), flush=True)
    return {v: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]} for v, ps in parts.items()}


def label(m):
    return os.path.splitext(os.path.basename(m))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    started = time.time()

    import coverage_sweep
    run_lengths = coverage_sweep.synthetic_run_lengths()
    matches = [external.canonical(m) for m in external.SPLITS["val"]]        # never "test"
    matches = [m for m in matches if gt_exp.present(m)]
    print("val {}  seeds {}".format(matches, args.seeds))
    data = collect(matches, args.seeds, args.models, run_lengths)

    for v in VALUES:
        d = data[v]
        preds = {"carry": (d["carry"] == gt_exp.A).astype(float),
                 "nearest": (d["near"] == gt_exp.A).astype(float)}
        for m in args.models:
            preds[label(m)] = d["pred:" + m]
            preds[label(m) + " +b"] = d["pred+b:" + m]
        y, dfl_m, met_m = d["y"], d["source"] == "dfl", d["source"] == "metrica"
        print("\n==== ground truth at min_possession_s = {:g} s ====".format(v))
        print(gt_exp.table("accuracy on gap frames (H/A truth, carry defined)", preds, y,
                           [dfl_m, met_m, dfl_m & d["changed"], dfl_m & d["post"]],
                           ["dfl", "metrica", "dfl chg gap", "dfl post-chg"]))
        gb = gt_exp.bucket(d["length"], np.array(gt_exp.GAP_BUCKETS_S, float))
        nb = len(gt_exp.GAP_BUCKETS_S) - 1
        print(gt_exp.table("dfl, by gap length", preds, y, [dfl_m & (gb == i) for i in range(nb)],
                           [gt_exp.edge_label(gt_exp.GAP_BUCKETS_S, i) for i in range(nb)]))
        sub = {k: val[dfl_m] for k, val in d.items()}
        print("dfl changed gaps recovered (post-change team on > half of post-change frames)")
        for name, pred in preds.items():
            share, n = gt_exp.per_gap_recovery(pred[dfl_m], sub)
            print("  {:<22}{:>8.3f}   (of {} changed gaps)".format(name, share, n))
    print("\nruntime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
